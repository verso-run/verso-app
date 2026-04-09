"""
Fast loop generator for sub-second click-to-route.

Uses batch Dijkstra on the ExpandedGraph to avoid ProcessPoolExecutor
overhead. One outbound Dijkstra covers all anchors; return legs are
individual Dijkstra calls with reuse penalty.

Typical latency: 300-800ms for 8 results (vs 31s with ProcessPoolExecutor).
"""

import math
import time

import numpy as np
from scipy.spatial import KDTree

from router.astar_rx import ExpandedGraph, N_BUCKETS
from router.loops import _route_metrics, REUSE_PENALTY, DISTANCE_TOLERANCE

# Anchor counts (fewer than loops.py — quality is marginal beyond this)
N_LOOP_ANCHORS = 24
N_OAB_ANCHORS = 8

# Turn angle threshold for counting turns
TURN_ANGLE_THRESHOLD = 60


class FastLoopGenerator:
    """Single-process loop generator using batch Dijkstra.

    Pre-builds KDTree and node arrays at init time (~100ms).
    generate() runs in <1s per request.
    """

    def __init__(self, G, eg: ExpandedGraph):
        self._G = G
        self._eg = eg

        # Build coordinate arrays and KDTree
        nodes = list(G.nodes())
        n = len(nodes)
        self._nodes = np.array(nodes, dtype=np.int64)
        self._lats = np.zeros(n, dtype=np.float64)
        self._lons = np.zeros(n, dtype=np.float64)
        self._quality = np.zeros(n, dtype=np.float64)

        # Node index: orig_node -> position in arrays
        self._node_idx = {}

        for i, node in enumerate(nodes):
            data = G.nodes[node]
            self._lats[i] = data["y"]
            self._lons[i] = data["x"]
            self._node_idx[node] = i

            # Max composite_score of incident edges
            best = 0.0
            for _, _, edata in G.edges(node, data=True):
                s = edata.get("composite_score", 0)
                if isinstance(s, (int, float)) and s > best:
                    best = s
            self._quality[i] = best

        # Build expanded-index lookup: (n_nodes, N_BUCKETS)
        # _exp_idx[i, b] = expanded index for nodes[i] at bucket b
        self._exp_idx = np.full((n, N_BUCKETS), -1, dtype=np.int64)
        nm = eg.node_map
        for i, node in enumerate(nodes):
            for b in range(N_BUCKETS):
                idx = nm.get((node, b), -1)
                if idx >= 0:
                    self._exp_idx[i, b] = idx

        # KDTree on (lat, lon) with cos(lat) correction
        mid_lat = np.median(self._lats)
        self._cos_lat = math.cos(math.radians(mid_lat))
        coords = np.column_stack([self._lats, self._lons * self._cos_lat])
        self._kdtree = KDTree(coords)

        # Main component nodes (set for fast lookup)
        import networkx as nx
        components = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
        self._main_comp = components[0]

        print(f"[loops_fast] Initialized: {n} nodes, KDTree built")

    def nearest_node(self, lat: float, lon: float) -> int:
        """Find nearest graph node via KDTree. O(log n)."""
        _, idx = self._kdtree.query([lat, lon * self._cos_lat])
        return int(self._nodes[idx])

    def generate(
        self,
        lat: float,
        lon: float,
        target_km: float,
        n_results: int = 8,
        terrain: str = "moderate",
    ) -> list[dict]:
        """Generate loop routes from a click point.

        Returns list of route dicts sorted by score, each containing:
            path, distance_km, metrics, coords
        """
        t0 = time.perf_counter()
        eg = self._eg
        G = self._G

        # 1. Find start node
        start = self.nearest_node(lat, lon)
        if start not in self._main_comp:
            # Fall back to brute force within main component
            from router.astar import nearest_node as nn_slow
            start = nn_slow(G, lat, lon, self._main_comp)

        # Cost budget: running_cost for half the target distance
        # running_cost ≈ length / composite_score; typical composite ~5
        # So half_km / 5 ≈ cost. Use 3x safety margin for limit.
        half_km = target_km / 2.0
        cost_limit = half_km * 0.6  # generous limit

        # 2. Batch outbound Dijkstra
        dist_arr, pred_arr = eg.batch_dijkstra_from(start, limit=cost_limit, terrain=terrain)

        try:
            # 3. Vectorized anchor selection
            min_cost, anchors_loop, anchors_oab = self._select_anchors(
                dist_arr, start, target_km)

            all_routes = []

            # 4. Full-loop routes
            for anchor in anchors_loop:
                route = self._build_full_loop(
                    dist_arr, pred_arr, start, anchor, target_km, terrain)
                if route:
                    all_routes.append(route)

            # 5. Lollipop routes (stem + loop)
            for anchor in anchors_loop[:12]:
                route = self._build_lollipop(
                    dist_arr, pred_arr, start, anchor, target_km, terrain)
                if route:
                    all_routes.append(route)

            # 6. Out-and-back routes
            for anchor in anchors_oab:
                route = self._build_oab(dist_arr, pred_arr, start, anchor, target_km)
                if route:
                    all_routes.append(route)

        finally:
            eg.restore_after_batch()

        # 7. Rank and dedup
        all_routes.sort(key=lambda r: r["metrics"]["score"], reverse=True)
        kept = self._dedup(all_routes)

        elapsed = time.perf_counter() - t0
        print(f"[loops_fast] {len(kept)} routes in {elapsed*1000:.0f}ms "
              f"(from {len(all_routes)} candidates)")

        return kept[:n_results]

    def _select_anchors(
        self,
        dist_arr: np.ndarray,
        start: int,
        target_km: float,
    ) -> tuple[np.ndarray, list[int], list[int]]:
        """Vectorized anchor selection from batch Dijkstra distances.

        Returns (min_cost_array, loop_anchors, oab_anchors).
        """
        eg = self._eg
        n = len(self._nodes)

        # Compute min cost across 8 buckets for each node (vectorized)
        exp_idx = self._exp_idx  # (n, 8)
        # Gather distances; -1 indices get inf
        valid = exp_idx >= 0
        # Use advanced indexing: flatten, gather, reshape
        flat_idx = np.where(valid, exp_idx, 0).ravel()
        flat_dist = dist_arr[flat_idx].reshape(n, N_BUCKETS)
        flat_dist[~valid] = np.inf
        min_cost = np.min(flat_dist, axis=1)

        # Target cost for half loop (outbound leg)
        half_km = target_km / 2.0
        # Estimate cost from distance: running_cost ≈ length / composite
        # Median composite ≈ 4-6, so cost ≈ km * 0.15 to km * 0.25
        target_cost = np.median(min_cost[np.isfinite(min_cost) & (min_cost > 0)])
        if np.isnan(target_cost) or target_cost <= 0:
            target_cost = half_km * 0.2

        # Scale: find cost that corresponds to ~half_km distance
        # Use the ratio of target_km/2 to typical cost
        # We'll select nodes in a band around target_cost
        cost_band_lo = target_cost * 0.3
        cost_band_hi = target_cost * 2.5

        # Filter: in cost band, in main component, reachable
        in_band = (min_cost >= cost_band_lo) & (min_cost <= cost_band_hi) & np.isfinite(min_cost)

        # Main component filter
        mc_mask = np.array([int(self._nodes[i]) in self._main_comp for i in range(n)], dtype=bool)
        eligible = in_band & mc_mask

        if eligible.sum() < 10:
            # Widen band
            cost_band_lo *= 0.3
            cost_band_hi *= 2.0
            in_band = (min_cost >= cost_band_lo) & (min_cost <= cost_band_hi) & np.isfinite(min_cost)
            eligible = in_band & mc_mask

        eligible_idx = np.where(eligible)[0]
        if len(eligible_idx) == 0:
            return min_cost, [], []

        # Compass octant diversification
        start_lat, start_lon = self._lats[self._node_idx.get(start, 0)], self._lons[self._node_idx.get(start, 0)]
        dlat = self._lats[eligible_idx] - start_lat
        dlon = self._lons[eligible_idx] - start_lon
        angles = np.degrees(np.arctan2(dlon, dlat)) % 360
        octants = (angles / 45).astype(int) % 8

        # Sort eligible by quality within each octant
        quality = self._quality[eligible_idx]

        loop_anchors = []
        oab_anchors = []
        per_octant = max(1, N_LOOP_ANCHORS // 8)
        oab_per_octant = max(1, N_OAB_ANCHORS // 8)

        for oct in range(8):
            mask = octants == oct
            if not mask.any():
                continue
            oct_idx = eligible_idx[mask]
            oct_quality = quality[mask]
            oct_cost = min_cost[oct_idx]

            # Sort by quality descending
            order = np.argsort(-oct_quality)
            oct_idx = oct_idx[order]
            oct_cost = oct_cost[order]

            # Loop anchors: prefer mid-range cost (farther from start)
            for j in range(min(per_octant, len(oct_idx))):
                loop_anchors.append(int(self._nodes[oct_idx[j]]))

            # OAB anchors: prefer highest quality
            for j in range(min(oab_per_octant, len(oct_idx))):
                oab_anchors.append(int(self._nodes[oct_idx[j]]))

        return min_cost, loop_anchors, oab_anchors

    def _build_full_loop(
        self,
        dist_arr: np.ndarray,
        pred_arr: np.ndarray,
        start: int,
        anchor: int,
        target_km: float,
        terrain: str,
    ) -> dict | None:
        """Build a full loop: outbound (from batch) + penalized return."""
        eg = self._eg
        G = self._G
        min_dist = target_km * (1 - DISTANCE_TOLERANCE)
        max_dist = target_km * (1 + DISTANCE_TOLERANCE)

        # Outbound: reconstruct from batch
        out_path, out_cost, out_km = eg.reconstruct_path(pred_arr, dist_arr, anchor)
        if not out_path or out_km < 0.5:
            return None

        # Return leg: individual Dijkstra with reuse penalty
        used = {(out_path[j], out_path[j + 1]) for j in range(len(out_path) - 1)}
        ret_path, ret_cost, ret_km = eg.route(
            anchor, start, used_edges=used, reuse_penalty=REUSE_PENALTY,
            terrain=terrain, limit=out_cost * 3)
        if not ret_path:
            return None

        full_path = out_path + ret_path[1:]
        total_dist = out_km + ret_km

        if not (min_dist <= total_dist <= max_dist):
            return None

        out_set = set(out_path)
        ret_set = set(ret_path)
        overlap = len(out_set & ret_set) / max(1, min(len(out_set), len(ret_set)))

        return self._score(full_path, total_dist, target_km, "loop", overlap)

    def _build_lollipop(
        self,
        dist_arr: np.ndarray,
        pred_arr: np.ndarray,
        start: int,
        stem_end: int,
        target_km: float,
        terrain: str,
    ) -> dict | None:
        """Build lollipop: stem out → loop → stem back."""
        eg = self._eg
        G = self._G
        min_dist = target_km * (1 - DISTANCE_TOLERANCE)
        max_dist = target_km * (1 + DISTANCE_TOLERANCE)

        # Stem: reconstruct from batch
        stem_path, stem_cost, stem_km = eg.reconstruct_path(pred_arr, dist_arr, stem_end)
        if not stem_path or stem_km < 0.3:
            return None

        # Find a loop anchor further from start (different direction)
        stem_lat = self._G.nodes[stem_end]["y"]
        stem_lon = self._G.nodes[stem_end]["x"]
        start_lat = self._G.nodes[start]["y"]
        start_lon = self._G.nodes[start]["x"]

        # Desired loop radius
        loop_budget = target_km - stem_km * 2
        if loop_budget < 1.0:
            return None

        loop_radius_km = loop_budget * 0.25  # crow-flies for loop anchor

        # Find closest node to stem_end that's roughly loop_radius away
        # Use KDTree query_ball
        deg_radius = loop_radius_km / (111.32 * self._cos_lat)
        candidates = self._kdtree.query_ball_point(
            [stem_lat, stem_lon * self._cos_lat],
            r=deg_radius * 2)
        if not candidates:
            return None

        # Pick highest quality candidate at appropriate distance
        best_anchor = None
        best_q = -1
        for ci in candidates[:50]:  # limit search
            node = int(self._nodes[ci])
            if node == stem_end or node == start:
                continue
            if node not in self._main_comp:
                continue
            d = self._haversine(stem_lat, stem_lon,
                                self._lats[ci], self._lons[ci])
            if loop_radius_km * 0.3 <= d <= loop_radius_km * 2.0:
                q = self._quality[ci]
                if q > best_q:
                    best_q = q
                    best_anchor = node

        if best_anchor is None:
            return None

        # Loop: stem_end → anchor → stem_end with reuse penalty
        loop_out, _, lout_km = eg.route(stem_end, best_anchor, terrain=terrain)
        if not loop_out:
            return None

        used = {(loop_out[j], loop_out[j + 1]) for j in range(len(loop_out) - 1)}
        loop_ret, _, lret_km = eg.route(
            best_anchor, stem_end, used_edges=used,
            reuse_penalty=REUSE_PENALTY, terrain=terrain)
        if not loop_ret:
            return None

        ret_stem = list(reversed(stem_path))
        full_path = stem_path + loop_out[1:] + loop_ret[1:] + ret_stem[1:]
        total_dist = stem_km + lout_km + lret_km + stem_km

        if not (min_dist <= total_dist <= max_dist):
            return None

        loop_overlap = (len(set(loop_out) & set(loop_ret)) /
                        max(1, min(len(loop_out), len(loop_ret))))

        return self._score(full_path, total_dist, target_km, "lollipop", loop_overlap)

    def _build_oab(
        self,
        dist_arr: np.ndarray,
        pred_arr: np.ndarray,
        start: int,
        anchor: int,
        target_km: float,
    ) -> dict | None:
        """Build out-and-back: outbound from batch, reverse for return."""
        eg = self._eg
        min_dist = target_km * (1 - DISTANCE_TOLERANCE)
        max_dist = target_km * (1 + DISTANCE_TOLERANCE)

        out_path, out_cost, out_km = eg.reconstruct_path(pred_arr, dist_arr, anchor)
        if not out_path:
            return None

        total_dist = out_km * 2
        if not (min_dist <= total_dist <= max_dist):
            return None

        ret_path = list(reversed(out_path))
        full_path = out_path + ret_path[1:]

        return self._score(full_path, total_dist, target_km, "out-and-back", 1.0)

    def _score(
        self,
        path: list[int],
        dist_km: float,
        target_km: float,
        shape: str,
        overlap: float,
    ) -> dict | None:
        """Score a route and return a route dict."""
        G = self._G
        metrics = _route_metrics(G, path, dist_km)
        if not metrics:
            return None

        # Turn penalty in ranking
        if metrics["turns_per_km"] > 8:
            metrics["score"] *= 0.6
        elif metrics["turns_per_km"] > 5:
            metrics["score"] *= 0.8

        # Distance accuracy bonus
        dist_error = abs(dist_km - target_km) / target_km
        metrics["score"] *= (1.0 - dist_error * 0.5)

        metrics["overlap"] = round(overlap, 2)
        metrics["distance_km"] = round(dist_km, 2)
        metrics["shape"] = shape

        coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in path]

        return {
            "path": path,
            "distance_km": round(dist_km, 2),
            "metrics": metrics,
            "coords": coords,
        }

    def _dedup(self, routes: list[dict]) -> list[dict]:
        """Deduplicate routes by coordinate overlap."""
        kept = []
        for r in routes:
            r_set = set((round(c[0], 5), round(c[1], 5)) for c in r["coords"][::3])
            is_dup = False
            for k in kept:
                k_set = set((round(c[0], 5), round(c[1], 5)) for c in k["coords"][::3])
                overlap = len(r_set & k_set) / max(1, min(len(r_set), len(k_set)))
                if overlap > 0.7:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(r)
        return kept

    @staticmethod
    def _haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dlon / 2) ** 2)
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

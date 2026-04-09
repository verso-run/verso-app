"""
Loop route generator: create running loops from a start point.

Three strategies:
1. **Lollipop**: stem out → loop → stem back (same stem both ways)
2. **Out-and-back**: route to a scenic destination and return same way
3. **Full loop**: outbound + penalized return (classic, may zigzag)

Strategies run in parallel using ProcessPoolExecutor for ~3-4x speedup.
Falls back to sequential execution if multiprocessing fails.
"""

import math
import os
import pickle as pkl
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import networkx as nx

from router.astar import route_astar, nearest_node, nearest_edge_node, get_main_component, _haversine_km

# Tuning constants
N_ANCHORS = 48           # total anchor candidates per strategy
REUSE_PENALTY = 2.0      # cost multiplier for loop return on same edges
DISTANCE_TOLERANCE = 0.30 # accept routes within +/-30% of target
TURN_ANGLE_THRESHOLD = 60 # degrees — sharper than this counts as a turn

# Loop closure constants
CLOSE_RADIUS_M = 500     # crow-flies distance to detect near-loops
MIN_GAP_EDGES = 10       # minimum path gap between candidate nodes
MAX_STITCH_KM = 0.8      # maximum stitch distance

# Parallel routing
_POC8_ROOT = str(Path(__file__).resolve().parent.parent)

# ── Pool worker state ────────────────────────────────────────────────
_pool_G = None


def _pool_init(graph_path, poc8_root):
    """Worker initializer: set up sys.path and load graph."""
    global _pool_G
    import sys
    if poc8_root not in sys.path:
        sys.path.insert(0, poc8_root)
    with open(graph_path, "rb") as f:
        _pool_G = pkl.load(f)


# ── Pool worker functions (must be module-level for pickling) ────────

def _pool_lollipop_stem(args):
    """Route one lollipop stem + its loop anchors in a worker."""
    (stem_anchor, start_node, loop_dist, main_comp,
     target_km, min_dist, max_dist, terrain) = args
    G = _pool_G
    stem_path, stem_cost, stem_km = route_astar(G, start_node, stem_anchor, terrain=terrain)
    if not stem_path or stem_km < 0.3:
        return []

    loop_anchors = _sample_anchors(G, stem_anchor, loop_dist, main_comp, n_anchors=8)
    routes = []
    for loop_anchor in loop_anchors:
        loop_out, _, lout_km = route_astar(G, stem_anchor, loop_anchor, terrain=terrain)
        if not loop_out:
            continue
        used = {(loop_out[j], loop_out[j + 1]) for j in range(len(loop_out) - 1)}
        loop_ret, _, lret_km = route_astar(
            G, loop_anchor, stem_anchor,
            used_edges=used, reuse_penalty=REUSE_PENALTY, terrain=terrain)
        if not loop_ret:
            continue

        ret_stem_path = list(reversed(stem_path))
        full_path = stem_path + loop_out[1:] + loop_ret[1:] + ret_stem_path[1:]
        total_dist = stem_km + lout_km + lret_km + stem_km

        loop_out_set = set(loop_out)
        loop_ret_set = set(loop_ret)
        loop_overlap = len(loop_out_set & loop_ret_set) / max(1, min(len(loop_out_set), len(loop_ret_set)))

        route = _score_and_filter(
            G, full_path, total_dist, target_km,
            min_dist, max_dist, shape="lollipop", overlap=loop_overlap)
        if route:
            routes.append(route)
    return routes


def _pool_full_loop(args):
    """Route one full-loop anchor in a worker."""
    anchor, start_node, target_km, min_dist, max_dist, terrain = args
    G = _pool_G
    out_path, _, out_dist = route_astar(G, start_node, anchor, terrain=terrain)
    if not out_path:
        return None
    used = {(out_path[j], out_path[j + 1]) for j in range(len(out_path) - 1)}
    ret_path, _, ret_dist = route_astar(
        G, anchor, start_node,
        used_edges=used, reuse_penalty=REUSE_PENALTY, terrain=terrain)
    if not ret_path:
        return None
    full_path = out_path + ret_path[1:]
    total_dist = out_dist + ret_dist
    out_set = set(out_path)
    ret_set = set(ret_path)
    overlap = len(out_set & ret_set) / max(1, min(len(out_set), len(ret_set)))
    return _score_and_filter(
        G, full_path, total_dist, target_km,
        min_dist, max_dist, shape="loop", overlap=overlap)


def _pool_oab(args):
    """Route one out-and-back anchor in a worker."""
    anchor, start_node, target_km, min_dist, max_dist, terrain = args
    G = _pool_G
    out_path, _, out_km = route_astar(G, start_node, anchor, terrain=terrain)
    if not out_path:
        return None
    ret_path = list(reversed(out_path))
    full_path = out_path + ret_path[1:]
    total_dist = out_km * 2
    return _score_and_filter(
        G, full_path, total_dist, target_km,
        min_dist, max_dist, shape="out-and-back", overlap=1.0)


# ── Shared helpers ───────────────────────────────────────────────────

def _node_quality(G: nx.MultiDiGraph, node: int) -> float:
    """Max composite_score of edges incident to this node."""
    scores = []
    for _, _, data in G.edges(node, data=True):
        s = data.get("composite_score", 0)
        if isinstance(s, (int, float)):
            scores.append(s)
    return max(scores) if scores else 0.0


def _sample_anchors(
    G: nx.MultiDiGraph,
    start_node: int,
    target_dist_km: float,
    node_set: set | None = None,
    n_anchors: int = N_ANCHORS,
) -> list[int]:
    """
    Sample anchor nodes at approximately target_dist_km from start,
    prioritizing nodes with high composite_score.
    """
    start_lat = G.nodes[start_node]["y"]
    start_lon = G.nodes[start_node]["x"]

    min_d = target_dist_km * 0.35
    max_d = target_dist_km * 1.70

    candidates = []
    iter_nodes = node_set if node_set is not None else G.nodes()
    for node in iter_nodes:
        data = G.nodes[node]
        d = _haversine_km(start_lat, start_lon, data["y"], data["x"])
        if min_d <= d <= max_d:
            quality = _node_quality(G, node)
            candidates.append((quality, d, node))

    if not candidates:
        return []

    candidates.sort(reverse=True)

    # Divide into 8 compass octants for directional diversity
    octants = {i: [] for i in range(8)}
    for quality, d, node in candidates:
        lat = G.nodes[node]["y"]
        lon = G.nodes[node]["x"]
        angle = math.degrees(math.atan2(lon - start_lon, lat - start_lat)) % 360
        octant = int(angle / 45) % 8
        octants[octant].append(node)

    anchors = []
    per_oct = max(1, n_anchors // 8)
    for oct_nodes in octants.values():
        anchors.extend(oct_nodes[:per_oct])

    if len(anchors) < n_anchors:
        anchor_set = set(anchors)
        for quality, d, node in candidates:
            if node not in anchor_set:
                anchors.append(node)
                anchor_set.add(node)
                if len(anchors) >= n_anchors:
                    break

    return anchors[:n_anchors]


def _route_metrics(
    G: nx.MultiDiGraph,
    path: list[int],
    distance_km: float,
) -> dict:
    """Compute quality metrics for a route path."""
    if len(path) < 2:
        return {}

    footpath_count = 0
    road_count = 0
    total_edges = 0
    composite_scores = []
    turn_count = 0
    bearings = []

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if v not in G[u]:
            continue
        edge_data = min(
            G[u][v].values(),
            key=lambda d: d.get("running_cost", float("inf")),
        )

        total_edges += 1
        hw = edge_data.get("highway", "")
        if isinstance(hw, list):
            hw = hw[0]

        if hw in ("footway", "path", "pedestrian", "cycleway", "steps", "track", "bridleway", "corridor"):
            footpath_count += 1
        elif hw in ("residential", "service", "living_street"):
            pass
        elif hw in ("tertiary", "tertiary_link", "secondary", "secondary_link",
                     "primary", "primary_link", "trunk", "trunk_link", "busway"):
            road_count += 1

        cs = edge_data.get("composite_score", 0)
        if isinstance(cs, (int, float)):
            composite_scores.append(cs)

        bearing = edge_data.get("bearing", 0)
        bearings.append(bearing)

    # Count turns
    for i in range(1, len(bearings)):
        diff = abs(bearings[i] - bearings[i - 1])
        if diff > 180:
            diff = 360 - diff
        if diff > TURN_ANGLE_THRESHOLD:
            turn_count += 1

    footpath_frac = footpath_count / total_edges if total_edges else 0
    road_frac = road_count / total_edges if total_edges else 0
    avg_composite = sum(composite_scores) / len(composite_scores) if composite_scores else 0
    turns_per_km = turn_count / distance_km if distance_km > 0 else 0

    score = avg_composite * 0.4 + footpath_frac * 10 * 0.4 - road_frac * 10 * 0.2

    return {
        "footpath_frac": round(footpath_frac, 3),
        "road_frac": round(road_frac, 3),
        "avg_composite": round(avg_composite, 2),
        "turn_count": turn_count,
        "turns_per_km": round(turns_per_km, 1),
        "score": round(score, 3),
        "edge_count": total_edges,
    }


def _close_near_loops(
    G: nx.MultiDiGraph,
    path: list[int],
) -> tuple[list[int], float]:
    """Detect and stitch near-loops to eliminate U-turn backtracks.

    Uses a spatial grid to efficiently find path nodes that revisit an area
    already passed (within CLOSE_RADIUS_M crow-flies). When a bearing
    reversal exists between the two positions, attempts a short A* stitch
    to close the loop and cut out the detour.

    Returns:
        (new_path, distance_saved_km) — original path if no closure found.
    """
    if len(path) < MIN_GAP_EDGES * 2:
        return path, 0.0

    cos_lat = math.cos(math.radians(G.nodes[path[0]]["y"]))
    # Grid cell size in degrees (~500m)
    cell_deg = CLOSE_RADIUS_M / (111320.0 * cos_lat)

    # Pre-compute positions and bearings
    positions = []  # (lat, lon) per path index
    for node in path:
        d = G.nodes[node]
        positions.append((d["y"], d["x"]))

    bearings = []
    for k in range(len(path) - 1):
        u, v = path[k], path[k + 1]
        if v in G[u]:
            ed = min(G[u][v].values(), key=lambda d: d.get("running_cost", float("inf")))
            bearings.append(ed.get("bearing", 0.0))
        else:
            bearings.append(0.0)

    # Build spatial grid: cell → list of path indices
    grid: dict[tuple[int, int], list[int]] = {}
    for idx, (lat, lon) in enumerate(positions):
        cx = int(lon / cell_deg)
        cy = int(lat / cell_deg)
        grid.setdefault((cx, cy), []).append(idx)

    best_saving = 0.0
    best_j = -1
    best_i = -1
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * cos_lat

    for i in range(MIN_GAP_EDGES, len(path)):
        lat_i, lon_i = positions[i]
        cx = int(lon_i / cell_deg)
        cy = int(lat_i / cell_deg)

        # Check neighboring grid cells
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cell = (cx + dx, cy + dy)
                if cell not in grid:
                    continue
                for j in grid[cell]:
                    if j > i - MIN_GAP_EDGES:
                        continue

                    lat_j, lon_j = positions[j]
                    dlat = (lat_i - lat_j) * m_per_deg_lat
                    dlon = (lon_i - lon_j) * m_per_deg_lon
                    dist_m = math.sqrt(dlat * dlat + dlon * dlon)
                    if dist_m > CLOSE_RADIUS_M:
                        continue

                    # Check for bearing reversal (U-turn) somewhere in path[j:i]
                    has_reversal = False
                    for k in range(j + 1, min(i, len(bearings))):
                        diff = abs(bearings[k] - bearings[k - 1])
                        if diff > 180:
                            diff = 360 - diff
                        if diff > 140:
                            has_reversal = True
                            break
                    if not has_reversal:
                        continue

                    # Compute detour distance along path[j..i]
                    detour_km = 0.0
                    for k in range(j, i):
                        u, v = path[k], path[k + 1]
                        if v in G[u]:
                            ed = min(G[u][v].values(), key=lambda d: d.get("running_cost", float("inf")))
                            detour_km += ed.get("length", 50) / 1000
                        else:
                            detour_km += 0.05

                    if detour_km > best_saving:
                        best_saving = detour_km
                        best_j = j
                        best_i = i

    if best_j < 0 or best_saving < 0.1:
        return path, 0.0

    # Route A* stitch from path[best_j] → path[best_i] (no reuse penalty)
    stitch_path, stitch_cost, stitch_km = route_astar(
        G, path[best_j], path[best_i],
        reuse_penalty=1.0,
        turn_penalty=True,
    )

    if not stitch_path or stitch_km > MAX_STITCH_KM:
        return path, 0.0

    if stitch_km >= best_saving:
        return path, 0.0

    new_path = path[:best_j] + stitch_path + path[best_i + 1:]
    saved = best_saving - stitch_km

    return new_path, saved


def _score_and_filter(
    G: nx.MultiDiGraph,
    full_path: list[int],
    total_dist: float,
    target_km: float,
    min_dist: float,
    max_dist: float,
    shape: str,
    overlap: float = 0.0,
) -> dict | None:
    """Score a route and return a route dict, or None if filtered out."""
    # Quick distance pre-check (closure can only shorten, so reject if too short)
    if total_dist < min_dist:
        return None

    # Attempt loop closure before scoring (only if route is within range)
    if min_dist <= total_dist <= max_dist:
        closed_path, saved_km = _close_near_loops(G, full_path)
        if saved_km > 0:
            full_path = closed_path
            total_dist -= saved_km

    if not (min_dist <= total_dist <= max_dist):
        return None

    metrics = _route_metrics(G, full_path, total_dist)
    if not metrics:
        return None

    # Turn penalty in ranking
    if metrics["turns_per_km"] > 8:
        metrics["score"] *= 0.6
    elif metrics["turns_per_km"] > 5:
        metrics["score"] *= 0.8

    # Distance accuracy bonus
    dist_error = abs(total_dist - target_km) / target_km
    metrics["score"] *= (1.0 - dist_error * 0.5)

    metrics["overlap"] = round(overlap, 2)
    metrics["distance_km"] = round(total_dist, 2)
    metrics["shape"] = shape

    coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in full_path]

    return {
        "path": full_path,
        "distance_km": round(total_dist, 2),
        "metrics": metrics,
        "coords": coords,
    }


# ── Sequential strategy implementations (also used as fallback) ──────

def _gen_lollipop(
    G: nx.MultiDiGraph,
    start_node: int,
    target_km: float,
    main_comp: set,
    min_dist: float,
    max_dist: float,
    terrain: str = "moderate",
) -> list[dict]:
    """
    Lollipop strategy: stem out → loop at far end → stem back.

    Stem is ~25% of target distance; loop is ~50%; return stem is same
    path as outbound (no zigzag).
    """
    stem_dist = target_km * 0.25  # crow-flies distance for stem
    loop_dist = target_km * 0.25  # crow-flies distance for loop anchor

    stem_anchors = _sample_anchors(G, start_node, stem_dist, main_comp, n_anchors=16)
    routes = []

    for stem_anchor in stem_anchors:
        # Route the stem: start → stem_anchor
        stem_path, stem_cost, stem_km = route_astar(G, start_node, stem_anchor, terrain=terrain)
        if not stem_path or stem_km < 0.3:
            continue

        # Sample loop anchors from the stem anchor
        loop_anchors = _sample_anchors(G, stem_anchor, loop_dist, main_comp, n_anchors=8)

        for loop_anchor in loop_anchors:
            # Loop outbound: stem_anchor → loop_anchor
            loop_out, lout_cost, lout_km = route_astar(G, stem_anchor, loop_anchor, terrain=terrain)
            if not loop_out:
                continue

            # Collect used edges for loop return
            used = set()
            for j in range(len(loop_out) - 1):
                used.add((loop_out[j], loop_out[j + 1]))

            # Loop return: loop_anchor → stem_anchor with reuse penalty
            loop_ret, lret_cost, lret_km = route_astar(
                G, loop_anchor, stem_anchor,
                used_edges=used,
                reuse_penalty=REUSE_PENALTY,
                terrain=terrain,
            )
            if not loop_ret:
                continue

            # Return stem: same path reversed
            ret_stem_path = list(reversed(stem_path))

            # Assemble: stem + loop_out + loop_ret + return_stem
            full_path = (
                stem_path
                + loop_out[1:]     # skip duplicate stem_anchor
                + loop_ret[1:]     # skip duplicate loop_anchor
                + ret_stem_path[1:]  # skip duplicate stem_anchor
            )
            total_dist = stem_km + lout_km + lret_km + stem_km

            # Overlap: stem is reused, loop should be mostly unique
            loop_out_set = set(loop_out)
            loop_ret_set = set(loop_ret)
            loop_overlap = len(loop_out_set & loop_ret_set) / max(1, min(len(loop_out_set), len(loop_ret_set)))

            route = _score_and_filter(
                G, full_path, total_dist, target_km,
                min_dist, max_dist,
                shape="lollipop",
                overlap=loop_overlap,
            )
            if route:
                routes.append(route)

    return routes


def _gen_out_and_back(
    G: nx.MultiDiGraph,
    start_node: int,
    target_km: float,
    main_comp: set,
    min_dist: float,
    max_dist: float,
    terrain: str = "moderate",
) -> list[dict]:
    """
    Out-and-back: route to a scenic point, return the same way.
    Simple and clean — no zigzag, minimal turns.
    """
    half_dist = target_km * 0.5  # crow-flies to far point
    anchors = _sample_anchors(G, start_node, half_dist, main_comp, n_anchors=24)
    routes = []

    for anchor in anchors:
        out_path, out_cost, out_km = route_astar(G, start_node, anchor, terrain=terrain)
        if not out_path:
            continue

        # Return is the same path reversed
        ret_path = list(reversed(out_path))
        full_path = out_path + ret_path[1:]
        total_dist = out_km * 2

        route = _score_and_filter(
            G, full_path, total_dist, target_km,
            min_dist, max_dist,
            shape="out-and-back",
            overlap=1.0,
        )
        if route:
            routes.append(route)

    return routes


def _gen_full_loop(
    G: nx.MultiDiGraph,
    start_node: int,
    target_km: float,
    main_comp: set,
    min_dist: float,
    max_dist: float,
    terrain: str = "moderate",
) -> list[dict]:
    """
    Full loop: outbound + penalized return (classic approach).
    """
    half_dist = target_km / 2.0
    anchors = _sample_anchors(G, start_node, half_dist, main_comp, n_anchors=N_ANCHORS)
    routes = []

    for anchor in anchors:
        out_path, out_cost, out_dist = route_astar(G, start_node, anchor, terrain=terrain)
        if not out_path:
            continue

        used = set()
        for j in range(len(out_path) - 1):
            used.add((out_path[j], out_path[j + 1]))

        ret_path, ret_cost, ret_dist = route_astar(
            G, anchor, start_node,
            used_edges=used,
            reuse_penalty=REUSE_PENALTY,
            terrain=terrain,
        )
        if not ret_path:
            continue

        full_path = out_path + ret_path[1:]
        total_dist = out_dist + ret_dist

        out_set = set(out_path)
        ret_set = set(ret_path)
        overlap = len(out_set & ret_set) / max(1, min(len(out_set), len(ret_set)))

        route = _score_and_filter(
            G, full_path, total_dist, target_km,
            min_dist, max_dist,
            shape="loop",
            overlap=overlap,
        )
        if route:
            routes.append(route)

    return routes


# ── Parallel orchestration ───────────────────────────────────────────

def _run_parallel(G, start_node, target_km, main_comp, min_dist, max_dist, terrain):
    """Run all three strategies in parallel using ProcessPoolExecutor.

    Submits individual anchor tasks (16 lollipop stems, 48 full-loop,
    24 out-and-back) to a worker pool for ~3-4x speedup on multi-core.
    """
    # Sample anchors (fast, no A*)
    stem_dist = target_km * 0.25
    loop_dist = target_km * 0.25
    half_dist = target_km * 0.5

    stem_anchors = _sample_anchors(G, start_node, stem_dist, main_comp, n_anchors=16)
    oab_anchors = _sample_anchors(G, start_node, half_dist, main_comp, n_anchors=24)
    loop_anchors = _sample_anchors(G, start_node, half_dist, main_comp, n_anchors=N_ANCHORS)

    # Serialize graph to temp file for workers
    tmp_fd, graph_tmp = tempfile.mkstemp(suffix=".pkl")
    os.close(tmp_fd)
    with open(graph_tmp, "wb") as f:
        pkl.dump(G, f, protocol=pkl.HIGHEST_PROTOCOL)

    n_workers = min(os.cpu_count() or 4, 6)

    try:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_pool_init,
            initargs=(graph_tmp, _POC8_ROOT),
        ) as pool:
            # Submit all anchor tasks
            lollipop_futures = [
                pool.submit(_pool_lollipop_stem,
                            (sa, start_node, loop_dist, main_comp,
                             target_km, min_dist, max_dist, terrain))
                for sa in stem_anchors
            ]
            loop_futures = [
                pool.submit(_pool_full_loop,
                            (a, start_node, target_km, min_dist, max_dist, terrain))
                for a in loop_anchors
            ]
            oab_futures = [
                pool.submit(_pool_oab,
                            (a, start_node, target_km, min_dist, max_dist, terrain))
                for a in oab_anchors
            ]

            # Collect lollipop results (each future returns a list)
            lollipops = []
            for f in as_completed(lollipop_futures):
                result = f.result()
                if result:
                    lollipops.extend(result)
            print(f"[loops] Lollipop: {len(lollipops)} candidates")

            # Collect full-loop results (each future returns one route or None)
            full_loops = []
            for f in as_completed(loop_futures):
                result = f.result()
                if result:
                    full_loops.append(result)
            print(f"[loops] Full loop: {len(full_loops)} candidates")

            # Collect out-and-back results
            oabs = []
            for f in as_completed(oab_futures):
                result = f.result()
                if result:
                    oabs.append(result)
            print(f"[loops] Out-and-back: {len(oabs)} candidates")

        return lollipops + oabs + full_loops

    finally:
        try:
            os.unlink(graph_tmp)
        except OSError:
            pass


def _run_sequential(G, start_node, target_km, main_comp, min_dist, max_dist, terrain):
    """Run all three strategies sequentially (fallback)."""
    lollipops = _gen_lollipop(G, start_node, target_km, main_comp, min_dist, max_dist, terrain=terrain)
    print(f"[loops] Lollipop: {len(lollipops)} candidates")

    oabs = _gen_out_and_back(G, start_node, target_km, main_comp, min_dist, max_dist, terrain=terrain)
    print(f"[loops] Out-and-back: {len(oabs)} candidates")

    full_loops = _gen_full_loop(G, start_node, target_km, main_comp, min_dist, max_dist, terrain=terrain)
    print(f"[loops] Full loop: {len(full_loops)} candidates")

    return lollipops + oabs + full_loops


# ── Public API ───────────────────────────────────────────────────────

def generate_loops(
    G: nx.MultiDiGraph,
    start_lat: float,
    start_lon: float,
    target_km: float,
    n_results: int = 5,
    terrain: str = "moderate",
    parallel: bool = True,
) -> list[dict]:
    """
    Generate loop routes using three strategies and rank the best.

    Strategies run in parallel by default (ProcessPoolExecutor).
    Falls back to sequential if multiprocessing fails.

    If the first pass finds too few routes, retries with a wider
    distance tolerance (+10 percentage points) to handle graphs
    with very high or low composite scores.

    Args:
        G: scored graph with running_cost
        start_lat, start_lon: starting coordinates
        target_km: target loop distance in km
        n_results: how many routes to return
        terrain: "flat", "moderate", or "hilly"
        parallel: if True, use ProcessPoolExecutor

    Returns:
        List of route dicts sorted by quality.
    """
    t0 = time.perf_counter()

    main_comp = get_main_component(G)
    start_node = nearest_edge_node(G, start_lat, start_lon, node_set=main_comp)

    print(f"[loops] Start node: {start_node}, target: {target_km:.1f}km")

    min_dist = target_km * (1 - DISTANCE_TOLERANCE)
    max_dist = target_km * (1 + DISTANCE_TOLERANCE)

    # Choose execution mode
    runner = _run_parallel if parallel else _run_sequential

    try:
        all_routes = runner(G, start_node, target_km, main_comp, min_dist, max_dist, terrain)
    except Exception as e:
        if parallel:
            print(f"[loops] Parallel failed ({e}), falling back to sequential")
            all_routes = _run_sequential(G, start_node, target_km, main_comp, min_dist, max_dist, terrain)
        else:
            raise

    # Adaptive retry: widen tolerance if too few routes
    if len(all_routes) < n_results:
        wider_tol = DISTANCE_TOLERANCE + 0.10
        wider_min = target_km * (1 - wider_tol)
        wider_max = target_km * (1 + wider_tol)
        print(f"[loops] Only {len(all_routes)} routes, retrying with ±{wider_tol:.0%} tolerance")
        try:
            extra = runner(G, start_node, target_km, main_comp, wider_min, wider_max, terrain)
        except Exception:
            extra = _run_sequential(G, start_node, target_km, main_comp, wider_min, wider_max, terrain)
        # Deduplicate by path hash
        existing = {tuple(r["path"][:5] + r["path"][-5:]) for r in all_routes}
        for r in extra:
            key = tuple(r["path"][:5] + r["path"][-5:])
            if key not in existing:
                all_routes.append(r)
                existing.add(key)

    # Merge and rank
    all_routes.sort(key=lambda r: r["metrics"]["score"], reverse=True)
    routes = all_routes[:n_results]

    elapsed = time.perf_counter() - t0
    print(f"[loops] Generated {len(routes)} routes in {elapsed:.1f}s")
    for i, r in enumerate(routes):
        m = r["metrics"]
        print(f"  #{i+1}: {r['distance_km']:.1f}km [{m['shape']}], score={m['score']:.2f}, "
              f"fp={m['footpath_frac']:.0%}, road={m['road_frac']:.0%}, "
              f"turns/km={m['turns_per_km']:.1f}")

    return routes

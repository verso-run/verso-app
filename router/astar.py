"""
A* point-to-point routing on the scored graph.

Uses running_cost as edge weight with optional turn penalty.
Falls back to length-based cost for edges missing running_cost.

When scipy is available, routes via a direction-expanded graph using
scipy.sparse.csgraph.dijkstra (~30-50ms per query vs 200-500ms pure
Python A*). Falls back to the Python A* implementation otherwise.
"""

import heapq
import math

import networkx as nx

# Expanded graph dispatch: try to import scipy-based accelerator
try:
    from router.astar_rx import ExpandedGraph
    _HAS_EXPANDED = True
except ImportError:
    _HAS_EXPANDED = False

# Expanded graph cache: id(G) -> ExpandedGraph
_expanded_cache: dict[int, "ExpandedGraph"] = {}

# Turn penalty: angles above this threshold incur a cost multiplier.
# 45° = ignore gentle curves, penalize real turns and U-turns.
TURN_THRESHOLD = 45    # degrees
TURN_PENALTY_MAX = 2.5  # multiplier at 180° (U-turn)

# Progressive reuse: short stitches (1-2 edges) nearly free, full penalty at 6+
REUSE_RAMP_EDGES = 6   # edges of consecutive reuse before full penalty

# U-turn surcharge: flat cost added for near-reversals at non-dead-end nodes
U_TURN_ANGLE = 150     # degrees — wider than this is a U-turn
U_TURN_SURCHARGE = 0.15  # ~1.5km of running_cost


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _turn_multiplier(prev_bearing: float | None, edge_bearing: float) -> float:
    """Cost multiplier for a turn from prev_bearing to edge_bearing.

    Returns 1.0 for straight or gentle curves, up to TURN_PENALTY_MAX
    for U-turns.
    """
    if prev_bearing is None:
        return 1.0
    diff = abs(edge_bearing - prev_bearing)
    if diff > 180:
        diff = 360 - diff
    if diff <= TURN_THRESHOLD:
        return 1.0
    # Linear ramp from 1.0 at threshold to TURN_PENALTY_MAX at 180°
    t = (diff - TURN_THRESHOLD) / (180 - TURN_THRESHOLD)
    return 1.0 + t * (TURN_PENALTY_MAX - 1.0)


def route_astar(
    G: nx.MultiDiGraph,
    start_node: int,
    end_node: int,
    weight: str = "running_cost",
    used_edges: set | None = None,
    reuse_penalty: float = 1.0,
    turn_penalty: bool = True,
    terrain: str = "moderate",
) -> tuple[list[int], float, float]:
    """
    A* shortest path on running_cost with haversine heuristic and
    optional turn penalty.

    When scipy is available and turn_penalty is True, dispatches to the
    direction-expanded graph for ~8x speedup. Falls back to pure-Python
    A* otherwise.

    Args:
        G: scored graph with running_cost on edges
        start_node: origin node ID
        end_node: destination node ID
        weight: edge attribute to use as cost
        used_edges: set of (u, v) tuples to penalize (for return leg)
        reuse_penalty: multiplier for used edges (e.g. 2.0)
        turn_penalty: if True, penalize sharp turns
        terrain: "flat" (penalize hills), "moderate" (no change),
                 or "hilly" (reward hills via view_potential)

    Returns:
        (path_nodes, total_cost, distance_km)
    """
    # Expanded graph fast path: use scipy CSR Dijkstra when available
    if _HAS_EXPANDED and turn_penalty and weight == "running_cost":
        g_id = id(G)
        eg = _expanded_cache.get(g_id)
        if eg is None:
            eg = ExpandedGraph(G)
            _expanded_cache[g_id] = eg
        return eg.route(start_node, end_node, used_edges, reuse_penalty, terrain)

    return _route_astar_python(
        G, start_node, end_node, weight, used_edges,
        reuse_penalty, turn_penalty, terrain,
    )


def clear_expanded_cache(G=None):
    """Clear the expanded graph cache.

    Call when edge costs on G have been modified (e.g. deepcopy + re-score).
    If G is given, only clears that graph's entry; otherwise clears all.
    """
    if G is not None:
        _expanded_cache.pop(id(G), None)
    else:
        _expanded_cache.clear()


def _route_astar_python(
    G: nx.MultiDiGraph,
    start_node: int,
    end_node: int,
    weight: str = "running_cost",
    used_edges: set | None = None,
    reuse_penalty: float = 1.0,
    turn_penalty: bool = True,
    terrain: str = "moderate",
) -> tuple[list[int], float, float]:
    """
    Pure-Python A* shortest path (fallback when scipy unavailable).

    Uses running_cost as edge weight with haversine heuristic and
    optional turn penalty.
    """
    if start_node not in G or end_node not in G:
        return [], 0.0, 0.0

    end_lat = G.nodes[end_node]["y"]
    end_lon = G.nodes[end_node]["x"]

    h_scale = 0.0005

    def heuristic(node):
        lat = G.nodes[node]["y"]
        lon = G.nodes[node]["x"]
        return _haversine_km(lat, lon, end_lat, end_lon) * h_scale

    # A* with priority queue
    open_set = [(heuristic(start_node), 0.0, start_node)]
    g_score = {start_node: 0.0}
    came_from = {}
    visited = set()

    # Track incoming bearing at each node (for turn penalty)
    node_bearing = {}  # node -> bearing of edge that reached it
    # Track consecutive reused-edge count at each node (for progressive reuse)
    node_reuse_streak = {}  # node -> int

    while open_set:
        f, g, current = heapq.heappop(open_set)

        if current == end_node:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()

            # Compute actual distance
            dist_km = 0.0
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                edge_data = min(
                    G[u][v].values(),
                    key=lambda d: d.get(weight, d.get("length", 100) / 1000),
                )
                dist_km += edge_data.get("length", 100) / 1000

            return path, g, dist_km

        if current in visited:
            continue
        visited.add(current)

        prev_bearing = node_bearing.get(current)
        current_degree = G.degree(current)  # cache — used in U-turn check
        current_streak = node_reuse_streak.get(current, 0)

        for neighbor in G.successors(current):
            if neighbor in visited:
                continue

            # Pick the best edge (lowest cost including turn penalty)
            best_cost = float("inf")
            best_bearing = 0.0
            best_is_reused = False
            for edge_data in G[current][neighbor].values():
                cost = edge_data.get(weight, edge_data.get("length", 100) / 1000)

                # Apply progressive reuse penalty
                is_reused = False
                if used_edges and reuse_penalty > 1.0:
                    if (current, neighbor) in used_edges or (neighbor, current) in used_edges:
                        is_reused = True
                        penalty = 1.0 + (reuse_penalty - 1.0) * min(1.0, (current_streak + 1) / REUSE_RAMP_EDGES)
                        cost *= penalty

                # Apply turn penalty + U-turn surcharge
                if turn_penalty and prev_bearing is not None:
                    eb = edge_data.get("bearing", 0)
                    tm = _turn_multiplier(prev_bearing, eb)
                    cost *= tm
                    # U-turn surcharge at non-dead-end nodes
                    if current_degree > 2:
                        turn_angle = abs(eb - prev_bearing)
                        if turn_angle > 180:
                            turn_angle = 360 - turn_angle
                        if turn_angle > U_TURN_ANGLE:
                            cost += U_TURN_SURCHARGE

                # Apply terrain preference via view_potential
                if terrain != "moderate":
                    vp = edge_data.get("view_potential", 0.0)
                    if isinstance(vp, (int, float)):
                        if terrain == "flat":
                            cost *= 1.0 + vp * 2.5
                        elif terrain == "hilly":
                            # Penalize flat edges (vp<0.4) AND heavily
                            # discount hilltops (vp=1.0 → 0.25x cost)
                            cost *= max(0.25, 1.5 - vp * 1.25)

                if cost < best_cost:
                    best_cost = cost
                    best_bearing = edge_data.get("bearing", 0)
                    best_is_reused = is_reused

            tentative_g = g + best_cost
            if tentative_g < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                node_bearing[neighbor] = best_bearing
                # Track reuse streak: continues if best edge was reused, resets otherwise
                node_reuse_streak[neighbor] = (current_streak + 1) if best_is_reused else 0
                f_score = tentative_g + heuristic(neighbor)
                heapq.heappush(open_set, (f_score, tentative_g, neighbor))

    # No path found
    return [], 0.0, 0.0


def get_main_component(G: nx.MultiDiGraph) -> set:
    """Return the node set of the largest weakly connected component."""
    components = list(nx.weakly_connected_components(G))
    components.sort(key=len, reverse=True)
    return components[0]


def nearest_node(G: nx.MultiDiGraph, lat: float, lon: float, node_set: set | None = None) -> int:
    """Find the nearest graph node to a lat/lon coordinate.

    If node_set is provided, only considers nodes in that set
    (use get_main_component to avoid disconnected islands).

    Uses cos(lat) correction so longitude degrees are scaled to
    approximate real-world distance at the given latitude.
    """
    cos_lat = math.cos(math.radians(lat))
    best_node = None
    best_dist = float("inf")
    nodes = node_set if node_set is not None else G.nodes()
    for node in nodes:
        data = G.nodes[node]
        dlat = data["y"] - lat
        dlon = (data["x"] - lon) * cos_lat
        d = dlat * dlat + dlon * dlon
        if d < best_dist:
            best_dist = d
            best_node = node
    return best_node


def nearest_edge_node(G: nx.MultiDiGraph, lat: float, lon: float, node_set: set | None = None) -> int:
    """Find the graph node closest to the nearest *edge* to a lat/lon point.

    Unlike nearest_node (which finds the closest intersection), this finds
    the closest edge segment and returns its nearest endpoint. This snaps
    to the correct street even when the point is mid-block.

    Only checks edges within ~500m for performance.
    """
    cos_lat = math.cos(math.radians(lat))
    px, py = lon * cos_lat, lat
    # ~500m bounding box in degrees
    margin = 0.005

    best_dist = float("inf")
    best_node = None

    for u, v, _ in G.edges(keys=True):
        if node_set and (u not in node_set or v not in node_set):
            continue
        u_data, v_data = G.nodes[u], G.nodes[v]

        # Quick bounding-box reject
        if (max(u_data["y"], v_data["y"]) < lat - margin or
            min(u_data["y"], v_data["y"]) > lat + margin or
            max(u_data["x"], v_data["x"]) < lon - margin or
            min(u_data["x"], v_data["x"]) > lon + margin):
            continue

        ax, ay = u_data["x"] * cos_lat, u_data["y"]
        bx, by = v_data["x"] * cos_lat, v_data["y"]

        # Project point onto line segment AB
        dx, dy = bx - ax, by - ay
        len_sq = dx * dx + dy * dy
        if len_sq < 1e-14:
            t = 0.0
        else:
            t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / len_sq))

        # Distance from point to nearest point on segment
        cx, cy = ax + t * dx, ay + t * dy
        d = (px - cx) ** 2 + (py - cy) ** 2

        if d < best_dist:
            best_dist = d
            # Pick the endpoint of this edge closest to the query point
            du = (u_data["y"] - lat) ** 2 + ((u_data["x"] - lon) * cos_lat) ** 2
            dv = (v_data["y"] - lat) ** 2 + ((v_data["x"] - lon) * cos_lat) ** 2
            best_node = u if du <= dv else v

    if best_node is None:
        return nearest_node(G, lat, lon, node_set)
    return best_node

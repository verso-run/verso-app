"""
Graph I/O helpers and edge ID conventions.

Edge IDs are strings "{u}-{v}-{k}" matching the NetworkX MultiDiGraph
edge key tuple. Used as keys in all layer JSON files so layers are
trivially mergeable.
"""

import pickle
from pathlib import Path

import networkx as nx


def edge_id(u: int, v: int, k: int = 0) -> str:
    """Canonical edge ID string."""
    return f"{u}-{v}-{k}"


def parse_edge_id(eid: str) -> tuple[int, int, int]:
    """Reverse of edge_id."""
    parts = eid.split("-")
    return int(parts[0]), int(parts[1]), int(parts[2])


def save_graph(G: nx.MultiDiGraph, path: str) -> None:
    """Save a graph as pickle."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(pickle.dumps(G))
    print(f"[graph] Saved: {p.name} ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")


def load_graph(path: str) -> nx.MultiDiGraph:
    """Load a cached graph from pickle."""
    return pickle.loads(Path(path).read_bytes())


def get_bbox(G: nx.MultiDiGraph) -> tuple[float, float, float, float]:
    """Get (south, west, north, east) bounding box from graph."""
    lats = [d["y"] for _, d in G.nodes(data=True)]
    lons = [d["x"] for _, d in G.nodes(data=True)]
    return (min(lats), min(lons), max(lats), max(lons))


def normalize_tag(data: dict, tag: str) -> str | None:
    """Normalize an OSM tag that may be a list to a single string."""
    val = data.get(tag)
    if isinstance(val, list):
        return val[0]
    return val

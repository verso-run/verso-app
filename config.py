"""
Central configuration for poc8-costmap.

Composite weights, paths, and constants.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
GRAPHS_DIR = DATA_DIR / "graphs"
LAYERS_DIR = DATA_DIR / "layers"
GENERATED_DIR = DATA_DIR / "generated"

# Ensure dirs exist
for d in (GRAPHS_DIR, LAYERS_DIR, GENERATED_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ── Composite Scoring Weights ──────────────────────────────────────
COMPOSITE_WEIGHTS = {
    # From way_tags layer
    "car_free": 1.5,
    "road_penalty": 3.0,
    "busy_adjacent": 3.0,
    "surface": 1.0,
    "lit": 0.5,
    "named": 0.5,
    "legibility": 2.5,
    "signal": 2.5,
    # From continuity layer
    "continuity": 3.0,
    # From landuse layer
    "environment": 2.5,
    "shade": 0.8,
    # From photos layer
    "photo_density": 1.5,
    # From landmarks layer
    "landmark": 1.5,
    # From elevation layer
    "view_potential": 1.0,
    # From claude layer
    "claude_desirability": 2.0,
    # From strava layer
    "strava_popularity": 3.0,
}

# Flat cost added per road crossing — bypasses composite weighted average.
# A traffic signal crossing adds this full amount; crossings/stops scale
# proportionally. Value of 0.05 ≈ 350m extra run on a nice footway.
CROSSING_SURCHARGE = 0.05

# Per-edge fixed cost: penalizes routes that string together many short
# fragments. A route with 350 tiny edges pays 350 * 0.002 = 0.7 extra;
# a route with 170 longer edges pays 0.34. This pushes the router
# toward long continuous paths instead of crosswalk fragment-hopping.
EDGE_FIXED_COST = 0.002


# ── Graph Construction ─────────────────────────────────────────────
def base_graph_path(city: str) -> Path:
    return GRAPHS_DIR / f"{city}_base.pickle"


def scored_graph_path(city: str) -> Path:
    return GRAPHS_DIR / f"{city}_scored.pickle"


def layer_path(city: str, layer_name: str) -> Path:
    return LAYERS_DIR / f"{city}_{layer_name}.json"

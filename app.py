"""
Verso — Curated Running Route Server

FastAPI server with click-to-route (<1s) and layer visualization.

Usage:
    python app.py
    # Open http://localhost:8090
"""

import asyncio
import json
import os
import sys
import time
import traceback
import uuid
from pathlib import Path
from threading import Thread

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from sse_starlette.sse import EventSourceResponse
import uvicorn

# ── Load .env ──────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
_env = _ROOT / ".env"
if _env.exists():
    for line in _env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if v:
            os.environ.setdefault(k, v)

# ── Imports from app modules ─────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config import scored_graph_path, LAYERS_DIR
from graph.helpers import load_graph, get_bbox, edge_id, parse_edge_id
from router.astar import get_main_component, nearest_node
from router.loops import generate_loops
from router.export import to_gpx

# ── App ────────────────────────────────────────────────────────────────
app = FastAPI(title="Verso Costmap Router")

# Pre-loaded state (set at startup)
_G = None
_main_comp = None
_layer_cache = None  # cached layer point data
_bounds = None

# Fast loop generator (set at startup)
_fast_gen = None

# Lock for route generation (batch Dijkstra modifies shared CSR)
_route_lock = asyncio.Lock()

# Job storage for SSE streaming (legacy)
_jobs: dict[str, dict] = {}

# Layer definitions: (layer_file_prefix, field_name, display_label)
LAYER_DEFS = [
    (None,                  "composite_score",   "Composite Score"),
    ("sf_way_tags",         "is_car_free",       "Car-Free Paths"),
    ("sf_landuse",          "environment_score",  "Green Space"),
    ("sf_claude_desirability", "claude_desirability", "Neighborhood Vibe"),
    ("sf_landmarks",        "landmark_proximity", "Landmarks"),
    ("sf_photo_density",    "photo_density",      "Photo Density"),
    ("sf_elevation",        "view_potential",     "Elevation/Views"),
    ("sf_continuity",       "continuity",         "Path Continuity"),
]


def _build_layer_cache():
    """Build layer visualization data as edge segments from scored graph + layer JSONs."""
    global _layer_cache, _bounds

    edges_list = list(_G.edges(keys=True, data=True))
    n_edges = len(edges_list)
    stride = max(1, n_edges // 5000)

    sampled = {}
    for i in range(0, n_edges, stride):
        u, v, k, data = edges_list[i]
        eid = edge_id(u, v, k)
        u_data = _G.nodes[u]
        v_data = _G.nodes[v]
        sampled[eid] = (
            round(u_data["y"], 6), round(u_data["x"], 6),
            round(v_data["y"], 6), round(v_data["x"], 6),
        )

    layers = {}
    for layer_file, field, label in LAYER_DEFS:
        edges_out = []
        if layer_file is None:
            for i in range(0, n_edges, stride):
                u, v, k, data = edges_list[i]
                val = data.get(field)
                if val is None:
                    continue
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    continue
                eid = edge_id(u, v, k)
                ulat, ulon, vlat, vlon = sampled[eid]
                edges_out.append([ulat, ulon, vlat, vlon, round(val, 3)])
        else:
            lp = LAYERS_DIR / f"{layer_file}.json"
            if not lp.exists():
                layers[field] = {"label": label, "edges": []}
                continue
            layer_data = json.loads(lp.read_text(encoding="utf-8"))
            layer_edges = layer_data.get("edges", {})
            for eid, (ulat, ulon, vlat, vlon) in sampled.items():
                edge_scores = layer_edges.get(eid)
                if not edge_scores:
                    continue
                val = edge_scores.get(field)
                if val is None:
                    continue
                if isinstance(val, bool):
                    val = 1.0 if val else 0.0
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    continue
                edges_out.append([ulat, ulon, vlat, vlon, round(val, 3)])
        layers[field] = {"label": label, "edges": edges_out}

    south, west, north, east = get_bbox(_G)
    _bounds = {"south": south, "west": west, "north": north, "east": east}
    _layer_cache = layers


@app.on_event("startup")
async def startup():
    """Pre-load scored graph, build ExpandedGraph + FastLoopGenerator."""
    global _G, _main_comp, _fast_gen

    graph_path = scored_graph_path("sf")
    if not Path(graph_path).exists():
        print(f"[app] WARNING: Scored graph not found at {graph_path}")
        print(f"[app] Copy sf_scored.pickle to data/graphs/")
        return

    t0 = time.perf_counter()
    _G = load_graph(str(graph_path))
    _main_comp = get_main_component(_G)
    print(f"[app] Graph loaded: {_G.number_of_nodes()} nodes, "
          f"{_G.number_of_edges()} edges in {time.perf_counter() - t0:.1f}s")

    # ── Penalize service roads at runtime ──
    svc_count = 0
    for u, v, k, data in _G.edges(keys=True, data=True):
        hw = data.get("highway", "")
        if isinstance(hw, list):
            hw = hw[0]
        if hw == "service":
            rc = data.get("running_cost")
            if rc is not None:
                data["running_cost"] = rc * 5.0
                svc_count += 1
    if svc_count:
        print(f"[app] Penalized {svc_count} service road edges (5x running_cost)")

    # ── Load elevation layer → merge view_potential onto graph edges ──
    elev_path = LAYERS_DIR / "sf_elevation.json"
    if elev_path.exists():
        elev_data = json.loads(elev_path.read_text(encoding="utf-8"))
        elev_edges = elev_data.get("edges", {})
        vp_count = 0
        for u, v, k, data in _G.edges(keys=True, data=True):
            eid = edge_id(u, v, k)
            scores = elev_edges.get(eid)
            if scores:
                vp = scores.get("view_potential")
                if vp is not None:
                    data["view_potential"] = float(vp)
                    vp_count += 1
        print(f"[app] Loaded view_potential onto {vp_count} edges")
    else:
        print(f"[app] No elevation layer found at {elev_path}")

    # ── Build layer cache ──
    t1 = time.perf_counter()
    _build_layer_cache()
    print(f"[app] Layer cache built in {time.perf_counter() - t1:.1f}s")

    # ── Build ExpandedGraph + FastLoopGenerator ──
    t2 = time.perf_counter()
    try:
        from router.astar_rx import ExpandedGraph
        from router.loops_fast import FastLoopGenerator
        eg = ExpandedGraph(_G)
        _fast_gen = FastLoopGenerator(_G, eg)
        print(f"[app] FastLoopGenerator ready in {time.perf_counter() - t2:.1f}s")
    except Exception as e:
        print(f"[app] WARNING: FastLoopGenerator failed ({e}), falling back to slow mode")
        traceback.print_exc()


def _dedup_and_rank(routes: list[dict]) -> list[dict]:
    """Deduplicate routes by coordinate overlap, rank by score."""
    if not routes:
        return []
    ranked = sorted(routes, key=lambda r: r["score"], reverse=True)
    kept = []
    for r in ranked:
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


def _format_route(r: dict) -> dict:
    """Format a raw route dict for the JSON API response."""
    m = r["metrics"]
    return {
        "distance_km": r["distance_km"],
        "shape": m["shape"],
        "footpath_pct": round(m["footpath_frac"] * 100),
        "road_pct": round(m["road_frac"] * 100),
        "turns_per_km": m["turns_per_km"],
        "score": m["score"],
        "avg_composite": m["avg_composite"],
        "edge_count": m["edge_count"],
        "overlap": m.get("overlap", 0),
        "coords": [[c[0], c[1]] for c in r["coords"]],
    }


# ── Endpoints ──────────────────────────────────────────────────────────

@app.get("/")
async def index():
    html_path = Path(__file__).parent / "templates" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.post("/api/routes")
async def fast_routes(request: Request):
    """Synchronous fast route generation (<1s). Used by click-to-route UI."""
    body = await request.json()
    lat = float(body.get("lat", 0))
    lon = float(body.get("lon", 0))
    min_km = float(body.get("min_km", 5))
    max_km = float(body.get("max_km", 10))
    terrain = body.get("terrain", "moderate")
    if terrain not in ("flat", "moderate", "hilly"):
        terrain = "moderate"

    if not lat or not lon:
        return JSONResponse({"error": "lat and lon are required"}, 400)
    if _G is None:
        return JSONResponse({"error": "Graph not loaded"}, 503)
    if _fast_gen is None:
        return JSONResponse({"error": "FastLoopGenerator not available"}, 503)

    # Generate at 2-3 target distances within range
    if max_km <= min_km:
        targets = [min_km]
    else:
        n_targets = min(3, max(2, int((max_km - min_km) / 2) + 1))
        step = (max_km - min_km) / (n_targets - 1) if n_targets > 1 else 0
        targets = [round(min_km + i * step, 1) for i in range(n_targets)]

    t0 = time.perf_counter()
    all_routes = []

    async with _route_lock:
        for target_km in targets:
            try:
                routes = _fast_gen.generate(lat, lon, target_km, n_results=10, terrain=terrain)
                all_routes.extend([_format_route(r) for r in routes])
            except Exception as e:
                print(f"[routes] Error at {target_km}km: {e}")
                traceback.print_exc()

    # Dedup and rank
    ranked = _dedup_and_rank(all_routes)
    for i, r in enumerate(ranked):
        r["id"] = i + 1

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "routes": ranked,
        "lat": lat,
        "lon": lon,
        "elapsed_ms": round(elapsed_ms),
        "count": len(ranked),
    }


@app.post("/api/gpx")
async def gpx_export(request: Request):
    """Export route coordinates as GPX file."""
    body = await request.json()
    coords = body.get("coords", [])
    distance_km = body.get("distance_km", 0)
    name = body.get("name", "Verso Route")

    if not coords or len(coords) < 2:
        return JSONResponse({"error": "coords required"}, 400)

    route = {
        "coords": [(c[0], c[1]) for c in coords],
        "distance_km": distance_km,
    }
    gpx_xml = to_gpx(route, name=name)

    return Response(
        content=gpx_xml,
        media_type="application/gpx+xml",
        headers={"Content-Disposition": f'attachment; filename="{name}.gpx"'},
    )


# ── Legacy SSE endpoints (backward compatibility) ─────────────────────

def _geocode(address: str) -> tuple[float, float, str] | None:
    """Geocode an address via Google Geocoding API."""
    import re
    m = re.match(r"^\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*$", address)
    if m:
        lat, lon = float(m.group(1)), float(m.group(2))
        return lat, lon, f"{lat:.5f}, {lon:.5f}"

    api_key = os.environ.get("GOOGLE_PLACES_API_KEY")
    if not api_key:
        return None

    import httpx
    resp = httpx.get(
        "https://maps.googleapis.com/maps/api/geocode/json",
        params={"address": address, "key": api_key},
        timeout=10,
    )
    data = resp.json()
    if data.get("status") != "OK" or not data.get("results"):
        return None

    loc = data["results"][0]["geometry"]["location"]
    fmt = data["results"][0].get("formatted_address", address)
    return loc["lat"], loc["lng"], fmt


def _run_route_job(job_id: str, address: str, min_km: float, max_km: float, terrain: str = "moderate"):
    """Background thread for legacy SSE endpoint."""
    job = _jobs[job_id]

    def emit(event: str, data: dict):
        job["events"].append({"event": event, "data": data})

    if _G is None:
        emit("error", {"message": "Graph not loaded."})
        job["status"] = "error"
        return

    emit("progress", {"message": f"Geocoding: {address}...", "pct": 5})
    result = _geocode(address)
    if not result:
        emit("error", {"message": f"Could not geocode '{address}'."})
        job["status"] = "error"
        return

    lat, lon, formatted = result
    emit("progress", {"message": f"Geocoded: {formatted}", "pct": 10})

    if max_km <= min_km:
        targets = [min_km]
    else:
        n_targets = min(5, max(2, int((max_km - min_km) / 1.5) + 1))
        step = (max_km - min_km) / (n_targets - 1) if n_targets > 1 else 0
        targets = [round(min_km + i * step, 1) for i in range(n_targets)]

    all_routes = []
    for ti, target_km in enumerate(targets):
        pct_base = 15 + int(75 * ti / len(targets))
        emit("progress", {"message": f"Generating {target_km} km routes...", "pct": pct_base})
        try:
            routes = generate_loops(_G, lat, lon, target_km, n_results=15, terrain=terrain)
        except Exception as e:
            print(f"[route] Error at {target_km}km: {e}")
            routes = []
        for r in routes:
            all_routes.append(_format_route(r))
        pct_after = 15 + int(75 * (ti + 1) / len(targets))
        emit("routes", {"routes": _dedup_and_rank(all_routes), "pct": pct_after,
                         "message": f"Completed {target_km} km"})

    ranked = _dedup_and_rank(all_routes)
    for i, r in enumerate(ranked):
        r["id"] = i + 1

    emit("done", {"routes": ranked, "address": formatted,
                   "lat": lat, "lon": lon, "count": len(ranked)})
    job["status"] = "done"


@app.post("/api/route")
async def route_legacy(request: Request):
    body = await request.json()
    address = body.get("address", "").strip()
    min_km = float(body.get("min_km", 5))
    max_km = float(body.get("max_km", 10))
    terrain = body.get("terrain", "moderate")
    if terrain not in ("flat", "moderate", "hilly"):
        terrain = "moderate"
    if not address:
        return JSONResponse({"error": "address is required"}, 400)

    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {"status": "running", "events": []}
    thread = Thread(target=_run_route_job, args=(job_id, address, min_km, max_km, terrain), daemon=True)
    thread.start()
    return {"job_id": job_id}


@app.get("/api/route/{job_id}/stream")
async def stream(job_id: str):
    if job_id not in _jobs:
        return JSONResponse({"error": "job not found"}, 404)

    async def event_generator():
        last_idx = 0
        while True:
            events = _jobs[job_id]["events"]
            while last_idx < len(events):
                evt = events[last_idx]
                last_idx += 1
                yield {"event": evt["event"], "data": json.dumps(evt["data"])}
                if evt["event"] in ("done", "error"):
                    return
            if _jobs[job_id]["status"] in ("done", "error"):
                return
            await asyncio.sleep(0.3)

    return EventSourceResponse(event_generator())


@app.get("/api/layers")
async def layers():
    if _layer_cache is None:
        return JSONResponse({"error": "Graph not loaded"}, 503)
    return {"layers": _layer_cache, "bounds": _bounds}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8090))
    print(f"Starting Verso on http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)

"""
Export routes as GeoJSON and GPX.
"""

import json
import time
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring


def to_geojson(route: dict) -> dict:
    """Convert a route dict to a GeoJSON Feature."""
    coords = route["coords"]
    # GeoJSON uses [lon, lat]
    geojson_coords = [[c[1], c[0]] for c in coords]

    return {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": geojson_coords,
        },
        "properties": {
            "distance_km": route["distance_km"],
            **route.get("metrics", {}),
        },
    }


def save_geojson(route: dict, output_path: str) -> str:
    """Save a route as a GeoJSON file."""
    feature = to_geojson(route)
    collection = {
        "type": "FeatureCollection",
        "features": [feature],
    }
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(collection, indent=2), encoding="utf-8")
    print(f"[export] GeoJSON saved: {p.name}")
    return output_path


def to_gpx(route: dict, name: str = "Verso Route") -> str:
    """Convert a route dict to a GPX XML string."""
    gpx = Element("gpx", {
        "version": "1.1",
        "creator": "Verso",
        "xmlns": "http://www.topografix.com/GPX/1/1",
    })

    metadata = SubElement(gpx, "metadata")
    name_el = SubElement(metadata, "name")
    name_el.text = name

    trk = SubElement(gpx, "trk")
    trk_name = SubElement(trk, "name")
    trk_name.text = name

    trkseg = SubElement(trk, "trkseg")
    for lat, lon in route["coords"]:
        SubElement(trkseg, "trkpt", {
            "lat": f"{lat:.7f}",
            "lon": f"{lon:.7f}",
        })

    return tostring(gpx, encoding="unicode", xml_declaration=True)


def save_gpx(route: dict, output_path: str, name: str = "Verso Route") -> str:
    """Save a route as a GPX file."""
    gpx_str = to_gpx(route, name)
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(gpx_str, encoding="utf-8")
    print(f"[export] GPX saved: {p.name}")
    return output_path

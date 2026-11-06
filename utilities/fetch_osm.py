"""Download the OpenStreetMap extracts for the project's campuses once, with provenance.

Each region is fetched from the Overpass API as OSM XML with ``out meta``, so every way and node
carries its version and last-edit timestamp (needed for the map view's label-age discount). A region
with ``as_of`` set is fetched as the map stood at that moment (Overpass attic data). Output:

    data/osm/<region>.osm        OSM XML (Map data (c) OpenStreetMap contributors, ODbL); user, uid,
                                 and changeset attributes are removed before saving
    data/osm/<region>.osm.json   sidecar: bbox, as_of, query, endpoint, fetch time, Overpass data
                                 timestamp, SHA-256 of the .osm file, element counts

After this runs once, nothing else needs the network. Existing files are kept unless
``--overwrite`` is given, so a re-run never silently changes the map the experiments used.

Usage (stdlib only; any Python 3.9+)::

    python -m utilities.fetch_osm                 # all regions
    python -m utilities.fetch_osm umd_mckeldin --overwrite
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "osm"
ENDPOINT = "https://overpass-api.de/api/interpreter"
STRIP_CONTRIBUTOR = re.compile(rb' (?:user|uid|changeset)="[^"]*"')
USER_AGENT = "ml4ss-terrain-gate/0.1 (course research project; one-time extract)"

# bbox = (south, west, north, east) in WGS84 degrees: generous boxes around the GND drives; tighten
# them from the bags' GPS once the cache adapter exists. as_of = the map as it stood when GND was
# recorded (Overpass attic query), so label ages are measured against what the robot would have had;
# None means the current map.
REGIONS = {
    # UMD map 3 north session: SLAM trajectory starts 2024-08-22 16:26:39 UTC.
    "umd_mckeldin": {"bbox": (38.9815, -76.9500, 38.9905, -76.9360), "as_of": "2024-08-22T16:26:39Z"},
    # Fallback campus; drive date not yet determined, so this is the current map.
    "gtown": {"bbox": (38.9030, -77.0800, 38.9120, -77.0670), "as_of": None},
}

# Every highway way (footways, steps, crossings, roads) with its nodes, plus point features that
# mark curbs and crossings. Tier T1/T2 rules and the map view read only these.
QUERY = """[out:xml][timeout:120]{date};
(
  way["highway"]({bbox});
  node["kerb"]({bbox});
  node["barrier"="kerb"]({bbox});
  node["highway"~"^(crossing|steps)$"]({bbox});
);
(._; >;);
out meta;
"""


def fetch(region: str, overwrite: bool) -> str:
    bbox, as_of = REGIONS[region]["bbox"], REGIONS[region]["as_of"]
    osm_path = OUT_DIR / f"{region}.osm"
    meta_path = OUT_DIR / f"{region}.osm.json"
    if osm_path.exists() and not overwrite:
        return f"{region}: {osm_path} exists (use --overwrite to refetch)"

    query = QUERY.format(bbox=",".join(f"{v:.6f}" for v in bbox),
                         date=f'[date:"{as_of}"]' if as_of else "")
    request = urllib.request.Request(
        ENDPOINT,
        data=urllib.parse.urlencode({"data": query}).encode(),
        headers={"User-Agent": USER_AGENT},
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                body = response.read()
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(10 * (attempt + 1))  # Overpass rate-limits bursts

    # Drop contributor identity (user name, user id, changeset id). The project needs only each
    # element's version and timestamp, and OSMF treats contributor identities as personal data.
    body = STRIP_CONTRIBUTOR.sub(b"", body)
    root = ET.fromstring(body)  # fail before writing anything if the response is not OSM XML
    if root.tag != "osm":
        raise RuntimeError(f"{region}: unexpected response root <{root.tag}>")
    remark = root.find("remark")
    if remark is not None:  # Overpass reports timeouts/errors here with a 200 status
        raise RuntimeError(f"{region}: Overpass remark: {remark.text}")
    counts = {tag: len(root.findall(tag)) for tag in ("node", "way", "relation")}
    meta_el = root.find("meta")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = osm_path.with_name(osm_path.name + ".part")
    tmp.write_bytes(body)
    tmp.replace(osm_path)
    meta_path.write_text(json.dumps({
        "region": region,
        "bbox_swne": list(bbox),
        "as_of": as_of,
        "endpoint": ENDPOINT,
        "query": query,
        "fetched_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "overpass_data_timestamp": meta_el.get("osm_base") if meta_el is not None else None,
        "sha256": hashlib.sha256(body).hexdigest(),
        "bytes": len(body),
        "counts": counts,
        "contributor_fields_removed": ["user", "uid", "changeset"],
        "license": "Map data (c) OpenStreetMap contributors, ODbL 1.0; https://www.openstreetmap.org/copyright",
    }, indent=2) + "\n")
    return f"{region}: {counts['way']} ways, {counts['node']} nodes, {len(body) / 1e3:.0f} kB -> {osm_path}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("regions", nargs="*", help=f"subset of {', '.join(REGIONS)} (default: all)")
    parser.add_argument("--overwrite", action="store_true", help="refetch and replace existing extracts")
    args = parser.parse_args(argv)
    unknown = set(args.regions) - set(REGIONS)
    if unknown:
        parser.error(f"unknown region(s): {', '.join(sorted(unknown))}")
    failures = 0
    for i, region in enumerate(args.regions or REGIONS):
        if i:
            time.sleep(2)  # be polite to the shared Overpass server
        try:
            print(fetch(region, args.overwrite), flush=True)
        except Exception as exc:
            failures += 1
            print(f"FAILED {region}: {exc}", file=sys.stderr, flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

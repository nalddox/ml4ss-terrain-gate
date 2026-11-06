# OpenStreetMap extracts

Map data © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright).

The `.osm` files in this directory contain data from OpenStreetMap, which is made available under the
[Open Database License (ODbL) 1.0](https://opendatacommons.org/licenses/odbl/1-0/). These extracts are
derived databases of OpenStreetMap and are likewise made available under the ODbL 1.0. The license of the
rest of this repository does not apply to them. Any further database derived from them, such as feature
caches built from OSM tags, is also subject to the ODbL.

| File | Area | Map as of |
|---|---|---|
| `umd_mckeldin.osm` | University of Maryland, McKeldin Mall (GND UMD map 3) | 2024-08-22 16:26:39 UTC (start of the GND north-session drive) |
| `gtown.osm` | Georgetown University main campus (GND fallback campus) | date of download (see sidecar) |

## How these were made

The extracts were downloaded once with `python -m utilities.fetch_osm` from the
[Overpass API](https://overpass-api.de/). The script fetches every `highway` way with its nodes, plus nodes
tagged `kerb`, `barrier=kerb`, `highway=crossing` or `highway=steps`. Each element keeps its version and
last-edit timestamp, which the project uses to estimate label age.

Changes from the source data: the `user`, `uid` and `changeset` attributes were removed, so contributor
identities are not republished here. No geometry or tags were changed.

Each `<file>.osm.json` sidecar records the bounding box, the as-of date, the exact query, the download
time, the Overpass data timestamp, the SHA-256 of the `.osm` file and element counts.

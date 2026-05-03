Place the official structural catalog bundle here as:

`official_structural_bundle.json`

Recommended command:

`uv run python scripts/rebuild_official_runtime.py`

Expected bundle shape:

- `sources[]`
- `municipalities[]`
- `roadSegments[]`
- `zones[]`

Every structural feature must include official provenance in `sourceId` and
`sourceRef`. Layer metadata files alone are not enough; the bundle must contain
actual municipality, zone, and road geometries.

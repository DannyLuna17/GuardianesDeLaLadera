"""Download the SGC SIMMA landslide inventory via its public ArcGIS REST service.

This mirrors the pipeline step "Paso 1" from the research brief:

    base: https://srvags.sgc.gov.co/arcgis/rest/services/SIMMA/Capas_Principales/MapServer
    layer 1  -> landslide points (records)
    layer 22 -> movement-type classification table

The script is resumable: it lists every OBJECTID once, splits them into fixed-size
chunks, and writes each chunk as its own GeoJSON file under ``lotes/``. Re-running
it skips chunks whose file already exists, so a partial download can finish after
a network blip without restarting from zero.

Usage:

    uv run python scripts/landslide_inventory/fetch_simma.py \
        --output-dir data/inventory/00_raw/simma

    # testing / smoke-check:
    uv run python scripts/landslide_inventory/fetch_simma.py --max-chunks 1

"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx


SIMMA_BASE_URL = (
    "https://srvags.sgc.gov.co/arcgis/rest/services/SIMMA/"
    "Capas_Principales/MapServer"
)
SIMMA_GENERALES_BASE_URL = (
    "https://srvags.sgc.gov.co/arcgis/rest/services/SIMMA/"
    "Capas_Generales/MapServer"
)
DEFAULT_POINTS_LAYER = 1
DEFAULT_CLASSIFICATION_LAYER = 22
DEFAULT_MUNICIPIOS_LAYER = 5
DEFAULT_DEPARTAMENTOS_LAYER = 6
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_ADMIN_CHUNK_SIZE = 200
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_RETRIES = 4
DEFAULT_RETRY_BACKOFF = 2.5


def _layer_url(layer_id: int, base_url: str = SIMMA_BASE_URL) -> str:
    return f"{base_url}/{layer_id}/query"


def _request_with_retry(
    client: httpx.Client,
    url: str,
    *,
    params: dict,
    method: str = "GET",
    max_retries: int,
    backoff_base: float,
) -> dict:
    """Call ArcGIS REST with retry. Use POST for large ID lists to avoid URL-too-long 404s."""
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            if method == "POST":
                response = client.post(url, data=params)
            else:
                response = client.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            last_exc = exc
            if attempt == max_retries:
                break
            sleep_for = backoff_base ** attempt
            print(
                f"  [retry {attempt}/{max_retries}] {exc!r} — sleeping {sleep_for:.1f}s",
                file=sys.stderr,
            )
            time.sleep(sleep_for)
    assert last_exc is not None
    raise last_exc


def list_object_ids(
    client: httpx.Client,
    *,
    layer_id: int,
    base_url: str,
    max_retries: int,
    backoff_base: float,
) -> list[int]:
    print(f"Listing all OBJECTIDs from layer {layer_id} ...")
    payload = _request_with_retry(
        client,
        _layer_url(layer_id, base_url),
        params={"where": "1=1", "returnIdsOnly": "true", "f": "json"},
        max_retries=max_retries,
        backoff_base=backoff_base,
    )
    object_ids = payload.get("objectIds") or []
    id_field = payload.get("objectIdFieldName")
    print(f"  found {len(object_ids)} ids (field={id_field})")
    return sorted(int(value) for value in object_ids)


def fetch_chunk(
    client: httpx.Client,
    *,
    layer_id: int,
    base_url: str,
    chunk: list[int],
    max_retries: int,
    backoff_base: float,
) -> dict:
    object_ids = ",".join(str(value) for value in chunk)
    # ArcGIS rejects GET URLs past ~4KB, and a 1000-id list easily blows that
    # budget. POST form body is the canonical way to pass long objectIds.
    return _request_with_retry(
        client,
        _layer_url(layer_id, base_url),
        params={
            "objectIds": object_ids,
            "outFields": "*",
            "returnGeometry": "true",
            "outSR": "4326",
            "f": "geojson",
        },
        method="POST",
        max_retries=max_retries,
        backoff_base=backoff_base,
    )


def fetch_classification(
    client: httpx.Client,
    *,
    layer_id: int,
    base_url: str,
    max_retries: int,
    backoff_base: float,
) -> dict:
    return _request_with_retry(
        client,
        _layer_url(layer_id, base_url),
        params={"where": "1=1", "outFields": "*", "f": "json"},
        max_retries=max_retries,
        backoff_base=backoff_base,
    )


def fetch_polygon_layer_all(
    client: httpx.Client,
    *,
    base_url: str,
    layer_id: int,
    label: str,
    chunk_size: int,
    max_retries: int,
    backoff_base: float,
) -> dict:
    """Fetch every feature from a polygon layer, merging chunks into one GeoJSON.

    The SGC ArcGIS service for Capas_Generales caps per-response records, so we
    still use the enumerate-ids + batch-objectIds pattern we use for points.
    """
    print(f"Listing OBJECTIDs for {label} (layer {layer_id}) ...")
    ids = _request_with_retry(
        client,
        _layer_url(layer_id, base_url),
        params={"where": "1=1", "returnIdsOnly": "true", "f": "json"},
        max_retries=max_retries,
        backoff_base=backoff_base,
    ).get("objectIds") or []
    ids = sorted(int(value) for value in ids)
    print(f"  {label}: {len(ids)} features to fetch")

    features: list[dict] = []
    for index, chunk in enumerate(_chunks(ids, chunk_size), start=1):
        print(
            f"  [{label} {index}/{(len(ids) + chunk_size - 1) // chunk_size}] "
            f"fetching OBJECTIDs {chunk[0]}..{chunk[-1]}"
        )
        payload = _request_with_retry(
            client,
            _layer_url(layer_id, base_url),
            params={
                "objectIds": ",".join(str(value) for value in chunk),
                "outFields": "*",
                "returnGeometry": "true",
                "outSR": "4326",
                "f": "geojson",
            },
            method="POST",
            max_retries=max_retries,
            backoff_base=backoff_base,
        )
        features.extend(payload.get("features") or [])

    return {"type": "FeatureCollection", "features": features}


def _chunks(values: list[int], size: int) -> list[list[int]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _chunk_filename(chunk_index: int, chunk: list[int]) -> str:
    return (
        f"lote_{chunk_index:05d}_ids_{chunk[0]:08d}_to_{chunk[-1]:08d}.geojson"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=SIMMA_BASE_URL,
        help="SIMMA ArcGIS REST base URL (override if SGC changes it).",
    )
    parser.add_argument(
        "--points-layer",
        type=int,
        default=DEFAULT_POINTS_LAYER,
        help="Layer ID for landslide points (default: %(default)s).",
    )
    parser.add_argument(
        "--classification-layer",
        type=int,
        default=DEFAULT_CLASSIFICATION_LAYER,
        help="Layer ID for the movement-type classification table (default: %(default)s).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="OBJECTIDs per REST query (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/inventory/00_raw/simma"),
        help="Where to write the raw lotes and classification table.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Optional cap on chunks to fetch (useful for smoke tests).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="HTTP timeout per request in seconds (default: %(default)s).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Retries per request (default: %(default)s).",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=DEFAULT_RETRY_BACKOFF,
        help="Exponential backoff base for retries (default: %(default)s).",
    )
    parser.add_argument(
        "--skip-classification",
        action="store_true",
        help="Do not re-download the classification table if it already exists.",
    )
    parser.add_argument(
        "--skip-admin-boundaries",
        action="store_true",
        help=(
            "Do not download the Capas_Generales municipios/departamentos "
            "polygons. These are used by normalize_simma.py to spatial-join "
            "each SIMMA point to a municipality, so you normally want them."
        ),
    )
    parser.add_argument(
        "--generales-base-url",
        default=SIMMA_GENERALES_BASE_URL,
        help="SIMMA Capas_Generales ArcGIS REST base URL.",
    )
    parser.add_argument(
        "--municipios-layer",
        type=int,
        default=DEFAULT_MUNICIPIOS_LAYER,
        help="Layer ID for Municipios in Capas_Generales (default: %(default)s).",
    )
    parser.add_argument(
        "--departamentos-layer",
        type=int,
        default=DEFAULT_DEPARTAMENTOS_LAYER,
        help="Layer ID for Departamentos in Capas_Generales (default: %(default)s).",
    )
    parser.add_argument(
        "--admin-chunk-size",
        type=int,
        default=DEFAULT_ADMIN_CHUNK_SIZE,
        help="Features per request for admin-boundary layers (default: %(default)s).",
    )
    args = parser.parse_args(argv)

    lotes_dir = args.output_dir / "lotes"
    lotes_dir.mkdir(parents=True, exist_ok=True)
    boundaries_dir = args.output_dir / "boundaries"
    boundaries_dir.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=args.timeout) as client:
        if not args.skip_admin_boundaries:
            municipios_path = boundaries_dir / "municipios.geojson"
            if municipios_path.exists():
                print(
                    f"Municipios polygons already on disk -> {municipios_path}"
                )
            else:
                municipios = fetch_polygon_layer_all(
                    client,
                    base_url=args.generales_base_url,
                    layer_id=args.municipios_layer,
                    label="Municipios",
                    chunk_size=args.admin_chunk_size,
                    max_retries=args.max_retries,
                    backoff_base=args.retry_backoff,
                )
                municipios_path.write_text(
                    json.dumps(municipios, ensure_ascii=False),
                    encoding="utf-8",
                )
                print(f"  wrote {municipios_path}")

            departamentos_path = boundaries_dir / "departamentos.geojson"
            if departamentos_path.exists():
                print(
                    f"Departamentos polygons already on disk -> {departamentos_path}"
                )
            else:
                departamentos = fetch_polygon_layer_all(
                    client,
                    base_url=args.generales_base_url,
                    layer_id=args.departamentos_layer,
                    label="Departamentos",
                    chunk_size=args.admin_chunk_size,
                    max_retries=args.max_retries,
                    backoff_base=args.retry_backoff,
                )
                departamentos_path.write_text(
                    json.dumps(departamentos, ensure_ascii=False),
                    encoding="utf-8",
                )
                print(f"  wrote {departamentos_path}")

        classification_path = args.output_dir / "simma_clasificacion.json"
        if classification_path.exists() and args.skip_classification:
            print(f"Classification already downloaded -> {classification_path}")
        else:
            print(
                f"Downloading classification table (layer {args.classification_layer}) ..."
            )
            classification = fetch_classification(
                client,
                layer_id=args.classification_layer,
                base_url=args.base_url,
                max_retries=args.max_retries,
                backoff_base=args.retry_backoff,
            )
            classification_path.write_text(
                json.dumps(classification, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  wrote {classification_path}")

        object_ids = list_object_ids(
            client,
            layer_id=args.points_layer,
            base_url=args.base_url,
            max_retries=args.max_retries,
            backoff_base=args.retry_backoff,
        )
        if not object_ids:
            print("No OBJECTIDs returned — nothing to download.", file=sys.stderr)
            return 1

        chunks = _chunks(object_ids, args.chunk_size)
        if args.max_chunks is not None:
            chunks = chunks[: args.max_chunks]
        print(
            f"Planning {len(chunks)} chunks of up to {args.chunk_size} OBJECTIDs each."
        )

        downloaded = 0
        skipped = 0
        for index, chunk in enumerate(chunks, start=1):
            file_path = lotes_dir / _chunk_filename(index, chunk)
            if file_path.exists():
                skipped += 1
                continue
            print(
                f"[{index}/{len(chunks)}] fetching OBJECTIDs "
                f"{chunk[0]}..{chunk[-1]} ({len(chunk)} records)"
            )
            payload = fetch_chunk(
                client,
                layer_id=args.points_layer,
                base_url=args.base_url,
                chunk=chunk,
                max_retries=args.max_retries,
                backoff_base=args.retry_backoff,
            )
            file_path.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
            downloaded += 1

    print(
        f"Done. Chunks downloaded now: {downloaded}. "
        f"Already on disk and skipped: {skipped}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

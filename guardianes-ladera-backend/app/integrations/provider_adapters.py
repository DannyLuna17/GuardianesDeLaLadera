from __future__ import annotations

import csv
import inspect
import io
import json
import logging
import time as time_module
import zipfile
from datetime import date, datetime, time as time_of_day, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.text import normalize_lookup_key
from app.integrations.base import SyncResult
from app.integrations.normalizers import sync_historical_events, sync_rain_series, sync_ungrd_records
from app.integrations.provider_parsers import (
    parse_ideam_payload,
    parse_sgc_payload,
    parse_ungrd_payload,
)
from app.models import Municipality


IDEAM_PRECIPITATION_CATEGORIES = {"AM", "CO", "CP", "ME", "PG", "PM", "SP", "SS"}
logger = logging.getLogger(__name__)


class BaseHttpAdapter:
    source_id: str
    endpoint_path: str = ""
    adapter_key: str
    transport = "http"

    def __init__(self) -> None:
        self.settings = get_settings()

    @property
    def base_url(self) -> str:
        base_url = self.settings.source_base_url(self.source_id)
        if not base_url:
            raise RuntimeError(f"No base URL configured for source '{self.source_id}'.")
        return base_url.rstrip("/") + "/"

    @property
    def endpoint_url(self) -> str:
        return urljoin(self.base_url, self.endpoint_path.lstrip("/"))

    def _request_headers(self, *, apply_auth: bool = True) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "User-Agent": "guardianes-ladera-backend/0.1",
        }
        header_name = self.settings.source_auth_header_name(self.source_id)
        token = self.settings.source_auth_token(self.source_id)
        if apply_auth and header_name and token:
            headers[header_name] = token
        return headers

    def _with_source_auth_query(self, url: str, *, apply_auth: bool = True) -> str:
        query_param = self.settings.source_auth_query_param(self.source_id)
        token = self.settings.source_auth_token(self.source_id)
        if not apply_auth or not query_param or not token:
            return url
        split = urlsplit(url)
        query_items = parse_qsl(split.query, keep_blank_values=True)
        if query_param not in {key for key, _ in query_items}:
            query_items.append((query_param, token))
        return urlunsplit(
            (
                split.scheme,
                split.netloc,
                split.path,
                urlencode(query_items),
                split.fragment,
            )
        )

    def _request(self, url: str, *, apply_auth: bool = True) -> bytes:
        request_url = self._with_source_auth_query(url, apply_auth=apply_auth)
        request = Request(
            request_url,
            headers=self._request_headers(apply_auth=apply_auth),
        )
        attempts = max(int(self.settings.provider_request_retry_attempts), 1)
        timeout_seconds = self.settings.provider_request_timeout_seconds
        backoff_seconds = max(float(self.settings.provider_request_retry_backoff_seconds), 0.0)

        for attempt in range(1, attempts + 1):
            try:
                with urlopen(request, timeout=timeout_seconds) as response:
                    return response.read()
            except HTTPError:
                raise
            except (TimeoutError, URLError) as exc:
                if attempt >= attempts:
                    raise
                logger.warning(
                    "Provider request failed for %s; retrying attempt %s/%s (%s) url=%s",
                    self.source_id,
                    attempt + 1,
                    attempts,
                    type(exc).__name__,
                    request_url,
                )
                if backoff_seconds > 0:
                    time_module.sleep(backoff_seconds)

        raise RuntimeError(
            f"Provider request exhausted all retry attempts for '{request_url}'."
        )

    def _cache_path(self, name: str) -> Path:
        cache_dir = self.settings.resolved_provider_cache_path
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / name

    def _cache_is_fresh(self, cache_path: Path, max_age_minutes: int) -> bool:
        if not cache_path.exists():
            return False
        age_seconds = datetime.now(timezone.utc).timestamp() - cache_path.stat().st_mtime
        return age_seconds <= (max_age_minutes * 60)

    def _read_or_download_bytes(
        self,
        *,
        url: str,
        cache_name: str,
        max_age_minutes: int,
        apply_auth: bool = True,
    ) -> bytes:
        cache_path = self._cache_path(cache_name)
        if self._cache_is_fresh(cache_path, max_age_minutes):
            return cache_path.read_bytes()
        payload = self._request(url, apply_auth=apply_auth)
        cache_path.write_bytes(payload)
        return payload

    def _read_or_download_json(
        self,
        *,
        url: str,
        cache_name: str,
        max_age_minutes: int,
        apply_auth: bool = True,
    ) -> object:
        raw = self._read_or_download_bytes(
            url=url,
            cache_name=cache_name,
            max_age_minutes=max_age_minutes,
            apply_auth=apply_auth,
        )
        return json.loads(raw.decode("utf-8"))

    def fetch_payload(self, session: Session | None = None) -> object:
        return self._read_or_download_json(
            url=self.endpoint_url,
            cache_name=f"{self.source_id.lower()}-payload.json",
            max_age_minutes=15,
        )

    def _fetch_payload_for_sync(self, session: Session) -> object:
        fetcher = self.fetch_payload
        if len(inspect.signature(fetcher).parameters) == 0:
            return fetcher()
        return fetcher(session)

    def sync(self, session: Session) -> SyncResult:
        payload = self._fetch_payload_for_sync(session)
        processed_records, details = self.apply_payload(session, payload)
        details = {
            **details,
            "base_url": self.base_url.rstrip("/"),
            "endpoint_url": self.endpoint_url,
        }
        return SyncResult(
            source_id=self.source_id,
            processed_records=processed_records,
            status="completed",
            message=f"HTTP {self.source_id} payload synchronized.",
            adapter_key=self.adapter_key,
            transport=self.transport,
            details=details,
        )

    def apply_payload(self, session: Session, payload: object) -> tuple[int, dict]:
        raise NotImplementedError


class IdeamHttpAdapter(BaseHttpAdapter):
    source_id = "IDEAM"
    endpoint_path = "PrecipitacionNacionalDiaria.zip"
    adapter_key = "http.ideam"

    def _list_municipalities(self, session: Session) -> list[Municipality]:
        statement = select(Municipality).order_by(Municipality.name)
        return list(session.scalars(statement).all())

    def _station_catalog_url(self, municipality: Municipality) -> str:
        center_lat, center_lng = municipality.center
        radius = self.settings.ideam_station_search_radius_degrees
        query = urlencode(
            {
                "f": "json",
                "where": (
                    "idestadoestaciontm='ESTA001' AND idcategoria IN "
                    "('AM','CO','CP','ME','PG','PM','SP','SS')"
                ),
                "geometry": (
                    f"{center_lng - radius},{center_lat - radius},"
                    f"{center_lng + radius},{center_lat + radius}"
                ),
                "geometryType": "esriGeometryEnvelope",
                "inSR": "4326",
                "spatialRel": "esriSpatialRelIntersects",
                "outFields": "idestacion,nombre,idcategoria,latitud,longitud",
                "returnGeometry": "false",
                "resultRecordCount": "100",
            }
        )
        return f"{self.settings.ideam_station_catalog_url}?{query}"

    @staticmethod
    def _distance(municipality: Municipality, latitude: float, longitude: float) -> float:
        center_lat, center_lng = municipality.center
        return ((latitude - center_lat) ** 2 + (longitude - center_lng) ** 2) ** 0.5

    def _fetch_station_candidates(
        self, municipality: Municipality
    ) -> list[dict[str, object]]:
        payload = self._read_or_download_json(
            url=self._station_catalog_url(municipality),
            cache_name=f"ideam-stations-{municipality.id.lower()}.json",
            max_age_minutes=self.settings.ideam_cache_max_age_minutes,
            apply_auth=False,
        )
        if not isinstance(payload, dict):
            return []
        candidates: list[dict[str, object]] = []
        for feature in payload.get("features", []):
            attributes = feature.get("attributes") if isinstance(feature, dict) else None
            if not isinstance(attributes, dict):
                continue
            station_id = str(attributes.get("idestacion") or "").strip()
            category = str(attributes.get("idcategoria") or "").strip().upper()
            latitude = attributes.get("latitud")
            longitude = attributes.get("longitud")
            if (
                not station_id
                or category not in IDEAM_PRECIPITATION_CATEGORIES
                or latitude is None
                or longitude is None
            ):
                continue
            candidates.append(
                {
                    "station_id": station_id,
                    "station_name": str(attributes.get("nombre") or station_id),
                    "category": category,
                    "latitude": float(latitude),
                    "longitude": float(longitude),
                    "distance": self._distance(
                        municipality,
                        float(latitude),
                        float(longitude),
                    ),
                }
            )
        return candidates

    @staticmethod
    def _parse_station_series(
        archive: zipfile.ZipFile, station_id: str
    ) -> list[tuple[date, float]]:
        entry_name = f"PTPM_CON_INTER@{station_id}.data"
        try:
            raw = archive.read(entry_name).decode("latin-1", errors="ignore")
        except KeyError:
            return []
        reader = csv.reader(io.StringIO(raw), delimiter="|")
        next(reader, None)
        points: list[tuple[date, float]] = []
        for row in reader:
            if len(row) < 2:
                continue
            timestamp_text = row[0].strip()
            value_text = row[1].strip()
            if not timestamp_text or not value_text:
                continue
            try:
                point_date = datetime.fromisoformat(timestamp_text).date()
                value = float(value_text)
            except ValueError:
                continue
            points.append((point_date, value))
        return points

    def _select_station_series(
        self, archive: zipfile.ZipFile, municipality: Municipality
    ) -> tuple[list[dict[str, object]], dict[date, list[float]], date | None]:
        candidates = self._fetch_station_candidates(municipality)
        station_series: list[dict[str, object]] = []
        for candidate in candidates:
            station_id = str(candidate["station_id"])
            series = self._parse_station_series(archive, station_id)
            if not series:
                continue
            latest_date = series[-1][0]
            station_series.append(
                {
                    **candidate,
                    "latest_date": latest_date,
                    "series": series[-self.settings.ideam_history_days :],
                }
            )

        station_series.sort(
            key=lambda item: (
                (datetime.now(timezone.utc).date() - item["latest_date"]).days,
                float(item["distance"]),
            )
        )
        selected = station_series[: self.settings.ideam_station_limit_per_municipality]
        aggregated: dict[date, list[float]] = {}
        latest_available: date | None = None
        for item in selected:
            for point_date, value in item["series"]:
                aggregated.setdefault(point_date, []).append(value)
                latest_available = (
                    point_date
                    if latest_available is None or point_date > latest_available
                    else latest_available
                )
        return selected, aggregated, latest_available

    def fetch_payload(self, session: Session | None = None) -> object:
        return self._read_or_download_bytes(
            url=self.endpoint_url,
            cache_name="ideam-precipitacion-diaria.zip",
            max_age_minutes=self.settings.ideam_cache_max_age_minutes,
        )

    def apply_payload(self, session: Session, payload: object) -> tuple[int, dict]:
        if isinstance(payload, dict):
            rain_series, parse_details = parse_ideam_payload(payload)
            processed_records = sync_rain_series(session, rain_series)
            return processed_records, {
                "dataset": "municipality_rain_points",
                "municipality_count": len(rain_series),
                **parse_details,
            }

        if not isinstance(payload, (bytes, bytearray)):
            raise RuntimeError("IDEAM payload must be a normalized object or the official ZIP archive bytes.")

        archive = zipfile.ZipFile(io.BytesIO(payload))

        rain_series: dict[str, list[dict[str, object]]] = {}
        selected_station_ids: dict[str, list[str]] = {}
        latest_provider_date: date | None = None

        for municipality in self._list_municipalities(session):
            selected, aggregated, municipality_latest_date = self._select_station_series(
                archive, municipality
            )
            selected_station_ids[municipality.name] = [
                str(item["station_id"]) for item in selected
            ]
            time_points = sorted(aggregated.items())[-self.settings.ideam_history_days :]
            rain_series[municipality.name] = [
                {
                    "time": point_date.isoformat(),
                    "observed": round(sum(values) / len(values), 1),
                }
                for point_date, values in time_points
            ]
            if municipality_latest_date and (
                latest_provider_date is None
                or municipality_latest_date > latest_provider_date
            ):
                latest_provider_date = municipality_latest_date

        processed_records = sync_rain_series(session, rain_series)
        details = {
            "dataset": "municipality_rain_points",
            "payload_mode": "official_ideam_daily_zip",
            "municipality_count": len(rain_series),
            "selected_stations": selected_station_ids,
        }
        if latest_provider_date is not None:
            details["provider_updated_at"] = datetime.combine(
                latest_provider_date,
                time_of_day(7, 0),
                tzinfo=timezone.utc,
            ).isoformat()
        return processed_records, details

    def sync(self, session: Session) -> SyncResult:
        payload = self._fetch_payload_for_sync(session)
        processed_records, details = self.apply_payload(session, payload)
        details = {
            **details,
            "base_url": self.base_url.rstrip("/"),
            "endpoint_url": self.endpoint_url,
        }
        return SyncResult(
            source_id=self.source_id,
            processed_records=processed_records,
            status="completed",
            message="HTTP IDEAM official precipitation files synchronized.",
            adapter_key=self.adapter_key,
            transport=self.transport,
            details=details,
        )


class SgcHttpAdapter(BaseHttpAdapter):
    source_id = "SGC"
    endpoint_path = "inventario"
    adapter_key = "http.sgc"

    def _list_target_municipalities(self, session: Session) -> set[str]:
        statement = select(Municipality.name)
        return {
            normalize_lookup_key(name)
            for name in session.scalars(statement).all()
        }

    def fetch_payload(self, session: Session | None = None) -> object:
        if session is None:
            return super().fetch_payload()

        target_municipalities = self._list_target_municipalities(session)
        cache_name = "sgc-inventario-filtered.json"
        cache_path = self._cache_path(cache_name)
        if self._cache_is_fresh(cache_path, self.settings.sgc_cache_max_age_minutes):
            return json.loads(cache_path.read_text(encoding="utf-8"))

        filtered_events: list[dict] = []
        total_elements: int | None = None
        pages_fetched = 0
        for page in range(self.settings.sgc_max_pages):
            query = urlencode(
                {
                    "size": self.settings.sgc_page_size,
                    "page": page,
                    "sort": "drFechaEvento,desc",
                }
            )
            payload = json.loads(
                self._request(f"{self.endpoint_url}?{query}").decode("utf-8")
            )
            if not isinstance(payload, dict):
                break
            total_elements = payload.get("totalElements") or total_elements
            content = payload.get("content") or []
            pages_fetched += 1
            for item in content:
                municipality_block = item.get("municipio") if isinstance(item, dict) else None
                municipality_name = (
                    municipality_block.get("nombre")
                    if isinstance(municipality_block, dict)
                    else municipality_block
                )
                if municipality_name is None:
                    continue
                if normalize_lookup_key(str(municipality_name)) in target_municipalities:
                    filtered_events.append(item)
            if payload.get("last") or not content:
                break

        merged_payload = {
            "content": filtered_events,
            "totalElements": total_elements if total_elements is not None else len(filtered_events),
            "request_meta": {
                "provider": "official_simma_api",
                "pages_fetched": pages_fetched,
                "requested_page_size": self.settings.sgc_page_size,
                "requested_max_pages": self.settings.sgc_max_pages,
            },
        }
        cache_path.write_text(
            json.dumps(merged_payload, ensure_ascii=False),
            encoding="utf-8",
        )
        return merged_payload

    def apply_payload(self, session: Session, payload: object) -> tuple[int, dict]:
        events, parse_details = parse_sgc_payload(payload)
        processed_records = sync_historical_events(session, events)
        provider_updated_at = max((item["date"] for item in events), default=None)
        details = {
            "dataset": "historical_events",
            "record_type": "mass_movement_event",
            **parse_details,
        }
        if provider_updated_at:
            details["provider_updated_at"] = datetime.combine(
                date.fromisoformat(provider_updated_at),
                time_of_day.min,
                tzinfo=timezone.utc,
            ).isoformat()
        return processed_records, details

    def sync(self, session: Session) -> SyncResult:
        return super().sync(session)


class UngrdHttpAdapter(BaseHttpAdapter):
    source_id = "UNGRD"
    endpoint_path = "4fd8-ptcr.json"
    adapter_key = "http.ungrd"

    def fetch_payload(self, session: Session | None = None) -> object:
        if session is None:
            return super().fetch_payload()

        municipality_names = list(
            session.scalars(select(Municipality.name).order_by(Municipality.name)).all()
        )
        filters = ",".join(
            f"'{normalize_lookup_key(name)}'" for name in municipality_names
        )
        cache_name = "ungrd-records-filtered.json"
        cache_path = self._cache_path(cache_name)
        if self._cache_is_fresh(cache_path, self.settings.ungrd_cache_max_age_minutes):
            return json.loads(cache_path.read_text(encoding="utf-8"))

        merged_records: list[dict] = []
        pages_fetched = 0
        for page in range(self.settings.ungrd_max_pages):
            query = urlencode(
                {
                    "$select": ":id,fecha,municipio,evento,personas,familias,divipola",
                    "$where": f"upper(municipio) in ({filters})",
                    "$order": "fecha DESC",
                    "$limit": self.settings.ungrd_page_size,
                    "$offset": page * self.settings.ungrd_page_size,
                }
            )
            payload = json.loads(
                self._request(f"{self.endpoint_url}?{query}").decode("utf-8")
            )
            if not isinstance(payload, list):
                raise RuntimeError("UNGRD official payload must be a record list.")
            pages_fetched += 1
            merged_records.extend(
                item for item in payload if isinstance(item, dict)
            )
            if len(payload) < self.settings.ungrd_page_size:
                break

        merged_payload = {
            "records": merged_records,
            "request_meta": {
                "provider": "official_socrata_api",
                "pages_fetched": pages_fetched,
                "requested_page_size": self.settings.ungrd_page_size,
                "requested_max_pages": self.settings.ungrd_max_pages,
            },
        }
        cache_path.write_text(
            json.dumps(merged_payload, ensure_ascii=False),
            encoding="utf-8",
        )
        return merged_payload

    def apply_payload(self, session: Session, payload: object) -> tuple[int, dict]:
        ungrd_records, parse_details = parse_ungrd_payload(payload)
        processed_records = sync_ungrd_records(session, ungrd_records)
        latest_date = max(
            (
                record["date"]
                for records in ungrd_records.values()
                for record in records
            ),
            default=None,
        )
        details = {
            "dataset": "ungrd_records",
            "municipality_count": len(ungrd_records),
            **parse_details,
        }
        if latest_date:
            details["provider_updated_at"] = datetime.combine(
                date.fromisoformat(latest_date),
                time_of_day.min,
                tzinfo=timezone.utc,
            ).isoformat()
        return processed_records, details

    def sync(self, session: Session) -> SyncResult:
        return super().sync(session)


HTTP_ADAPTERS = {
    "IDEAM": IdeamHttpAdapter,
    "SGC": SgcHttpAdapter,
    "UNGRD": UngrdHttpAdapter,
}

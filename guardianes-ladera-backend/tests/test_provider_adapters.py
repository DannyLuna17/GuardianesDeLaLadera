from __future__ import annotations

import json

from app.integrations import provider_adapters


class _DummyHttpAdapter(provider_adapters.BaseHttpAdapter):
    source_id = "IDEAM"
    adapter_key = "http.ideam"


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def read(self) -> bytes:
        return self.payload


def test_base_http_adapter_retries_timeout_errors(monkeypatch):
    monkeypatch.setenv("PROVIDER_REQUEST_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("PROVIDER_REQUEST_RETRY_ATTEMPTS", "3")
    monkeypatch.setenv("PROVIDER_REQUEST_RETRY_BACKOFF_SECONDS", "0")

    from app.core.config import get_settings

    get_settings.cache_clear()

    attempts = {"count": 0}

    def fake_urlopen(request, timeout):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise TimeoutError("provider timed out")
        return _FakeResponse(b'{"ok": true}')

    monkeypatch.setattr(provider_adapters, "urlopen", fake_urlopen)
    adapter = _DummyHttpAdapter()

    payload = adapter._request("https://provider.gov.co/test")

    assert payload == b'{"ok": true}'
    assert attempts["count"] == 3


def test_base_http_adapter_uses_sleep_backoff_without_datetime_collision(monkeypatch):
    monkeypatch.setenv("PROVIDER_REQUEST_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("PROVIDER_REQUEST_RETRY_ATTEMPTS", "2")
    monkeypatch.setenv("PROVIDER_REQUEST_RETRY_BACKOFF_SECONDS", "0.25")

    from app.core.config import get_settings

    get_settings.cache_clear()

    attempts = {"count": 0}
    sleeps: list[float] = []

    def fake_urlopen(request, timeout):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise TimeoutError("provider timed out")
        return _FakeResponse(b'{"ok": true}')

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(provider_adapters, "urlopen", fake_urlopen)
    monkeypatch.setattr(provider_adapters.time_module, "sleep", fake_sleep)
    adapter = _DummyHttpAdapter()

    payload = adapter._request("https://provider.gov.co/test")

    assert payload == b'{"ok": true}'
    assert attempts["count"] == 2
    assert sleeps == [0.25]


def test_base_http_adapter_applies_source_auth_header_and_query_param(monkeypatch):
    monkeypatch.setenv("IDEAM_AUTH_HEADER_NAME", "Authorization")
    monkeypatch.setenv("IDEAM_AUTH_TOKEN", "Bearer secret-token")
    monkeypatch.setenv("IDEAM_AUTH_QUERY_PARAM", "api_key")

    from app.core.config import get_settings

    get_settings.cache_clear()

    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        return _FakeResponse(b'{"ok": true}')

    monkeypatch.setattr(provider_adapters, "urlopen", fake_urlopen)
    adapter = _DummyHttpAdapter()

    payload = adapter._request("https://provider.gov.co/test?existing=1")

    assert payload == b'{"ok": true}'
    assert captured["url"] == "https://provider.gov.co/test?existing=1&api_key=Bearer+secret-token"
    assert captured["headers"]["Authorization"] == "Bearer secret-token"


def test_sgc_http_adapter_fetch_payload_paginates_and_filters(monkeypatch, tmp_path):
    monkeypatch.setenv("PROVIDER_CACHE_PATH", str(tmp_path))
    monkeypatch.setenv("SGC_PAGE_SIZE", "2")
    monkeypatch.setenv("SGC_MAX_PAGES", "5")
    monkeypatch.setenv("SGC_BASE_URL", "https://provider.test/sgc/")

    from app.core.config import get_settings

    get_settings.cache_clear()

    class _ScalarResult:
        def all(self):
            return ["Mocoa"]

    class _Session:
        def scalars(self, statement):
            return _ScalarResult()

    requests: list[str] = []
    page_payloads = {
        0: {
            "content": [
                {
                    "id": "sgc-a",
                    "municipio": {"nombre": "Mocoa"},
                    "drFechaEvento": "2026-03-28T00:00:00Z",
                    "lgLatitud": 1.1,
                    "lgLongitud": -76.6,
                },
                {
                    "id": "sgc-b",
                    "municipio": {"nombre": "Pasto"},
                    "drFechaEvento": "2026-03-27T00:00:00Z",
                    "lgLatitud": 1.2,
                    "lgLongitud": -76.7,
                },
            ],
            "totalElements": 3,
            "last": False,
        },
        1: {
            "content": [
                {
                    "id": "sgc-c",
                    "municipio": {"nombre": "Mocoa"},
                    "drFechaEvento": "2026-03-26T00:00:00Z",
                    "lgLatitud": 1.3,
                    "lgLongitud": -76.8,
                }
            ],
            "totalElements": 3,
            "last": True,
        },
    }

    def fake_request(url, *, apply_auth=True):
        requests.append(url)
        page = int(url.split("page=")[1].split("&")[0])
        return json.dumps(page_payloads[page]).encode("utf-8")

    adapter = provider_adapters.SgcHttpAdapter()
    monkeypatch.setattr(adapter, "_request", fake_request)

    payload = adapter.fetch_payload(_Session())

    assert [item["id"] for item in payload["content"]] == ["sgc-a", "sgc-c"]
    assert payload["request_meta"]["pages_fetched"] == 2
    assert payload["request_meta"]["requested_page_size"] == 2
    assert len(requests) == 2


def test_ungrd_http_adapter_fetch_payload_paginates_records(monkeypatch, tmp_path):
    monkeypatch.setenv("PROVIDER_CACHE_PATH", str(tmp_path))
    monkeypatch.setenv("UNGRD_PAGE_SIZE", "2")
    monkeypatch.setenv("UNGRD_MAX_PAGES", "5")
    monkeypatch.setenv("UNGRD_BASE_URL", "https://provider.test/ungrd/")

    from app.core.config import get_settings

    get_settings.cache_clear()

    class _ScalarResult:
        def all(self):
            return ["Mocoa", "Pasto"]

    class _Session:
        def scalars(self, statement):
            return _ScalarResult()

    requests: list[str] = []
    page_payloads = {
        0: [
            {
                ":id": "1",
                "fecha": "2026-03-28T00:00:00Z",
                "municipio": "Mocoa",
                "evento": "Deslizamiento",
                "personas": "5",
            },
            {
                ":id": "2",
                "fecha": "2026-03-27T00:00:00Z",
                "municipio": "Pasto",
                "evento": "Avenida torrencial",
                "familias": "3",
            },
        ],
        1: [
            {
                ":id": "3",
                "fecha": "2026-03-26T00:00:00Z",
                "municipio": "Mocoa",
                "evento": "Movimiento en masa",
            }
        ],
    }

    def fake_request(url, *, apply_auth=True):
        requests.append(url)
        offset = int(url.split("%24offset=")[1].split("&")[0])
        page = offset // 2
        return json.dumps(page_payloads[page]).encode("utf-8")

    adapter = provider_adapters.UngrdHttpAdapter()
    monkeypatch.setattr(adapter, "_request", fake_request)

    payload = adapter.fetch_payload(_Session())

    assert [item[":id"] for item in payload["records"]] == ["1", "2", "3"]
    assert payload["request_meta"]["pages_fetched"] == 2
    assert payload["request_meta"]["requested_page_size"] == 2
    assert len(requests) == 2

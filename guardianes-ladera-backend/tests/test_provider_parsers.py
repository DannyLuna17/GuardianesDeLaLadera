from app.integrations.provider_parsers import parse_ideam_payload, parse_sgc_payload, parse_ungrd_payload


def test_parse_ideam_provider_payload_normalizes_municipality_series():
    payload = {
        "data": {
            "municipalities": [
                {
                    "nombre": "Mocoa",
                    "mediciones": [
                        {
                            "timestamp": "2026-03-24T03:00:00Z",
                            "observed_mm": 14,
                            "forecast_mm": 20,
                            "forecast_range": {"min": 12, "max": 25},
                        },
                        {
                            "timestamp": "2026-03-24T06:00:00Z",
                            "forecast_mm": 18,
                            "forecast_min": 10,
                            "forecast_max": 24,
                        },
                    ],
                }
            ]
        }
    }

    normalized, details = parse_ideam_payload(payload)

    assert list(normalized) == ["Mocoa"]
    assert normalized["Mocoa"][0]["time"] == "2026-03-24 03:00"
    assert normalized["Mocoa"][0]["observed"] == 14.0
    assert normalized["Mocoa"][0]["forecast"] == 20.0
    assert normalized["Mocoa"][0]["forecastLow"] == 12.0
    assert normalized["Mocoa"][0]["forecastHigh"] == 25.0
    assert normalized["Mocoa"][0]["forecastRange"] == 13.0
    assert details["payload_mode"] == "provider_http_municipalities"
    assert details["point_count"] == 2


def test_parse_sgc_provider_payload_normalizes_events_and_geojson_coordinates():
    payload = {
        "events": [
            {
                "event_id": "sgc-http-01",
                "municipio": "Pasto",
                "fecha": "2026-02-11T14:30:00Z",
                "nivel": "alta",
                "movement_type": "Deslizamiento",
                "geometry": {"type": "Point", "coordinates": [-77.274, 1.225]},
            }
        ]
    }

    normalized, details = parse_sgc_payload(payload)

    assert normalized == [
        {
            "id": "sgc-http-01",
            "municipality": "Pasto",
            "date": "2026-02-11",
            "severity": "Alta",
            "type": "Deslizamiento",
            "coords": [1.225, -77.274],
            "source": "SGC",
        }
    ]
    assert details["payload_mode"] == "provider_http_events"
    assert details["record_count"] == 1


def test_parse_sgc_official_simma_payload_normalizes_nested_fields():
    payload = {
        "content": [
            {
                "id": 31553,
                "drFechaEvento": [2026, 3, 11],
                "municipio": {"nombre": "Mocoa"},
                "lgLatitud": 1.155,
                "lgLongitud": -76.661,
                "estadoErosion": {"valor": "Moderada"},
                "tipoMovimiento": {"valor": "Deslizamiento"},
            }
        ],
        "totalElements": 1,
    }

    normalized, details = parse_sgc_payload(payload)

    assert normalized == [
        {
            "id": "31553",
            "municipality": "Mocoa",
            "date": "2026-03-11",
            "severity": "Media",
            "type": "Deslizamiento",
            "coords": [1.155, -76.661],
            "source": "SGC",
        }
    ]
    assert details["payload_mode"] == "official_simma_api"
    assert details["record_count"] == 1


def test_parse_ungrd_provider_payload_groups_records_by_municipality():
    payload = {
        "reports": [
            {
                "recordId": "ungrd-http-01",
                "municipio": "Popayan",
                "reported_at": "2026-01-09T09:15:00Z",
                "resumen": "Cierre preventivo de corredor por saturacion de ladera.",
            },
            {
                "recordId": "ungrd-http-02",
                "municipality": "Popayan",
                "date": "2026-01-10",
                "summary": "Seguimiento tecnico con gestion del riesgo municipal.",
            },
        ]
    }

    normalized, details = parse_ungrd_payload(payload)

    assert list(normalized) == ["Popayan"]
    assert len(normalized["Popayan"]) == 2
    assert normalized["Popayan"][0]["id"] == "ungrd-http-01"
    assert normalized["Popayan"][0]["date"] == "2026-01-09"
    assert normalized["Popayan"][1]["summary"] == "Seguimiento tecnico con gestion del riesgo municipal."
    assert details["payload_mode"] == "provider_http_records"
    assert details["record_count"] == 2


def test_parse_ungrd_official_socrata_payload_groups_records():
    payload = [
        {
            ":id": "row-test-1",
            "fecha": "2026-03-12T00:00:00.000",
            "municipio": "MOCOA",
            "evento": "MOVIMIENTO EN MASA",
            "personas": "12",
        }
    ]

    normalized, details = parse_ungrd_payload(payload)

    assert list(normalized) == ["MOCOA"]
    assert normalized["MOCOA"][0]["id"] == "row-test-1"
    assert normalized["MOCOA"][0]["date"] == "2026-03-12"
    assert "MOVIMIENTO EN MASA" in normalized["MOCOA"][0]["summary"]
    assert details["payload_mode"] == "official_socrata_api"

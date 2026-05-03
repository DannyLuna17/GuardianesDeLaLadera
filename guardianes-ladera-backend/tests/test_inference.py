from app.ml.inference import BaselineInferenceService
from app.ml.features import ZoneFeatureSnapshot
from app.ml.model_registry import ModelRegistry
from app.ml.training import export_beta_regression_artifact


def test_model_registry_loads_spatial_baseline_artifact():
    registry = ModelRegistry()
    artifact = registry.load("trained-spatial-seed-v1")
    assert artifact["model_id"] == "trained-spatial-seed-linear-regression"
    assert artifact["version"] == "trained-spatial-seed-v1"


def test_baseline_inference_returns_bounded_score_and_trace():
    service = BaselineInferenceService()
    result = service.predict(
        zone_id="moc-01",
        previous_drivers={
            "rain_6h": 40,
            "rain_24h": 100,
            "rain_72h": 180,
            "slope_deg": 31,
            "geology_class": "Coluvion activo",
            "soil_class": "Franco arcilloso",
            "deforestation_proxy": 0.62,
        },
        run_index=4,
        snapshot={
            "IDEAM": "Fresco",
            "NASA": "Fresco",
            "UNGRD": "Retrasado",
            "SENTINEL": "Desactualizado",
            "SGC": "Estatico",
        },
        feature_snapshot=ZoneFeatureSnapshot(
            municipality_event_count=3,
            zone_event_count=2,
            recent_zone_event_count=1,
            intersecting_road_count=2,
            intersecting_road_length_km=70.0,
            rain_overlay_count=2,
            rain_overlay_peak_intensity=3,
            rain_overlay_peak_label="alta",
        ),
    )
    assert 0.08 <= result.score <= 0.92
    assert result.confidence in {"Alta", "Media", "Baja"}
    assert result.trace["model_version"] == "trained-spatial-seed-v1"
    assert result.trace["uses_spatial_features"] is True
    assert result.trace["feature_snapshot"]["zone_event_count"] == 2
    assert "intersecting_road_length_km" in result.trace["component_scores"]
    assert result.trace["artifact_type"] == "trained_linear_model"
    assert result.trace["calibration_method"] in {"affine", "identity"}
    assert "raw_model_score" in result.trace
    assert "calibrated_model_score_before_freshness" in result.trace


def test_inference_supports_beta_regression_active_artifact(tmp_path):
    artifacts_path = tmp_path / "artifacts"
    manifest_path = tmp_path / "active-model.json"
    registry = ModelRegistry(
        artifacts_path=artifacts_path,
        manifest_path=manifest_path,
    )
    export_beta_regression_artifact(
        version="beta-inference-v1",
        alpha=0.75,
        artifacts_path=artifacts_path,
    )
    registry.set_active_version("beta-inference-v1")
    service = BaselineInferenceService(model_registry=registry)

    result = service.predict(
        zone_id="moc-01",
        previous_drivers={
            "rain_6h": 42,
            "rain_24h": 108,
            "rain_72h": 185,
            "slope_deg": 31,
            "geology_class": "Coluvion activo",
            "soil_class": "Franco arcilloso",
            "deforestation_proxy": 0.62,
        },
        run_index=4,
        snapshot={
            "IDEAM": "Fresco",
            "NASA": "Fresco",
            "UNGRD": "Retrasado",
            "SENTINEL": "Desactualizado",
            "SGC": "Estatico",
        },
        feature_snapshot=ZoneFeatureSnapshot(
            municipality_event_count=3,
            zone_event_count=2,
            recent_zone_event_count=1,
            intersecting_road_count=2,
            intersecting_road_length_km=70.0,
            rain_overlay_count=2,
            rain_overlay_peak_intensity=3,
            rain_overlay_peak_label="alta",
        ),
    )
    assert 0.08 <= result.score <= 0.92
    assert result.trace["artifact_type"] == "beta_regression_model"
    assert result.trace["component_score_space"] == "logit"
    assert result.trace["calibration_method"] in {"affine", "identity"}

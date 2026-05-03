from __future__ import annotations

from dataclasses import dataclass

from app.core.config import get_settings
from app.db.bootstrap import clamp
from app.ml.additive_splines import predict_additive_spline_regressor
from app.ml.beta_regression import predict_beta_regression_model
from app.ml.features import ZoneFeatureSnapshot, build_scoring_feature_vector
from app.ml.model_registry import ModelRegistry
from app.ml.tree_boosting import predict_gradient_boosted_ensemble
from app.ml.training import apply_affine_calibration
from app.ml.xgboost_models import predict_xgboost_model


@dataclass
class InferenceResult:
    score: float
    confidence: str
    drivers: dict
    trace: dict


class BaselineInferenceService:
    def __init__(self, model_registry: ModelRegistry | None = None) -> None:
        self.settings = get_settings()
        self.model_registry = model_registry or ModelRegistry()

    @staticmethod
    def stable_wave(zone_id: str, run_index: int, offset: int) -> int:
        value = sum(ord(char) for char in zone_id) + run_index * 17 + offset * 11
        return (value % 9) - 4

    def evolve_drivers(
        self, zone_id: str, previous_drivers: dict, run_index: int
    ) -> dict:
        return {
            **previous_drivers,
            "rain_6h": max(
                0,
                int(previous_drivers["rain_6h"])
                + self.stable_wave(zone_id, run_index, 1) * 2,
            ),
            "rain_24h": max(
                0,
                int(previous_drivers["rain_24h"])
                + self.stable_wave(zone_id, run_index, 2) * 4,
            ),
            "rain_72h": max(
                0,
                int(previous_drivers["rain_72h"])
                + self.stable_wave(zone_id, run_index, 3) * 6,
            ),
        }

    def confidence_from_snapshot(self, snapshot: dict[str, str], artifact: dict) -> str:
        dynamic_status = [
            status
            for source_id, status in snapshot.items()
            if source_id in {"IDEAM", "NASA", "UNGRD", "SENTINEL"}
        ]
        score = sum(artifact["confidence_scores"][status] for status in dynamic_status)
        if score >= 6:
            return "Alta"
        if score >= 3:
            return "Media"
        return "Baja"

    @staticmethod
    def freshness_penalty_total(snapshot: dict[str, str], artifact: dict) -> float:
        total = 0.0
        for freshness, penalty in artifact["freshness_penalties"].items():
            total += sum(penalty for status in snapshot.values() if status == freshness)
        return total

    @staticmethod
    def score_weighted_sum_artifact(
        drivers: dict, feature_snapshot: ZoneFeatureSnapshot, artifact: dict
    ) -> tuple[float, dict]:
        weights = artifact["weights"]
        normalizers = artifact["normalizers"]
        component_scores = {
            "rain_72h": min(float(drivers["rain_72h"]) / normalizers["rain_72h"], 1.0)
            * weights["rain_72h"],
            "rain_24h": min(float(drivers["rain_24h"]) / normalizers["rain_24h"], 1.0)
            * weights["rain_24h"],
            "slope_deg": min(
                float(drivers["slope_deg"]) / normalizers["slope_deg"], 1.0
            )
            * weights["slope_deg"],
            "municipality_event_count": min(
                float(feature_snapshot.municipality_event_count)
                / normalizers["municipality_event_count"],
                1.0,
            )
            * weights["municipality_event_count"],
            "zone_event_count": min(
                float(feature_snapshot.zone_event_count)
                / normalizers["zone_event_count"],
                1.0,
            )
            * weights["zone_event_count"],
            "intersecting_road_length_km": min(
                float(feature_snapshot.intersecting_road_length_km)
                / normalizers["intersecting_road_length_km"],
                1.0,
            )
            * weights["intersecting_road_length_km"],
            "rain_overlay_peak_intensity": min(
                float(feature_snapshot.rain_overlay_peak_intensity)
                / normalizers["rain_overlay_peak_intensity"],
                1.0,
            )
            * weights["rain_overlay_peak_intensity"],
            "rain_overlay_count": min(
                float(feature_snapshot.rain_overlay_count)
                / normalizers["rain_overlay_count"],
                1.0,
            )
            * weights["rain_overlay_count"],
            "deforestation_proxy": min(
                float(drivers.get("deforestation_proxy") or 0.0)
                / normalizers["deforestation_proxy"],
                1.0,
            )
            * weights["deforestation_proxy"],
        }
        return sum(component_scores.values()), component_scores

    @staticmethod
    def score_trained_linear_artifact(
        drivers: dict, feature_snapshot: ZoneFeatureSnapshot, artifact: dict
    ) -> tuple[float, dict, dict]:
        feature_vector = build_scoring_feature_vector(drivers, feature_snapshot)
        standardized_vector = {
            feature_name: (
                (feature_vector[feature_name] - artifact["feature_means"][feature_name])
                / artifact["feature_scales"][feature_name]
            )
            for feature_name in artifact["feature_order"]
        }
        component_scores = {
            feature_name: standardized_vector[feature_name]
            * artifact["coefficients"][feature_name]
            for feature_name in artifact["feature_order"]
        }
        score = artifact["intercept"] + sum(component_scores.values())
        return score, component_scores, feature_vector

    @staticmethod
    def score_gradient_boosted_tree_artifact(
        drivers: dict, feature_snapshot: ZoneFeatureSnapshot, artifact: dict
    ) -> tuple[float, dict, dict]:
        feature_vector = build_scoring_feature_vector(drivers, feature_snapshot)
        raw_score, component_scores = predict_gradient_boosted_ensemble(
            artifact,
            feature_vector,
            with_contributions=True,
        )
        return float(raw_score), component_scores, feature_vector

    @staticmethod
    def score_additive_spline_artifact(
        drivers: dict, feature_snapshot: ZoneFeatureSnapshot, artifact: dict
    ) -> tuple[float, dict, dict]:
        feature_vector = build_scoring_feature_vector(drivers, feature_snapshot)
        raw_score, component_scores = predict_additive_spline_regressor(
            artifact,
            feature_vector,
            with_contributions=True,
        )
        return float(raw_score), component_scores, feature_vector

    @staticmethod
    def score_beta_regression_artifact(
        drivers: dict, feature_snapshot: ZoneFeatureSnapshot, artifact: dict
    ) -> tuple[float, dict, dict]:
        feature_vector = build_scoring_feature_vector(drivers, feature_snapshot)
        raw_score, component_scores, _ = predict_beta_regression_model(
            artifact,
            feature_vector,
            with_contributions=True,
        )
        return float(raw_score), component_scores, feature_vector

    @staticmethod
    def score_xgboost_artifact(
        drivers: dict, feature_snapshot: ZoneFeatureSnapshot, artifact: dict
    ) -> tuple[float, dict, dict]:
        feature_vector = build_scoring_feature_vector(drivers, feature_snapshot)
        raw_score, component_scores, _ = predict_xgboost_model(
            artifact,
            feature_vector,
            with_contributions=True,
        )
        return float(raw_score), component_scores, feature_vector

    def predict(
        self,
        zone_id: str,
        previous_drivers: dict,
        run_index: int,
        snapshot: dict[str, str],
        feature_snapshot: ZoneFeatureSnapshot,
    ) -> InferenceResult:
        artifact = self.model_registry.load_active()
        drivers = self.evolve_drivers(
            zone_id=zone_id, previous_drivers=previous_drivers, run_index=run_index
        )
        feature_vector = build_scoring_feature_vector(drivers, feature_snapshot)

        if artifact.get("artifact_type") == "trained_linear_model":
            raw_score, component_scores, feature_vector = (
                self.score_trained_linear_artifact(
                    drivers,
                    feature_snapshot,
                    artifact,
                )
            )
            score = clamp(
                apply_affine_calibration(raw_score, artifact.get("calibration")),
                artifact["bounds"]["min"],
                artifact["bounds"]["max"],
            )
        elif artifact.get("artifact_type") == "gradient_boosted_tree_model":
            raw_score, component_scores, feature_vector = (
                self.score_gradient_boosted_tree_artifact(
                    drivers,
                    feature_snapshot,
                    artifact,
                )
            )
            score = clamp(
                apply_affine_calibration(raw_score, artifact.get("calibration")),
                artifact["bounds"]["min"],
                artifact["bounds"]["max"],
            )
        elif artifact.get("artifact_type") == "additive_spline_model":
            raw_score, component_scores, feature_vector = (
                self.score_additive_spline_artifact(
                    drivers,
                    feature_snapshot,
                    artifact,
                )
            )
            score = clamp(
                apply_affine_calibration(raw_score, artifact.get("calibration")),
                artifact["bounds"]["min"],
                artifact["bounds"]["max"],
            )
        elif artifact.get("artifact_type") == "beta_regression_model":
            raw_score, component_scores, feature_vector = (
                self.score_beta_regression_artifact(
                    drivers,
                    feature_snapshot,
                    artifact,
                )
            )
            score = clamp(
                apply_affine_calibration(raw_score, artifact.get("calibration")),
                artifact["bounds"]["min"],
                artifact["bounds"]["max"],
            )
        elif artifact.get("artifact_type") == "xgboost_model":
            raw_score, component_scores, feature_vector = (
                self.score_xgboost_artifact(
                    drivers,
                    feature_snapshot,
                    artifact,
                )
            )
            score = clamp(
                apply_affine_calibration(raw_score, artifact.get("calibration")),
                artifact["bounds"]["min"],
                artifact["bounds"]["max"],
            )
        else:
            raw_score, component_scores = self.score_weighted_sum_artifact(
                drivers,
                feature_snapshot,
                artifact,
            )
            score = raw_score

        freshness_penalty_total = self.freshness_penalty_total(snapshot, artifact)
        score -= freshness_penalty_total

        bounded_score = round(
            clamp(score, artifact["bounds"]["min"], artifact["bounds"]["max"]), 3
        )
        confidence = self.confidence_from_snapshot(snapshot, artifact)

        return InferenceResult(
            score=bounded_score,
            confidence=confidence,
            drivers=drivers,
            trace={
                "model_id": artifact["model_id"],
                "model_version": artifact["version"],
                "artifact_type": artifact.get("artifact_type", "weighted_sum"),
                "run_index": run_index,
                "feature_snapshot": feature_snapshot.as_dict(),
                "feature_vector": feature_vector,
                "raw_model_score": round(raw_score, 6),
                "calibrated_model_score_before_freshness": round(
                    score + freshness_penalty_total, 6
                ),
                "calibration_method": artifact.get("calibration", {}).get(
                    "method", "none"
                ),
                "component_scores": {
                    key: round(value, 3) for key, value in component_scores.items()
                },
                "component_score_space": artifact.get(
                    "component_score_space", "response"
                ),
                "freshness_penalty_total": round(freshness_penalty_total, 3),
                "uses_spatial_features": True,
            },
        )

from __future__ import annotations

from collections import defaultdict

from app.core.exceptions import ApiError
from app.ml.model_drift import ModelDriftReportRegistry
from app.ml.model_registry import ModelRegistry
from app.ml.model_selection import ModelSelectionRunRegistry
from app.ml.model_shadow import ModelShadowRunRegistry
from app.schemas.admin import (
    ModelDriftSummaryRead,
    ModelMonitoringDetailRead,
    ModelMonitoringFamilyRollupRead,
    ModelMonitoringSummaryRead,
    ModelPromotionHistoryEntryRead,
    ModelSelectionAppearanceRead,
    ModelShadowRunSummaryRead,
)


class ModelMonitoringService:
    def __init__(self) -> None:
        self.model_registry = ModelRegistry()
        self.selection_registry = ModelSelectionRunRegistry()
        self.drift_registry = ModelDriftReportRegistry()
        self.shadow_registry = ModelShadowRunRegistry()

    def _selection_runs(self) -> list[dict]:
        runs = [
            self.selection_registry.load(version)
            for version in self.selection_registry.list_versions()
        ]
        return sorted(runs, key=lambda item: item.get("created_at", ""), reverse=True)

    def _known_versions(self) -> set[str]:
        versions = set(self.model_registry.list_versions())
        for entry in self._promotion_entries():
            versions.add(entry.model_version)
        for run in self._selection_runs():
            for candidate in run.get("candidates") or []:
                versions.add(str(candidate.get("model_version") or "unknown"))
        for report in self._drift_reports():
            versions.add(report.model_version)
        for run in self._shadow_runs():
            shadow_detail = self.shadow_registry.load(run.version)
            for candidate in shadow_detail.get("candidates") or []:
                versions.add(str(candidate.get("model_version") or "unknown"))
        return versions

    def _artifact_meta(self, version: str) -> tuple[str, str]:
        try:
            artifact = self.model_registry.load(version)
        except ApiError:
            return "unknown", "unknown"
        return (
            str(artifact.get("model_id", "unknown")),
            str(artifact.get("artifact_type", "unknown")),
        )

    def _promotion_entries(self) -> list[ModelPromotionHistoryEntryRead]:
        manifest = self.model_registry.active_manifest() or {}
        current_active_version = self.model_registry.active_version()
        history_entries = list(manifest.get("history") or [])
        reads: list[ModelPromotionHistoryEntryRead] = []
        for entry in history_entries:
            source = str(entry.get("source") or "unknown")
            rollback = "rollback" in source
            previous_active = entry.get("previous_active_model_version")
            reads.append(
                ModelPromotionHistoryEntryRead(
                    modelVersion=str(entry.get("version") or "unknown"),
                    previousActiveModelVersion=previous_active,
                    rolledBackFromModelVersion=previous_active if rollback else None,
                    promotedAt=str(entry.get("promoted_at") or ""),
                    promotedBy=entry.get("promoted_by"),
                    reason=entry.get("reason"),
                    source=source,
                    rollback=rollback,
                    currentActive=str(entry.get("version") or "") == current_active_version,
                )
            )
        return sorted(reads, key=lambda item: item.promoted_at, reverse=True)

    def _drift_reports(self) -> list[ModelDriftSummaryRead]:
        reports: list[ModelDriftSummaryRead] = []
        for version in self.drift_registry.list_versions():
            report = self.drift_registry.load(version)
            current = report.get("current") or {}
            baseline = report.get("baseline") or {}
            summary = report.get("drift_summary") or {}
            dataset_context = current.get("dataset_context") or {}
            reports.append(
                ModelDriftSummaryRead(
                    version=version,
                    driftId=report.get("drift_id", "unknown"),
                    artifactType=report.get("artifact_type", "model_drift_report"),
                    createdAt=report.get("created_at", ""),
                    modelVersion=report.get("model_version", "unknown"),
                    datasetVersion=report.get("dataset_version", "unknown"),
                    evaluationVersion=report.get("evaluation_version", "unknown"),
                    severity=str(summary.get("severity") or "unavailable"),
                    driftDetected=bool(summary.get("drift_detected")),
                    baselineSource=baseline.get("source"),
                    baselineReferenceVersion=baseline.get("reference_version"),
                    baselineDatasetVersion=baseline.get("dataset_version"),
                    validationRmse=current.get("validation_rmse"),
                    baselineValidationRmse=baseline.get("validation_rmse"),
                    validationRmseDelta=summary.get("validation_rmse_delta"),
                    validationRiskLevelAccuracy=current.get(
                        "validation_risk_level_accuracy"
                    ),
                    baselineValidationRiskLevelAccuracy=baseline.get(
                        "validation_risk_level_accuracy"
                    ),
                    validationRiskLevelAccuracyDelta=summary.get(
                        "validation_risk_level_accuracy_delta"
                    ),
                    validationRows=int(current.get("validation_rows") or 0),
                    datasetFamily=str(
                        dataset_context.get("dataset_family") or "unknown"
                    ),
                    taxonomyGroup=(dataset_context.get("dataset_taxonomy") or {}).get(
                        "taxonomy_group"
                    ),
                    evaluationCohortLabel=(
                        dataset_context.get("evaluation_cohort") or {}
                    ).get("bucket_label"),
                )
            )
        return sorted(reports, key=lambda item: item.created_at, reverse=True)

    def _drift_history_for_version(self, version: str) -> list[ModelDriftSummaryRead]:
        return [report for report in self._drift_reports() if report.model_version == version]

    def _shadow_runs(self) -> list[ModelShadowRunSummaryRead]:
        runs: list[ModelShadowRunSummaryRead] = []
        for version in self.shadow_registry.list_versions():
            run = self.shadow_registry.load(version)
            runs.append(
                ModelShadowRunSummaryRead(
                    version=version,
                    shadowId=run.get("shadow_id", "unknown"),
                    artifactType=run.get("artifact_type", "model_shadow_run"),
                    datasetVersion=run.get("dataset_version", "unknown"),
                    createdAt=run.get("created_at", ""),
                    activeModelVersion=run.get("active_model_version", "unknown"),
                    bestModelVersion=run.get("best_model_version", "unknown"),
                    activeStillBest=bool(run.get("active_still_best")),
                    candidateCount=int(run.get("candidate_count", 0)),
                    recommendation=run.get("recommendation") or {},
                )
            )
        return sorted(runs, key=lambda item: item.created_at, reverse=True)

    def _shadow_history_for_version(self, version: str) -> list[ModelShadowRunSummaryRead]:
        history: list[ModelShadowRunSummaryRead] = []
        for run in self._shadow_runs():
            shadow_detail = self.shadow_registry.load(run.version)
            candidate_versions = {
                str(candidate.get("model_version") or "unknown")
                for candidate in shadow_detail.get("candidates") or []
            }
            if version in candidate_versions:
                history.append(run)
        return history

    def list_promotion_history(
        self, *, model_version: str | None = None
    ) -> list[ModelPromotionHistoryEntryRead]:
        entries = self._promotion_entries()
        if model_version is None:
            return entries
        return [entry for entry in entries if entry.model_version == model_version]

    def _selection_history_for_version(
        self, version: str
    ) -> list[ModelSelectionAppearanceRead]:
        appearances: list[ModelSelectionAppearanceRead] = []
        for run in self._selection_runs():
            run_context = dict(run.get("dataset_context") or {})
            for candidate in run.get("candidates") or []:
                candidate_version = str(candidate.get("model_version") or "unknown")
                if candidate_version != version:
                    continue
                appearances.append(
                    ModelSelectionAppearanceRead(
                        selectionVersion=run["version"],
                        datasetVersion=run["dataset_version"],
                        datasetFamily=str(
                            run_context.get("dataset_family") or "unknown"
                        ),
                        datasetMode=str(run_context.get("dataset_mode") or "unknown"),
                        createdAt=str(run.get("created_at") or ""),
                        candidateRank=int(candidate.get("rank", 0)),
                        wasBestCandidate=int(candidate.get("rank", 0)) == 1,
                        promoted=bool(run.get("promoted"))
                        and str(run.get("best_model_version")) == version,
                        validationRmse=float(candidate.get("validation_rmse", 0.0)),
                        validationRiskLevelAccuracy=float(
                            candidate.get("validation_risk_level_accuracy", 0.0)
                        ),
                        comparison=candidate.get("comparison") or {},
                    )
                )
        return appearances

    @staticmethod
    def _family_rollups(
        selection_history: list[ModelSelectionAppearanceRead],
    ) -> list[ModelMonitoringFamilyRollupRead]:
        grouped: dict[tuple[str, str], list[ModelSelectionAppearanceRead]] = defaultdict(
            list
        )
        for appearance in selection_history:
            grouped[(appearance.dataset_family, appearance.dataset_mode)].append(
                appearance
            )

        rollups: list[ModelMonitoringFamilyRollupRead] = []
        for (dataset_family, dataset_mode), appearances in grouped.items():
            latest = appearances[0]
            rollups.append(
                ModelMonitoringFamilyRollupRead(
                    datasetFamily=dataset_family,
                    datasetMode=dataset_mode,
                    selectionRunCount=len(appearances),
                    bestCandidateCount=sum(
                        1 for appearance in appearances if appearance.was_best_candidate
                    ),
                    promotedCount=sum(
                        1 for appearance in appearances if appearance.promoted
                    ),
                    latestSelectionAt=latest.created_at,
                    latestValidationRmse=latest.validation_rmse,
                    latestValidationRiskLevelAccuracy=latest.validation_risk_level_accuracy,
                )
            )
        return sorted(
            rollups,
            key=lambda item: (
                -item.best_candidate_count,
                -item.selection_run_count,
                item.dataset_family,
            ),
        )

    def _summary_for_version(self, version: str) -> ModelMonitoringSummaryRead:
        model_id, artifact_type = self._artifact_meta(version)
        selection_history = self._selection_history_for_version(version)
        promotion_history = self.list_promotion_history(model_version=version)
        drift_history = self._drift_history_for_version(version)
        shadow_history = self._shadow_history_for_version(version)
        latest_selection = selection_history[0] if selection_history else None
        latest_promotion = promotion_history[0] if promotion_history else None
        latest_drift = drift_history[0] if drift_history else None
        latest_shadow = shadow_history[0] if shadow_history else None
        dataset_families_seen = sorted(
            {appearance.dataset_family for appearance in selection_history}
        )
        return ModelMonitoringSummaryRead(
            version=version,
            modelId=model_id,
            artifactType=artifact_type,
            active=version == self.model_registry.active_version(),
            selectionRunCount=len(selection_history),
            bestCandidateCount=sum(
                1 for appearance in selection_history if appearance.was_best_candidate
            ),
            labeledBestCandidateCount=sum(
                1
                for appearance in selection_history
                if appearance.was_best_candidate and appearance.dataset_mode == "labels"
            ),
            promotionCount=len(promotion_history),
            latestSelectionAt=latest_selection.created_at if latest_selection else None,
            latestPromotionAt=latest_promotion.promoted_at if latest_promotion else None,
            latestValidationRmse=latest_selection.validation_rmse
            if latest_selection
            else None,
            latestValidationRiskLevelAccuracy=latest_selection.validation_risk_level_accuracy
            if latest_selection
            else None,
            datasetFamiliesSeen=dataset_families_seen,
            latestDriftStatus=latest_drift.severity if latest_drift else None,
            latestDriftAt=latest_drift.created_at if latest_drift else None,
            latestDriftDatasetVersion=latest_drift.dataset_version if latest_drift else None,
            latestDriftValidationRmseDelta=latest_drift.validation_rmse_delta
            if latest_drift
            else None,
            latestDriftValidationRiskLevelAccuracyDelta=latest_drift.validation_risk_level_accuracy_delta
            if latest_drift
            else None,
            latestShadowStatus=(latest_shadow.recommendation or {}).get("status")
            if latest_shadow
            else None,
            latestShadowAt=latest_shadow.created_at if latest_shadow else None,
            latestShadowDatasetVersion=latest_shadow.dataset_version if latest_shadow else None,
            latestShadowBestModelVersion=latest_shadow.best_model_version if latest_shadow else None,
            latestShadowActiveStillBest=latest_shadow.active_still_best
            if latest_shadow
            else None,
        )

    def list_model_monitoring(self) -> list[ModelMonitoringSummaryRead]:
        summaries = [self._summary_for_version(version) for version in self._known_versions()]
        return sorted(
            summaries,
            key=lambda item: (
                not item.active,
                -item.promotion_count,
                -item.best_candidate_count,
                item.version,
            ),
        )

    def get_model_monitoring(self, version: str) -> ModelMonitoringDetailRead:
        known_versions = self._known_versions()
        if version not in known_versions:
            raise ApiError(
                404,
                "model_monitoring_not_found",
                f"Model monitoring view for '{version}' was not found.",
            )
        summary = self._summary_for_version(version)
        selection_history = self._selection_history_for_version(version)
        promotion_history = self.list_promotion_history(model_version=version)
        drift_history = self._drift_history_for_version(version)
        shadow_history = self._shadow_history_for_version(version)
        family_rollups = self._family_rollups(selection_history)
        return ModelMonitoringDetailRead(
            version=summary.version,
            modelId=summary.model_id,
            artifactType=summary.artifact_type,
            active=summary.active,
            selectionRunCount=summary.selection_run_count,
            bestCandidateCount=summary.best_candidate_count,
            labeledBestCandidateCount=summary.labeled_best_candidate_count,
            promotionCount=summary.promotion_count,
            latestSelectionAt=summary.latest_selection_at,
            latestPromotionAt=summary.latest_promotion_at,
            latestValidationRmse=summary.latest_validation_rmse,
            latestValidationRiskLevelAccuracy=summary.latest_validation_risk_level_accuracy,
            datasetFamiliesSeen=summary.dataset_families_seen,
            latestDriftStatus=summary.latest_drift_status,
            latestDriftAt=summary.latest_drift_at,
            latestDriftDatasetVersion=summary.latest_drift_dataset_version,
            latestDriftValidationRmseDelta=summary.latest_drift_validation_rmse_delta,
            latestDriftValidationRiskLevelAccuracyDelta=summary.latest_drift_validation_risk_level_accuracy_delta,
            latestShadowStatus=summary.latest_shadow_status,
            latestShadowAt=summary.latest_shadow_at,
            latestShadowDatasetVersion=summary.latest_shadow_dataset_version,
            latestShadowBestModelVersion=summary.latest_shadow_best_model_version,
            latestShadowActiveStillBest=summary.latest_shadow_active_still_best,
            promotionHistory=promotion_history,
            selectionHistory=selection_history,
            familyRollups=family_rollups,
            driftHistory=drift_history,
            shadowHistory=shadow_history,
        )

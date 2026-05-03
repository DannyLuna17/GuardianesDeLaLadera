from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.dashboard import RunSummaryRead


class SchemaBase(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        from_attributes=True,
        protected_namespaces=(),
    )


class RequestBase(BaseModel):
    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())


class TriggerRunRequest(BaseModel):
    note: str | None = None


class TriggerIngestionRequest(BaseModel):
    sources: list[str] | None = None
    note: str | None = None


class TriggerPipelineRequest(BaseModel):
    sources: list[str] | None = None
    note: str | None = None


class RetrainModelRequest(RequestBase):
    version: str | None = None
    alpha: float = Field(default=0.75, gt=0)
    model_family: Literal[
        "linear_ridge",
        "beta_regression",
        "additive_spline",
        "gradient_boosted_tree",
        "xgboost",
    ] = Field(
        default="linear_ridge", alias="modelFamily"
    )
    knot_count: int | None = Field(default=None, alias="knotCount", ge=0, le=8)
    learning_rate: float | None = Field(
        default=None, alias="learningRate", gt=0, le=1
    )
    estimator_count: int | None = Field(
        default=None, alias="estimatorCount", ge=1, le=256
    )
    max_depth: int | None = Field(default=None, alias="maxDepth", ge=1, le=8)
    min_leaf_size: int | None = Field(
        default=None, alias="minLeafSize", ge=1, le=32
    )
    min_split_gain: float | None = Field(
        default=None, alias="minSplitGain", ge=0
    )
    early_stopping_rounds: int | None = Field(
        default=None, alias="earlyStoppingRounds", ge=0, le=50
    )
    dataset_version: str | None = Field(default=None, alias="datasetVersion")
    review_task_id: int | None = Field(default=None, alias="reviewTaskId", ge=1)


class EvaluateModelRequest(RequestBase):
    version: str | None = None
    model_version: str = Field(alias="modelVersion", min_length=1)
    dataset_version: str = Field(alias="datasetVersion", min_length=1)
    top_error_count: int = Field(default=10, alias="topErrorCount", ge=1, le=25)


class ScanModelDriftRequest(RequestBase):
    version: str | None = None
    model_version: str | None = Field(default=None, alias="modelVersion")
    dataset_version: str | None = Field(default=None, alias="datasetVersion")
    evaluation_version: str | None = Field(default=None, alias="evaluationVersion")
    top_error_count: int = Field(default=10, alias="topErrorCount", ge=1, le=25)
    warning_validation_rmse_increase: float | None = Field(
        default=None, alias="warningValidationRmseIncrease"
    )
    critical_validation_rmse_increase: float | None = Field(
        default=None, alias="criticalValidationRmseIncrease"
    )
    warning_accuracy_drop: float | None = Field(
        default=None, alias="warningAccuracyDrop"
    )
    critical_accuracy_drop: float | None = Field(
        default=None, alias="criticalAccuracyDrop"
    )


class ScanModelShadowRequest(RequestBase):
    version: str | None = None
    dataset_version: str = Field(alias="datasetVersion", min_length=1)
    model_versions: list[str] | None = Field(default=None, alias="modelVersions")
    max_candidates: int = Field(default=4, alias="maxCandidates", ge=1, le=12)
    top_error_count: int = Field(default=5, alias="topErrorCount", ge=1, le=25)


class ScanModelMonitoringRequest(BaseModel):
    dataset_version: str | None = Field(default=None, alias="datasetVersion")
    drift_top_error_count: int = Field(
        default=10, alias="driftTopErrorCount", ge=1, le=25
    )
    shadow_top_error_count: int = Field(
        default=5, alias="shadowTopErrorCount", ge=1, le=25
    )
    shadow_max_candidates: int = Field(
        default=4, alias="shadowMaxCandidates", ge=1, le=12
    )


class OpenModelReviewTasksRequest(BaseModel):
    notification_ids: list[int] = Field(
        alias="notificationIds", min_length=1, max_length=100
    )
    review_type: Literal[
        "promotion_review", "rollback_review", "retraining_review"
    ] = Field(alias="reviewType")
    assigned_reviewer: str | None = Field(default=None, alias="assignedReviewer")
    due_at: datetime | None = Field(default=None, alias="dueAt")
    notes: str | None = None


class UpdateModelReviewTaskRequest(BaseModel):
    status: Literal["open", "in_progress", "resolved", "cancelled"]
    assigned_reviewer: str | None = Field(default=None, alias="assignedReviewer")
    due_at: datetime | None = Field(default=None, alias="dueAt")
    decision: str | None = Field(default=None, min_length=1, max_length=64)
    notes: str | None = None


class TuneModelRequest(RequestBase):
    version: str | None = None
    dataset_version: str = Field(alias="datasetVersion", min_length=1)
    model_family: Literal[
        "linear_ridge",
        "beta_regression",
        "additive_spline",
        "gradient_boosted_tree",
        "xgboost",
    ] = Field(
        default="linear_ridge", alias="modelFamily"
    )
    model_families: list[
        Literal[
            "linear_ridge",
            "beta_regression",
            "additive_spline",
            "gradient_boosted_tree",
            "xgboost",
        ]
    ] | None = Field(default=None, alias="modelFamilies", min_length=1, max_length=4)
    selection_mode: Literal["single_stage", "nested_outer_estimate"] = Field(
        default="single_stage", alias="selectionMode"
    )
    validation_strategy: Literal[
        "dataset_holdout", "spatial_block_kfold", "temporal_holdout_backtest"
    ] = Field(default="dataset_holdout", alias="validationStrategy")
    validation_fold_count: int | None = Field(
        default=None, alias="validationFoldCount", ge=1, le=12
    )
    nested_outer_fold_count: int | None = Field(
        default=None, alias="nestedOuterFoldCount", ge=1, le=12
    )
    alphas: list[float] | None = Field(default=None, min_length=1, max_length=12)
    knot_counts: list[int] | None = Field(
        default=None, alias="knotCounts", min_length=1, max_length=8
    )
    learning_rates: list[float] | None = Field(
        default=None, alias="learningRates", min_length=1, max_length=12
    )
    estimator_counts: list[int] | None = Field(
        default=None, alias="estimatorCounts", min_length=1, max_length=12
    )
    max_depths: list[int] | None = Field(
        default=None, alias="maxDepths", min_length=1, max_length=8
    )
    min_leaf_sizes: list[int] | None = Field(
        default=None, alias="minLeafSizes", min_length=1, max_length=8
    )
    min_split_gains: list[float] | None = Field(
        default=None, alias="minSplitGains", min_length=1, max_length=8
    )
    early_stopping_rounds: int | None = Field(
        default=None, alias="earlyStoppingRounds", ge=0, le=50
    )
    version_prefix: str | None = Field(default=None, alias="versionPrefix")
    promote_best: bool = Field(default=False, alias="promoteBest")
    promotion_reason: str | None = Field(default=None, alias="promotionReason")
    top_error_count: int = Field(default=5, alias="topErrorCount", ge=1, le=25)
    min_validation_rmse_improvement: float | None = Field(
        default=None, alias="minValidationRmseImprovement"
    )
    min_validation_rows: int | None = Field(
        default=None, alias="minValidationRows", ge=1
    )
    require_labels_dataset_for_promotion: bool | None = Field(
        default=None, alias="requireLabelsDatasetForPromotion"
    )
    require_nested_estimation_for_promotion: bool | None = Field(
        default=None, alias="requireNestedEstimationForPromotion"
    )
    min_calibration_gain: float | None = Field(
        default=None, alias="minCalibrationGain"
    )
    min_nested_outer_validation_rmse_improvement: float | None = Field(
        default=None, alias="minNestedOuterValidationRmseImprovement"
    )
    min_nested_outer_selection_rate: float | None = Field(
        default=None, alias="minNestedOuterSelectionRate", ge=0, le=1
    )
    require_nested_temporal_latest_win_for_promotion: bool | None = Field(
        default=None, alias="requireNestedTemporalLatestWinForPromotion"
    )
    min_nested_temporal_outer_bucket_count: int | None = Field(
        default=None, alias="minNestedTemporalOuterBucketCount", ge=0
    )
    min_nested_temporal_latest_validation_rmse_improvement: float | None = Field(
        default=None, alias="minNestedTemporalLatestValidationRmseImprovement"
    )
    nested_temporal_recent_window_size: int | None = Field(
        default=None, alias="nestedTemporalRecentWindowSize", ge=1, le=12
    )
    min_nested_temporal_recent_win_rate: float | None = Field(
        default=None, alias="minNestedTemporalRecentWinRate", ge=0, le=1
    )
    min_nested_temporal_recent_average_validation_rmse_improvement: float | None = (
        Field(
            default=None,
            alias="minNestedTemporalRecentAverageValidationRmseImprovement",
        )
    )
    max_spatial_slice_validation_rmse_regression: float | None = Field(
        default=None, alias="maxSpatialSliceValidationRmseRegression", ge=0
    )
    max_temporal_slice_validation_rmse_regression: float | None = Field(
        default=None, alias="maxTemporalSliceValidationRmseRegression", ge=0
    )
    max_spatial_slice_regression_count: int | None = Field(
        default=None, alias="maxSpatialSliceRegressionCount", ge=0
    )
    max_temporal_slice_regression_count: int | None = Field(
        default=None, alias="maxTemporalSliceRegressionCount", ge=0
    )
    slice_regression_min_rows: int | None = Field(
        default=None, alias="sliceRegressionMinRows", ge=1
    )
    stability_window_runs: int | None = Field(
        default=None, alias="stabilityWindowRuns", ge=1
    )
    required_consistent_wins: int | None = Field(
        default=None, alias="requiredConsistentWins", ge=1
    )
    stability_require_same_dataset_family: bool | None = Field(
        default=None, alias="stabilityRequireSameDatasetFamily"
    )
    stability_require_same_dataset_taxonomy: bool | None = Field(
        default=None, alias="stabilityRequireSameDatasetTaxonomy"
    )
    stability_require_same_evaluation_cohort: bool | None = Field(
        default=None, alias="stabilityRequireSameEvaluationCohort"
    )
    stability_max_time_window_gap_days: int | None = Field(
        default=None, alias="stabilityMaxTimeWindowGapDays", ge=0
    )
    stability_max_cohort_distance: int | None = Field(
        default=None, alias="stabilityMaxCohortDistance", ge=0
    )


class RunModernLabelsBenchmarkRequest(RequestBase):
    dataset_version: str | None = Field(default=None, alias="datasetVersion")
    auto_export_dataset: bool = Field(default=True, alias="autoExportDataset")
    dataset_export_version: str | None = Field(
        default=None, alias="datasetExportVersion"
    )
    label_sources: list[str] | None = Field(default=None, alias="labelSources")
    max_labels: int = Field(default=250, alias="maxLabels", ge=1, le=500)
    observed_after: datetime | None = Field(default=None, alias="observedAfter")
    observed_before: datetime | None = Field(default=None, alias="observedBefore")
    validation_strategy: Literal[
        "dataset_holdout", "spatial_block_kfold", "temporal_holdout_backtest"
    ] | None = Field(default=None, alias="validationStrategy")
    validation_fold_count: int | None = Field(
        default=None, alias="validationFoldCount", ge=1, le=12
    )
    nested_outer_fold_count: int | None = Field(
        default=None, alias="nestedOuterFoldCount", ge=1, le=12
    )
    version: str | None = None
    version_prefix: str | None = Field(default=None, alias="versionPrefix")
    promote_best: bool = Field(default=False, alias="promoteBest")
    promotion_reason: str | None = Field(default=None, alias="promotionReason")


class RunModernLabelsBenchmarkReviewRequest(RunModernLabelsBenchmarkRequest):
    open_review_task: bool = Field(default=True, alias="openReviewTask")
    assigned_reviewer: str | None = Field(default=None, alias="assignedReviewer")
    due_at: datetime | None = Field(default=None, alias="dueAt")
    notes: str | None = None


class PromoteModelRequest(RequestBase):
    model_version: str = Field(alias="modelVersion", min_length=1)
    reason: str | None = None
    review_task_id: int | None = Field(default=None, alias="reviewTaskId", ge=1)


class RollbackModelRequest(BaseModel):
    reason: str | None = None
    review_task_id: int | None = Field(default=None, alias="reviewTaskId", ge=1)


class UserAccountAdminRead(SchemaBase):
    id: int
    username: str
    role: str
    is_active: bool = Field(alias="isActive")
    created_at: datetime = Field(alias="createdAt")


class CreateUserAccountRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=8, max_length=256)
    role: str = Field(default="viewer", min_length=3, max_length=32)
    is_active: bool = Field(default=True, alias="isActive")


class UpdateUserAccountRequest(BaseModel):
    role: str | None = Field(default=None, min_length=3, max_length=32)
    is_active: bool | None = Field(default=None, alias="isActive")


class ResetUserPasswordRequest(BaseModel):
    new_password: str = Field(alias="newPassword", min_length=8, max_length=256)


class ExportTrainingDatasetRequest(BaseModel):
    version: str | None = None
    source_mode: Literal["seed", "operational", "labels"] = Field(
        default="seed", alias="sourceMode"
    )
    run_ids: list[int] | None = Field(default=None, alias="runIds")
    max_runs: int = Field(default=5, alias="maxRuns", ge=1, le=50)
    label_ids: list[int] | None = Field(default=None, alias="labelIds")
    label_sources: list[str] | None = Field(default=None, alias="labelSources")
    max_labels: int = Field(default=100, alias="maxLabels", ge=1, le=500)
    observed_after: datetime | None = Field(default=None, alias="observedAfter")
    observed_before: datetime | None = Field(default=None, alias="observedBefore")


class OutcomeLabelWrite(BaseModel):
    zone_id: str = Field(alias="zoneId")
    observed_at: datetime = Field(alias="observedAt")
    target_score: float = Field(alias="targetScore", ge=0, le=1)
    source: str
    feature_run_id: int | None = Field(default=None, alias="featureRunId")
    status: Literal["draft", "confirmed", "rejected", "needs_revision"] = "confirmed"
    notes: str | None = None
    evidence: dict = Field(default_factory=dict)


class UpsertOutcomeLabelsRequest(BaseModel):
    labels: list[OutcomeLabelWrite] = Field(min_length=1, max_length=100)


class ImportHistoricalLabelsRequest(BaseModel):
    municipality: str | None = None
    zone_id: str | None = Field(default=None, alias="zoneId")
    event_ids: list[str] | None = Field(default=None, alias="eventIds")
    event_source: str | None = Field(default=None, alias="eventSource")
    status: Literal["draft", "confirmed"] = "draft"
    max_events: int = Field(default=100, alias="maxEvents", ge=1, le=500)
    severity_score_overrides: dict[str, float] | None = Field(
        default=None, alias="severityScoreOverrides"
    )


class ImportUngrdLabelsRequest(BaseModel):
    municipality: str | None = None
    zone_id: str | None = Field(default=None, alias="zoneId")
    record_ids: list[str] | None = Field(default=None, alias="recordIds")
    status: Literal["draft", "confirmed"] = "draft"
    max_records: int = Field(default=50, alias="maxRecords", ge=1, le=500)
    max_zones_per_record: int = Field(default=2, alias="maxZonesPerRecord", ge=1, le=10)
    summary_score_overrides: dict[str, float] | None = Field(
        default=None, alias="summaryScoreOverrides"
    )


class FieldValidationObservationWrite(BaseModel):
    observation_id: str = Field(alias="observationId")
    zone_id: str = Field(alias="zoneId")
    observed_at: datetime = Field(alias="observedAt")
    severity: str | None = None
    target_score: float | None = Field(default=None, alias="targetScore", ge=0, le=1)
    feature_run_id: int | None = Field(default=None, alias="featureRunId")
    observer: str | None = None
    site_visit_id: str | None = Field(default=None, alias="siteVisitId")
    team_id: str | None = Field(default=None, alias="teamId")
    media_refs: list[str] | None = Field(default=None, alias="mediaRefs")
    attachment_refs: list[str] | None = Field(default=None, alias="attachmentRefs")
    gps_accuracy_meters: float | None = Field(
        default=None, alias="gpsAccuracyMeters", ge=0
    )
    location_notes: str | None = Field(default=None, alias="locationNotes")
    status: Literal["draft", "confirmed"] = "draft"
    notes: str | None = None
    evidence: dict = Field(default_factory=dict)


class ImportFieldValidationLabelsRequest(BaseModel):
    observations: list[FieldValidationObservationWrite] = Field(
        min_length=1, max_length=200
    )
    severity_score_overrides: dict[str, float] | None = Field(
        default=None, alias="severityScoreOverrides"
    )


class ReviewOutcomeLabelsRequest(BaseModel):
    label_ids: list[int] = Field(alias="labelIds", min_length=1, max_length=200)
    decision: Literal["confirmed", "rejected", "needs_revision"]
    review_notes: str | None = Field(default=None, alias="reviewNotes")


class AssignOutcomeLabelsRequest(BaseModel):
    label_ids: list[int] = Field(alias="labelIds", min_length=1, max_length=200)
    reviewer_username: str = Field(
        alias="reviewerUsername", min_length=1, max_length=64
    )
    review_due_at: datetime | None = Field(default=None, alias="reviewDueAt")
    assignment_notes: str | None = Field(default=None, alias="assignmentNotes")


class UpdateTrainingEligibilityRequest(BaseModel):
    label_ids: list[int] = Field(alias="labelIds", min_length=1, max_length=200)
    training_eligibility_status: Literal["eligible", "hold", "ineligible"] = Field(
        alias="trainingEligibilityStatus"
    )
    notes: str | None = None


class RequestTrainingReleaseRequest(BaseModel):
    label_ids: list[int] = Field(alias="labelIds", min_length=1, max_length=200)
    release_criteria: list[str] = Field(
        alias="releaseCriteria", min_length=1, max_length=20
    )
    notes: str | None = None


class ReviewTrainingReleaseRequest(BaseModel):
    label_ids: list[int] = Field(alias="labelIds", min_length=1, max_length=200)
    decision: Literal["approved", "rejected"]
    notes: str | None = None


class AssignTrainingReleaseRequest(BaseModel):
    label_ids: list[int] = Field(alias="labelIds", min_length=1, max_length=200)
    reviewer_username: str = Field(
        alias="reviewerUsername", min_length=1, max_length=64
    )
    review_due_at: datetime | None = Field(default=None, alias="reviewDueAt")
    assignment_notes: str | None = Field(default=None, alias="assignmentNotes")


class EscalateTrainingReleaseRequest(BaseModel):
    label_ids: list[int] = Field(alias="labelIds", min_length=1, max_length=200)
    escalation_reason: str = Field(
        alias="escalationReason", min_length=1, max_length=500
    )
    escalation_level: int | None = Field(
        default=None, alias="escalationLevel", ge=1, le=5
    )


class ReassignTrainingReleaseRequest(BaseModel):
    label_ids: list[int] = Field(alias="labelIds", min_length=1, max_length=200)
    reviewer_username: str = Field(
        alias="reviewerUsername", min_length=1, max_length=64
    )
    reassignment_reason: str = Field(
        alias="reassignmentReason", min_length=1, max_length=500
    )
    review_due_at: datetime | None = Field(default=None, alias="reviewDueAt")


class TriggerTrainingReleaseSlaScanRequest(BaseModel):
    max_labels: int = Field(default=100, alias="maxLabels", ge=1, le=500)
    note: str | None = None


class TriggerTrainingReleaseReassignmentScanRequest(BaseModel):
    max_labels: int = Field(default=100, alias="maxLabels", ge=1, le=500)
    note: str | None = None


class TriggerNotificationAckScanRequest(BaseModel):
    max_notifications: int = Field(default=100, alias="maxNotifications", ge=1, le=500)
    note: str | None = None


class TriggerNotificationDeliveryRetryScanRequest(BaseModel):
    max_notifications: int = Field(default=100, alias="maxNotifications", ge=1, le=500)
    note: str | None = None


class TriggerNotificationDeliveryFailureScanRequest(BaseModel):
    max_notifications: int = Field(default=100, alias="maxNotifications", ge=1, le=500)
    note: str | None = None


class RetryNotificationDeliveryRequest(BaseModel):
    notification_ids: list[int] = Field(
        alias="notificationIds", min_length=1, max_length=200
    )
    channels: list[str] | None = None
    note: str | None = None


class AcknowledgeNotificationsRequest(BaseModel):
    notification_ids: list[int] = Field(
        alias="notificationIds", min_length=1, max_length=200
    )


class RefreshExplanationRequest(BaseModel):
    run_id: int | None = Field(default=None, alias="runId")


class JobExecutionRead(SchemaBase):
    id: int
    job_type: str = Field(alias="jobType")
    status: str
    started_at: datetime = Field(alias="startedAt")
    completed_at: datetime | None = Field(alias="completedAt")
    details: dict


class TriggerRunResponse(SchemaBase):
    job: JobExecutionRead
    run: RunSummaryRead


class RefreshExplanationResponse(SchemaBase):
    job: JobExecutionRead
    refreshed_count: int = Field(alias="refreshedCount")
    run_id: int = Field(alias="runId")


class IngestionSourceRead(SchemaBase):
    source_id: str = Field(alias="sourceId")
    processed_records: int = Field(alias="processedRecords")
    adapter_key: str = Field(alias="adapterKey")
    transport: str
    status: str
    message: str
    details: dict


class SourceSyncEventRead(SchemaBase):
    id: int
    source_id: str = Field(alias="sourceId")
    source_label: str = Field(alias="sourceLabel")
    origin: str
    adapter_key: str = Field(alias="adapterKey")
    transport: str
    status: str
    processed_records: int = Field(alias="processedRecords")
    started_at: datetime = Field(alias="startedAt")
    completed_at: datetime | None = Field(alias="completedAt")
    message: str
    details: dict


class TriggerIngestionResponse(SchemaBase):
    job: JobExecutionRead
    synced_sources: list[IngestionSourceRead] = Field(alias="syncedSources")


class TriggerPipelineResponse(SchemaBase):
    job: JobExecutionRead
    ingestion: TriggerIngestionResponse
    run: TriggerRunResponse
    explanations: RefreshExplanationResponse


class SchedulerJobRead(SchemaBase):
    id: str
    trigger: str
    next_run_time: datetime | None = Field(alias="nextRunTime")


class SchedulerStatusRead(SchemaBase):
    enabled: bool
    running: bool
    timezone: str
    execution_mode: str = Field(alias="executionMode")
    scheduler_sources: list[str] = Field(alias="schedulerSources")
    ingestion_interval_minutes: int = Field(alias="ingestionIntervalMinutes")
    prediction_interval_minutes: int = Field(alias="predictionIntervalMinutes")
    operational_pipeline_interval_minutes: int = Field(
        alias="operationalPipelineIntervalMinutes"
    )
    training_release_sla_monitor_enabled: bool = Field(
        alias="trainingReleaseSlaMonitorEnabled"
    )
    training_release_sla_interval_minutes: int = Field(
        alias="trainingReleaseSlaIntervalMinutes"
    )
    training_release_reassignment_monitor_enabled: bool = Field(
        alias="trainingReleaseReassignmentMonitorEnabled"
    )
    training_release_reassignment_interval_minutes: int = Field(
        alias="trainingReleaseReassignmentIntervalMinutes"
    )
    training_release_auto_reassign_reviewer: str = Field(
        alias="trainingReleaseAutoReassignReviewer"
    )
    notification_ack_monitor_enabled: bool = Field(
        alias="notificationAckMonitorEnabled"
    )
    notification_ack_monitor_interval_minutes: int = Field(
        alias="notificationAckMonitorIntervalMinutes"
    )
    notification_delivery_retry_monitor_enabled: bool = Field(
        alias="notificationDeliveryRetryMonitorEnabled"
    )
    notification_delivery_retry_interval_minutes: int = Field(
        alias="notificationDeliveryRetryIntervalMinutes"
    )
    notification_delivery_failure_monitor_enabled: bool = Field(
        alias="notificationDeliveryFailureMonitorEnabled"
    )
    notification_delivery_failure_interval_minutes: int = Field(
        alias="notificationDeliveryFailureIntervalMinutes"
    )
    model_monitoring_cycle_enabled: bool = Field(alias="modelMonitoringCycleEnabled")
    model_monitoring_interval_minutes: int = Field(
        alias="modelMonitoringIntervalMinutes"
    )
    model_monitoring_drift_top_error_count: int = Field(
        alias="modelMonitoringDriftTopErrorCount"
    )
    model_monitoring_shadow_top_error_count: int = Field(
        alias="modelMonitoringShadowTopErrorCount"
    )
    model_monitoring_shadow_max_candidates: int = Field(
        alias="modelMonitoringShadowMaxCandidates"
    )
    model_monitoring_alerts_enabled: bool = Field(alias="modelMonitoringAlertsEnabled")
    notification_model_monitoring_usernames: list[str] = Field(
        alias="notificationModelMonitoringUsernames"
    )
    jobs: list[SchedulerJobRead]
    last_job: JobExecutionRead | None = Field(default=None, alias="lastJob")


class RetrainModelResponse(SchemaBase):
    job: JobExecutionRead
    model_version: str = Field(alias="modelVersion")
    model_family: str = Field(alias="modelFamily")
    artifact_path: str = Field(alias="artifactPath")
    rows: int
    alpha: float | None = None
    hyperparameters: dict = Field(default_factory=dict)
    dataset_version: str = Field(alias="datasetVersion")
    feature_order: list[str] = Field(alias="featureOrder")
    metrics: dict[str, float]
    active_model_version: str = Field(alias="activeModelVersion")
    overwrote_active_version: bool = Field(alias="overwroteActiveVersion")


class EvaluateModelResponse(SchemaBase):
    job: JobExecutionRead
    evaluation_version: str = Field(alias="evaluationVersion")
    artifact_path: str = Field(alias="artifactPath")
    model_version: str = Field(alias="modelVersion")
    dataset_version: str = Field(alias="datasetVersion")
    rows: int
    metrics: dict
    validation_metrics: dict = Field(alias="validationMetrics")
    diagnostics: dict
    top_errors: list[dict] = Field(alias="topErrors")


class ModelDriftSummaryRead(SchemaBase):
    version: str
    drift_id: str = Field(alias="driftId")
    artifact_type: str = Field(alias="artifactType")
    created_at: str = Field(alias="createdAt")
    model_version: str = Field(alias="modelVersion")
    dataset_version: str = Field(alias="datasetVersion")
    evaluation_version: str = Field(alias="evaluationVersion")
    severity: str
    drift_detected: bool = Field(alias="driftDetected")
    baseline_source: str | None = Field(default=None, alias="baselineSource")
    baseline_reference_version: str | None = Field(
        default=None, alias="baselineReferenceVersion"
    )
    baseline_dataset_version: str | None = Field(
        default=None, alias="baselineDatasetVersion"
    )
    validation_rmse: float | None = Field(default=None, alias="validationRmse")
    baseline_validation_rmse: float | None = Field(
        default=None, alias="baselineValidationRmse"
    )
    validation_rmse_delta: float | None = Field(
        default=None, alias="validationRmseDelta"
    )
    validation_risk_level_accuracy: float | None = Field(
        default=None, alias="validationRiskLevelAccuracy"
    )
    baseline_validation_risk_level_accuracy: float | None = Field(
        default=None, alias="baselineValidationRiskLevelAccuracy"
    )
    validation_risk_level_accuracy_delta: float | None = Field(
        default=None, alias="validationRiskLevelAccuracyDelta"
    )
    validation_rows: int = Field(alias="validationRows")
    dataset_family: str = Field(alias="datasetFamily")
    taxonomy_group: str | None = Field(default=None, alias="taxonomyGroup")
    evaluation_cohort_label: str | None = Field(
        default=None, alias="evaluationCohortLabel"
    )


class ModelDriftDetailRead(SchemaBase):
    version: str
    drift_id: str = Field(alias="driftId")
    artifact_type: str = Field(alias="artifactType")
    artifact_path: str = Field(alias="artifactPath")
    created_at: str = Field(alias="createdAt")
    model_version: str = Field(alias="modelVersion")
    dataset_version: str = Field(alias="datasetVersion")
    evaluation_version: str = Field(alias="evaluationVersion")
    baseline: dict
    current: dict
    drift_summary: dict = Field(alias="driftSummary")
    diagnostics: dict
    top_errors: list[dict] = Field(alias="topErrors")


class ModelShadowCandidateRead(SchemaBase):
    rank: int
    role: str
    model_version: str = Field(alias="modelVersion")
    model_family: str = Field(alias="modelFamily")
    artifact_path: str = Field(alias="artifactPath")
    evaluation_version: str = Field(alias="evaluationVersion")
    overall_rmse: float = Field(alias="overallRmse")
    validation_rmse: float = Field(alias="validationRmse")
    overall_risk_level_accuracy: float = Field(alias="overallRiskLevelAccuracy")
    validation_risk_level_accuracy: float = Field(alias="validationRiskLevelAccuracy")
    overall_brier_score: float | None = Field(default=None, alias="overallBrierScore")
    validation_brier_score: float | None = Field(
        default=None, alias="validationBrierScore"
    )
    overall_auroc: float | None = Field(default=None, alias="overallAuroc")
    validation_auroc: float | None = Field(default=None, alias="validationAuroc")
    overall_auprc: float | None = Field(default=None, alias="overallAuprc")
    validation_auprc: float | None = Field(default=None, alias="validationAuprc")
    validation_recall: float | None = Field(default=None, alias="validationRecall")
    validation_specificity: float | None = Field(
        default=None, alias="validationSpecificity"
    )
    validation_mcc: float | None = Field(default=None, alias="validationMcc")
    validation_ece: float | None = Field(default=None, alias="validationEce")
    validation_rows: int = Field(alias="validationRows")
    hyperparameters: dict = Field(default_factory=dict)
    comparison: dict = Field(default_factory=dict)
    top_errors: list[dict] = Field(alias="topErrors")


class ModelShadowRunSummaryRead(SchemaBase):
    version: str
    shadow_id: str = Field(alias="shadowId")
    artifact_type: str = Field(alias="artifactType")
    dataset_version: str = Field(alias="datasetVersion")
    created_at: str = Field(alias="createdAt")
    active_model_version: str = Field(alias="activeModelVersion")
    best_model_version: str = Field(alias="bestModelVersion")
    active_still_best: bool = Field(alias="activeStillBest")
    candidate_count: int = Field(alias="candidateCount")
    recommendation: dict


class ModelShadowRunDetailRead(SchemaBase):
    version: str
    shadow_id: str = Field(alias="shadowId")
    artifact_type: str = Field(alias="artifactType")
    artifact_path: str = Field(alias="artifactPath")
    dataset_version: str = Field(alias="datasetVersion")
    created_at: str = Field(alias="createdAt")
    dataset_context: dict = Field(alias="datasetContext")
    comparison_policy: dict = Field(alias="comparisonPolicy")
    active_model_version: str = Field(alias="activeModelVersion")
    best_model_version: str = Field(alias="bestModelVersion")
    active_still_best: bool = Field(alias="activeStillBest")
    active_candidate_rank: int = Field(alias="activeCandidateRank")
    candidate_count: int = Field(alias="candidateCount")
    candidate_selection: dict = Field(alias="candidateSelection")
    recommendation: dict
    candidates: list[ModelShadowCandidateRead]


class PromoteModelResponse(SchemaBase):
    model_version: str = Field(alias="modelVersion")
    active_model_version: str = Field(alias="activeModelVersion")
    previous_active_model_version: str | None = Field(
        default=None, alias="previousActiveModelVersion"
    )
    manifest_path: str = Field(alias="manifestPath")
    promoted_at: str = Field(alias="promotedAt")
    promoted_by: str | None = Field(default=None, alias="promotedBy")
    reason: str | None = None
    source: str


class RollbackModelResponse(SchemaBase):
    model_version: str = Field(alias="modelVersion")
    active_model_version: str = Field(alias="activeModelVersion")
    rolled_back_from_model_version: str = Field(alias="rolledBackFromModelVersion")
    previous_active_model_version: str | None = Field(
        default=None, alias="previousActiveModelVersion"
    )
    manifest_path: str = Field(alias="manifestPath")
    promoted_at: str = Field(alias="promotedAt")
    promoted_by: str | None = Field(default=None, alias="promotedBy")
    reason: str | None = None
    source: str


class ModelPromotionHistoryEntryRead(SchemaBase):
    model_version: str = Field(alias="modelVersion")
    previous_active_model_version: str | None = Field(
        default=None, alias="previousActiveModelVersion"
    )
    rolled_back_from_model_version: str | None = Field(
        default=None, alias="rolledBackFromModelVersion"
    )
    promoted_at: str = Field(alias="promotedAt")
    promoted_by: str | None = Field(default=None, alias="promotedBy")
    reason: str | None = None
    source: str
    rollback: bool
    current_active: bool = Field(alias="currentActive")


class ModelSelectionCandidateRead(SchemaBase):
    rank: int
    model_version: str = Field(alias="modelVersion")
    model_family: str = Field(alias="modelFamily")
    alpha: float | None = None
    artifact_path: str = Field(alias="artifactPath")
    overall_rmse: float = Field(alias="overallRmse")
    validation_rmse: float = Field(alias="validationRmse")
    overall_risk_level_accuracy: float = Field(alias="overallRiskLevelAccuracy")
    validation_risk_level_accuracy: float = Field(alias="validationRiskLevelAccuracy")
    overall_brier_score: float | None = Field(default=None, alias="overallBrierScore")
    validation_brier_score: float | None = Field(
        default=None, alias="validationBrierScore"
    )
    overall_auroc: float | None = Field(default=None, alias="overallAuroc")
    validation_auroc: float | None = Field(default=None, alias="validationAuroc")
    overall_auprc: float | None = Field(default=None, alias="overallAuprc")
    validation_auprc: float | None = Field(default=None, alias="validationAuprc")
    validation_recall: float | None = Field(default=None, alias="validationRecall")
    validation_specificity: float | None = Field(
        default=None, alias="validationSpecificity"
    )
    validation_mcc: float | None = Field(default=None, alias="validationMcc")
    validation_ece: float | None = Field(default=None, alias="validationEce")
    validation_rows: int = Field(alias="validationRows")
    hyperparameters: dict = Field(default_factory=dict)
    validation_summary: dict = Field(default_factory=dict, alias="validationSummary")
    comparison: dict = Field(default_factory=dict)
    top_errors: list[dict] = Field(alias="topErrors")


class ModelSelectionAppearanceRead(SchemaBase):
    selection_version: str = Field(alias="selectionVersion")
    dataset_version: str = Field(alias="datasetVersion")
    dataset_family: str = Field(alias="datasetFamily")
    dataset_mode: str = Field(alias="datasetMode")
    created_at: str = Field(alias="createdAt")
    candidate_rank: int = Field(alias="candidateRank")
    was_best_candidate: bool = Field(alias="wasBestCandidate")
    promoted: bool
    validation_rmse: float = Field(alias="validationRmse")
    validation_risk_level_accuracy: float = Field(alias="validationRiskLevelAccuracy")
    comparison: dict = Field(default_factory=dict)


class ModelMonitoringFamilyRollupRead(SchemaBase):
    dataset_family: str = Field(alias="datasetFamily")
    dataset_mode: str = Field(alias="datasetMode")
    selection_run_count: int = Field(alias="selectionRunCount")
    best_candidate_count: int = Field(alias="bestCandidateCount")
    promoted_count: int = Field(alias="promotedCount")
    latest_selection_at: str | None = Field(default=None, alias="latestSelectionAt")
    latest_validation_rmse: float | None = Field(
        default=None, alias="latestValidationRmse"
    )
    latest_validation_risk_level_accuracy: float | None = Field(
        default=None, alias="latestValidationRiskLevelAccuracy"
    )


class ModelMonitoringSummaryRead(SchemaBase):
    version: str
    model_id: str = Field(alias="modelId")
    artifact_type: str = Field(alias="artifactType")
    active: bool
    selection_run_count: int = Field(alias="selectionRunCount")
    best_candidate_count: int = Field(alias="bestCandidateCount")
    labeled_best_candidate_count: int = Field(alias="labeledBestCandidateCount")
    promotion_count: int = Field(alias="promotionCount")
    latest_selection_at: str | None = Field(default=None, alias="latestSelectionAt")
    latest_promotion_at: str | None = Field(default=None, alias="latestPromotionAt")
    latest_validation_rmse: float | None = Field(
        default=None, alias="latestValidationRmse"
    )
    latest_validation_risk_level_accuracy: float | None = Field(
        default=None, alias="latestValidationRiskLevelAccuracy"
    )
    dataset_families_seen: list[str] = Field(alias="datasetFamiliesSeen")
    latest_drift_status: str | None = Field(default=None, alias="latestDriftStatus")
    latest_drift_at: str | None = Field(default=None, alias="latestDriftAt")
    latest_drift_dataset_version: str | None = Field(
        default=None, alias="latestDriftDatasetVersion"
    )
    latest_drift_validation_rmse_delta: float | None = Field(
        default=None, alias="latestDriftValidationRmseDelta"
    )
    latest_drift_validation_risk_level_accuracy_delta: float | None = Field(
        default=None, alias="latestDriftValidationRiskLevelAccuracyDelta"
    )
    latest_shadow_status: str | None = Field(default=None, alias="latestShadowStatus")
    latest_shadow_at: str | None = Field(default=None, alias="latestShadowAt")
    latest_shadow_dataset_version: str | None = Field(
        default=None, alias="latestShadowDatasetVersion"
    )
    latest_shadow_best_model_version: str | None = Field(
        default=None, alias="latestShadowBestModelVersion"
    )
    latest_shadow_active_still_best: bool | None = Field(
        default=None, alias="latestShadowActiveStillBest"
    )


class ModelMonitoringDetailRead(SchemaBase):
    version: str
    model_id: str = Field(alias="modelId")
    artifact_type: str = Field(alias="artifactType")
    active: bool
    selection_run_count: int = Field(alias="selectionRunCount")
    best_candidate_count: int = Field(alias="bestCandidateCount")
    labeled_best_candidate_count: int = Field(alias="labeledBestCandidateCount")
    promotion_count: int = Field(alias="promotionCount")
    latest_selection_at: str | None = Field(default=None, alias="latestSelectionAt")
    latest_promotion_at: str | None = Field(default=None, alias="latestPromotionAt")
    latest_validation_rmse: float | None = Field(
        default=None, alias="latestValidationRmse"
    )
    latest_validation_risk_level_accuracy: float | None = Field(
        default=None, alias="latestValidationRiskLevelAccuracy"
    )
    dataset_families_seen: list[str] = Field(alias="datasetFamiliesSeen")
    latest_drift_status: str | None = Field(default=None, alias="latestDriftStatus")
    latest_drift_at: str | None = Field(default=None, alias="latestDriftAt")
    latest_drift_dataset_version: str | None = Field(
        default=None, alias="latestDriftDatasetVersion"
    )
    latest_drift_validation_rmse_delta: float | None = Field(
        default=None, alias="latestDriftValidationRmseDelta"
    )
    latest_drift_validation_risk_level_accuracy_delta: float | None = Field(
        default=None, alias="latestDriftValidationRiskLevelAccuracyDelta"
    )
    latest_shadow_status: str | None = Field(default=None, alias="latestShadowStatus")
    latest_shadow_at: str | None = Field(default=None, alias="latestShadowAt")
    latest_shadow_dataset_version: str | None = Field(
        default=None, alias="latestShadowDatasetVersion"
    )
    latest_shadow_best_model_version: str | None = Field(
        default=None, alias="latestShadowBestModelVersion"
    )
    latest_shadow_active_still_best: bool | None = Field(
        default=None, alias="latestShadowActiveStillBest"
    )
    promotion_history: list[ModelPromotionHistoryEntryRead] = Field(
        alias="promotionHistory"
    )
    selection_history: list[ModelSelectionAppearanceRead] = Field(
        alias="selectionHistory"
    )
    family_rollups: list[ModelMonitoringFamilyRollupRead] = Field(
        alias="familyRollups"
    )
    drift_history: list[ModelDriftSummaryRead] = Field(alias="driftHistory")
    shadow_history: list[ModelShadowRunSummaryRead] = Field(alias="shadowHistory")


class ModelReviewTaskRead(SchemaBase):
    id: int
    review_type: str = Field(alias="reviewType")
    status: str
    source_notification_id: int = Field(alias="sourceNotificationId")
    source_event_type: str = Field(alias="sourceEventType")
    source_alert_severity: str = Field(alias="sourceAlertSeverity")
    source_alert_status: str = Field(alias="sourceAlertStatus")
    active_model_version: str = Field(alias="activeModelVersion")
    candidate_model_version: str | None = Field(
        default=None, alias="candidateModelVersion"
    )
    dataset_version: str | None = Field(default=None, alias="datasetVersion")
    title: str
    summary: str
    recommended_action: str | None = Field(
        default=None, alias="recommendedAction"
    )
    assigned_reviewer: str | None = Field(default=None, alias="assignedReviewer")
    due_at: datetime | None = Field(default=None, alias="dueAt")
    decision: str | None = None
    resolution_notes: str | None = Field(default=None, alias="resolutionNotes")
    details: dict
    created_at: datetime = Field(alias="createdAt")
    created_by: str = Field(alias="createdBy")
    updated_at: datetime = Field(alias="updatedAt")
    updated_by: str | None = Field(default=None, alias="updatedBy")
    resolved_at: datetime | None = Field(default=None, alias="resolvedAt")
    resolved_by: str | None = Field(default=None, alias="resolvedBy")


class ScanModelDriftResponse(SchemaBase):
    job: JobExecutionRead
    drift_version: str = Field(alias="driftVersion")
    artifact_path: str = Field(alias="artifactPath")
    evaluation_version: str = Field(alias="evaluationVersion")
    evaluation_artifact_path: str = Field(alias="evaluationArtifactPath")
    model_version: str = Field(alias="modelVersion")
    dataset_version: str = Field(alias="datasetVersion")
    severity: str
    drift_detected: bool = Field(alias="driftDetected")
    drift_summary: dict = Field(alias="driftSummary")
    baseline: dict
    current: dict


class ScanModelShadowResponse(SchemaBase):
    job: JobExecutionRead
    shadow_version: str = Field(alias="shadowVersion")
    artifact_path: str = Field(alias="artifactPath")
    dataset_version: str = Field(alias="datasetVersion")
    active_model_version: str = Field(alias="activeModelVersion")
    best_model_version: str = Field(alias="bestModelVersion")
    active_still_best: bool = Field(alias="activeStillBest")
    candidate_count: int = Field(alias="candidateCount")
    recommendation: dict
    candidates: list[ModelShadowCandidateRead]


class ScanModelMonitoringResponse(SchemaBase):
    job: JobExecutionRead
    active_model_version: str = Field(alias="activeModelVersion")
    dataset_version: str | None = Field(default=None, alias="datasetVersion")
    skipped: bool = False
    reason: str | None = None
    created_alert_count: int = Field(default=0, alias="createdAlertCount")
    updated_alert_count: int = Field(default=0, alias="updatedAlertCount")
    resolved_alert_count: int = Field(default=0, alias="resolvedAlertCount")
    alerts: list["NotificationEventRead"] = Field(default_factory=list)
    drift: ScanModelDriftResponse | None = None
    shadow: ScanModelShadowResponse | None = None


class OpenModelReviewTasksResponse(SchemaBase):
    created_count: int = Field(alias="createdCount")
    skipped_count: int = Field(alias="skippedCount")
    tasks: list[ModelReviewTaskRead]


class TuneModelResponse(SchemaBase):
    job: JobExecutionRead
    selection_version: str = Field(alias="selectionVersion")
    artifact_path: str = Field(alias="artifactPath")
    dataset_version: str = Field(alias="datasetVersion")
    candidate_count: int = Field(alias="candidateCount")
    best_model_version: str = Field(alias="bestModelVersion")
    promoted: bool
    active_model_version: str = Field(alias="activeModelVersion")
    promotion_decision: dict = Field(alias="promotionDecision")
    best_candidate_comparison: dict = Field(alias="bestCandidateComparison")
    family_rollups: list[dict] = Field(default_factory=list, alias="familyRollups")
    nested_estimation: dict = Field(default_factory=dict, alias="nestedEstimation")
    candidates: list[ModelSelectionCandidateRead]


class TrainingDatasetSummaryRead(SchemaBase):
    version: str
    dataset_id: str = Field(alias="datasetId")
    artifact_type: str = Field(alias="artifactType")
    description: str | None = None
    rows: int
    zones: int
    feature_count: int = Field(alias="featureCount")
    train_rows: int = Field(alias="trainRows")
    validation_rows: int = Field(alias="validationRows")
    provenance_source: str | None = Field(default=None, alias="provenanceSource")
    exported_at: str | None = Field(default=None, alias="exportedAt")


class TrainingDatasetRowPreviewRead(SchemaBase):
    zone_id: str = Field(alias="zoneId")
    phase: str
    split: str
    target_score: float = Field(alias="targetScore")
    feature_vector: dict[str, float] = Field(alias="featureVector")
    context: dict = Field(default_factory=dict)


class TrainingDatasetDetailRead(SchemaBase):
    version: str
    dataset_id: str = Field(alias="datasetId")
    artifact_type: str = Field(alias="artifactType")
    description: str | None = None
    artifact_path: str = Field(alias="artifactPath")
    label_name: str = Field(alias="labelName")
    feature_order: list[str] = Field(alias="featureOrder")
    summary: dict
    provenance: dict
    sample_rows: list[TrainingDatasetRowPreviewRead] = Field(alias="sampleRows")


class ExportTrainingDatasetResponse(SchemaBase):
    job: JobExecutionRead
    dataset_version: str = Field(alias="datasetVersion")
    dataset_path: str = Field(alias="datasetPath")
    source_mode: Literal["seed", "operational", "labels"] = Field(alias="sourceMode")
    run_count: int | None = Field(default=None, alias="runCount")
    label_count: int | None = Field(default=None, alias="labelCount")
    rows: int
    feature_order: list[str] = Field(alias="featureOrder")
    split_counts: dict[str, int] = Field(alias="splitCounts")


class RunModernLabelsBenchmarkResponse(SchemaBase):
    preset: str
    dataset_resolution: dict = Field(alias="datasetResolution")
    dataset_export: ExportTrainingDatasetResponse | None = Field(
        default=None, alias="datasetExport"
    )
    resolved_policy: dict = Field(alias="resolvedPolicy")
    selection: TuneModelResponse


class OutcomeLabelRead(SchemaBase):
    id: int
    zone_id: str = Field(alias="zoneId")
    zone_name: str = Field(alias="zoneName")
    municipality: str
    observed_at: datetime = Field(alias="observedAt")
    target_score: float = Field(alias="targetScore")
    target_risk_level: str = Field(alias="targetRiskLevel")
    source: str
    status: str
    feature_run_id: int | None = Field(default=None, alias="featureRunId")
    feature_run_completed_at: datetime | None = Field(
        default=None, alias="featureRunCompletedAt"
    )
    notes: str | None = None
    evidence: dict
    assigned_reviewer: str | None = Field(default=None, alias="assignedReviewer")
    assigned_at: datetime | None = Field(default=None, alias="assignedAt")
    review_due_at: datetime | None = Field(default=None, alias="reviewDueAt")
    training_eligibility_status: str = Field(alias="trainingEligibilityStatus")
    training_eligibility_updated_at: datetime | None = Field(
        default=None, alias="trainingEligibilityUpdatedAt"
    )
    training_eligibility_updated_by: str | None = Field(
        default=None, alias="trainingEligibilityUpdatedBy"
    )
    training_eligibility_notes: str | None = Field(
        default=None, alias="trainingEligibilityNotes"
    )
    training_release_status: str | None = Field(
        default=None, alias="trainingReleaseStatus"
    )
    training_release_criteria: list[str] = Field(
        default_factory=list, alias="trainingReleaseCriteria"
    )
    training_release_requested_at: datetime | None = Field(
        default=None, alias="trainingReleaseRequestedAt"
    )
    training_release_requested_by: str | None = Field(
        default=None, alias="trainingReleaseRequestedBy"
    )
    training_release_requested_notes: str | None = Field(
        default=None, alias="trainingReleaseRequestedNotes"
    )
    training_release_reviewed_at: datetime | None = Field(
        default=None, alias="trainingReleaseReviewedAt"
    )
    training_release_reviewed_by: str | None = Field(
        default=None, alias="trainingReleaseReviewedBy"
    )
    training_release_review_notes: str | None = Field(
        default=None, alias="trainingReleaseReviewNotes"
    )
    training_release_assigned_reviewer: str | None = Field(
        default=None, alias="trainingReleaseAssignedReviewer"
    )
    training_release_assigned_at: datetime | None = Field(
        default=None, alias="trainingReleaseAssignedAt"
    )
    training_release_due_at: datetime | None = Field(
        default=None, alias="trainingReleaseDueAt"
    )
    training_release_is_overdue: bool = Field(alias="trainingReleaseIsOverdue")
    training_release_escalation_status: str | None = Field(
        default=None, alias="trainingReleaseEscalationStatus"
    )
    training_release_escalation_level: int | None = Field(
        default=None, alias="trainingReleaseEscalationLevel"
    )
    training_release_escalated_at: datetime | None = Field(
        default=None, alias="trainingReleaseEscalatedAt"
    )
    training_release_escalated_by: str | None = Field(
        default=None, alias="trainingReleaseEscalatedBy"
    )
    training_release_escalation_reason: str | None = Field(
        default=None, alias="trainingReleaseEscalationReason"
    )
    training_release_is_escalated: bool = Field(alias="trainingReleaseIsEscalated")
    evidence_completeness_score: float = Field(alias="evidenceCompletenessScore")
    missing_evidence_fields: list[str] = Field(alias="missingEvidenceFields")
    ready_for_review: bool = Field(alias="readyForReview")
    is_overdue: bool = Field(alias="isOverdue")
    reviewed_at: datetime | None = Field(default=None, alias="reviewedAt")
    reviewed_by: str | None = Field(default=None, alias="reviewedBy")
    review_notes: str | None = Field(default=None, alias="reviewNotes")
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")


class UpsertOutcomeLabelsResponse(SchemaBase):
    created_count: int = Field(alias="createdCount")
    updated_count: int = Field(alias="updatedCount")
    labels: list[OutcomeLabelRead]


class ImportHistoricalLabelsResponse(SchemaBase):
    created_count: int = Field(alias="createdCount")
    updated_count: int = Field(alias="updatedCount")
    skipped_count: int = Field(alias="skippedCount")
    imported_event_ids: list[str] = Field(alias="importedEventIds")
    labels: list[OutcomeLabelRead]


class ImportUngrdLabelsResponse(SchemaBase):
    created_count: int = Field(alias="createdCount")
    updated_count: int = Field(alias="updatedCount")
    skipped_count: int = Field(alias="skippedCount")
    imported_record_ids: list[str] = Field(alias="importedRecordIds")
    labels: list[OutcomeLabelRead]


class ImportFieldValidationLabelsResponse(SchemaBase):
    created_count: int = Field(alias="createdCount")
    updated_count: int = Field(alias="updatedCount")
    skipped_count: int = Field(alias="skippedCount")
    imported_observation_ids: list[str] = Field(alias="importedObservationIds")
    labels: list[OutcomeLabelRead]


class ReviewOutcomeLabelsResponse(SchemaBase):
    reviewed_count: int = Field(alias="reviewedCount")
    labels: list[OutcomeLabelRead]


class AssignOutcomeLabelsResponse(SchemaBase):
    assigned_count: int = Field(alias="assignedCount")
    labels: list[OutcomeLabelRead]


class UpdateTrainingEligibilityResponse(SchemaBase):
    updated_count: int = Field(alias="updatedCount")
    labels: list[OutcomeLabelRead]


class RequestTrainingReleaseResponse(SchemaBase):
    requested_count: int = Field(alias="requestedCount")
    labels: list[OutcomeLabelRead]


class ReviewTrainingReleaseResponse(SchemaBase):
    reviewed_count: int = Field(alias="reviewedCount")
    labels: list[OutcomeLabelRead]


class AssignTrainingReleaseResponse(SchemaBase):
    assigned_count: int = Field(alias="assignedCount")
    labels: list[OutcomeLabelRead]


class EscalateTrainingReleaseResponse(SchemaBase):
    escalated_count: int = Field(alias="escalatedCount")
    labels: list[OutcomeLabelRead]


class ReassignTrainingReleaseResponse(SchemaBase):
    reassigned_count: int = Field(alias="reassignedCount")
    labels: list[OutcomeLabelRead]


class TriggerTrainingReleaseSlaScanResponse(SchemaBase):
    job: JobExecutionRead
    escalated_count: int = Field(alias="escalatedCount")
    notification_count: int = Field(alias="notificationCount")
    labels: list[OutcomeLabelRead]


class TriggerTrainingReleaseReassignmentScanResponse(SchemaBase):
    job: JobExecutionRead
    reassigned_count: int = Field(alias="reassignedCount")
    notification_count: int = Field(alias="notificationCount")
    labels: list[OutcomeLabelRead]


class TriggerNotificationAckScanResponse(SchemaBase):
    job: JobExecutionRead
    source_count: int = Field(alias="sourceCount")
    reminded_count: int = Field(alias="remindedCount")
    notifications: list["NotificationEventRead"]


class OutcomeLabelReviewQueueRead(SchemaBase):
    total: int
    ready_count: int = Field(alias="readyCount")
    assigned_count: int = Field(alias="assignedCount")
    overdue_count: int = Field(alias="overdueCount")
    labels: list[OutcomeLabelRead]


class OutcomeLabelReleaseQueueRead(SchemaBase):
    total: int
    assigned_count: int = Field(alias="assignedCount")
    unassigned_count: int = Field(alias="unassignedCount")
    overdue_count: int = Field(alias="overdueCount")
    escalated_count: int = Field(alias="escalatedCount")
    labels: list[OutcomeLabelRead]


class NotificationDeliveryAttemptRead(SchemaBase):
    id: int
    notification_id: int = Field(alias="notificationId")
    event_type: str = Field(alias="eventType")
    channel: str
    adapter_key: str = Field(alias="adapterKey")
    provider_name: str | None = Field(default=None, alias="providerName")
    provider_status: str | None = Field(default=None, alias="providerStatus")
    status: str
    failure_classification: str | None = Field(
        default=None, alias="failureClassification"
    )
    retryable: bool = False
    delivery_origin: str | None = Field(default=None, alias="deliveryOrigin")
    payload_preview: dict | None = Field(default=None, alias="payloadPreview")
    provider_receipt: dict | None = Field(default=None, alias="providerReceipt")
    target_username: str | None = Field(default=None, alias="targetUsername")
    related_label_id: int | None = Field(default=None, alias="relatedLabelId")
    attempted_at: datetime = Field(alias="attemptedAt")
    completed_at: datetime | None = Field(default=None, alias="completedAt")
    delivery_reference: str | None = Field(default=None, alias="deliveryReference")
    error_message: str | None = Field(default=None, alias="errorMessage")
    details: dict


class NotificationEventRead(SchemaBase):
    id: int
    event_type: str = Field(alias="eventType")
    severity: str
    status: str
    channel: str
    delivery_channels: list[str] = Field(default_factory=list, alias="deliveryChannels")
    delivery_status: str = Field(alias="deliveryStatus")
    delivery_attempt_count: int = Field(alias="deliveryAttemptCount")
    failed_delivery_count: int = Field(alias="failedDeliveryCount")
    last_delivery_attempt_at: datetime | None = Field(
        default=None, alias="lastDeliveryAttemptAt"
    )
    title: str
    message: str
    target_username: str | None = Field(default=None, alias="targetUsername")
    related_label_id: int | None = Field(default=None, alias="relatedLabelId")
    details: dict
    created_at: datetime = Field(alias="createdAt")
    ack_due_at: datetime | None = Field(default=None, alias="ackDueAt")
    reminder_count: int = Field(alias="reminderCount")
    last_reminder_at: datetime | None = Field(default=None, alias="lastReminderAt")
    is_ack_overdue: bool = Field(alias="isAckOverdue")
    acknowledged_at: datetime | None = Field(default=None, alias="acknowledgedAt")
    acknowledged_by: str | None = Field(default=None, alias="acknowledgedBy")


class RunModernLabelsBenchmarkReviewResponse(SchemaBase):
    benchmark: RunModernLabelsBenchmarkResponse
    recommendation: dict
    created_alerts: list[NotificationEventRead] = Field(
        default_factory=list, alias="createdAlerts"
    )
    acknowledged_alerts: list[NotificationEventRead] = Field(
        default_factory=list, alias="acknowledgedAlerts"
    )
    review_task: ModelReviewTaskRead | None = Field(
        default=None, alias="reviewTask"
    )


class AcknowledgeNotificationsResponse(SchemaBase):
    acknowledged_count: int = Field(alias="acknowledgedCount")
    notifications: list[NotificationEventRead]


class RetryNotificationDeliveryResponse(SchemaBase):
    job: JobExecutionRead
    retried_count: int = Field(alias="retriedCount")
    skipped_count: int = Field(alias="skippedCount")
    attempts: list[NotificationDeliveryAttemptRead]


class TriggerNotificationDeliveryRetryScanResponse(SchemaBase):
    job: JobExecutionRead
    candidate_count: int = Field(alias="candidateCount")
    retried_count: int = Field(alias="retriedCount")
    skipped_count: int = Field(alias="skippedCount")
    attempts: list[NotificationDeliveryAttemptRead]


class NotificationDeliverySummaryRead(SchemaBase):
    total_notifications: int = Field(alias="totalNotifications")
    open_notifications: int = Field(alias="openNotifications")
    acknowledged_notifications: int = Field(alias="acknowledgedNotifications")
    resolved_notifications: int = Field(alias="resolvedNotifications")
    delivery_status_counts: dict[str, int] = Field(alias="deliveryStatusCounts")
    severity_counts: dict[str, int] = Field(alias="severityCounts")
    channel_failure_counts: dict[str, int] = Field(alias="channelFailureCounts")
    provider_failure_counts: dict[str, int] = Field(alias="providerFailureCounts")
    failure_classification_counts: dict[str, int] = Field(
        alias="failureClassificationCounts"
    )
    notifications_with_failures: int = Field(alias="notificationsWithFailures")
    retryable_failure_notification_count: int = Field(
        alias="retryableFailureNotificationCount"
    )
    non_retryable_failure_notification_count: int = Field(
        alias="nonRetryableFailureNotificationCount"
    )
    max_attempt_reached_notification_count: int = Field(
        alias="maxAttemptReachedNotificationCount"
    )
    ack_overdue_count: int = Field(alias="ackOverdueCount")
    active_alert_count: int = Field(alias="activeAlertCount")
    oldest_outstanding_failure_at: datetime | None = Field(
        default=None, alias="oldestOutstandingFailureAt"
    )


class TriggerNotificationDeliveryFailureScanResponse(SchemaBase):
    job: JobExecutionRead
    candidate_count: int = Field(alias="candidateCount")
    alerted_count: int = Field(alias="alertedCount")
    skipped_count: int = Field(alias="skippedCount")
    resolved_alert_count: int = Field(alias="resolvedAlertCount")
    alerts: list[NotificationEventRead]


class ModelArtifactSummaryRead(SchemaBase):
    version: str
    model_id: str = Field(alias="modelId")
    artifact_type: str = Field(alias="artifactType")
    model_family: str = Field(alias="modelFamily")
    description: str | None = None
    active: bool
    feature_count: int = Field(alias="featureCount")
    trained_at: str | None = Field(default=None, alias="trainedAt")
    dataset: str | None = None
    rows: int | None = None
    metrics: dict[str, float]


class ModelArtifactDetailRead(SchemaBase):
    version: str
    model_id: str = Field(alias="modelId")
    artifact_type: str = Field(alias="artifactType")
    model_family: str = Field(alias="modelFamily")
    description: str | None = None
    active: bool
    artifact_path: str = Field(alias="artifactPath")
    feature_order: list[str] = Field(alias="featureOrder")
    bounds: dict
    freshness_penalties: dict[str, float] = Field(alias="freshnessPenalties")
    training: dict
    calibration: dict
    metrics: dict[str, float]


class ModelEvaluationSummaryRead(SchemaBase):
    version: str
    evaluation_id: str = Field(alias="evaluationId")
    artifact_type: str = Field(alias="artifactType")
    model_version: str = Field(alias="modelVersion")
    dataset_version: str = Field(alias="datasetVersion")
    evaluated_at: str = Field(alias="evaluatedAt")
    rows: int
    overall_rmse: float = Field(alias="overallRmse")
    validation_rmse: float = Field(alias="validationRmse")
    overall_risk_level_accuracy: float = Field(alias="overallRiskLevelAccuracy")
    validation_risk_level_accuracy: float = Field(alias="validationRiskLevelAccuracy")


class ModelEvaluationDetailRead(SchemaBase):
    version: str
    evaluation_id: str = Field(alias="evaluationId")
    artifact_type: str = Field(alias="artifactType")
    artifact_path: str = Field(alias="artifactPath")
    model_version: str = Field(alias="modelVersion")
    dataset_version: str = Field(alias="datasetVersion")
    evaluated_at: str = Field(alias="evaluatedAt")
    model_summary: dict = Field(alias="modelSummary")
    dataset_summary: dict = Field(alias="datasetSummary")
    metrics: dict
    diagnostics: dict
    top_errors: list[dict] = Field(alias="topErrors")


class ModelSelectionRunSummaryRead(SchemaBase):
    version: str
    selection_id: str = Field(alias="selectionId")
    artifact_type: str = Field(alias="artifactType")
    dataset_version: str = Field(alias="datasetVersion")
    created_at: str = Field(alias="createdAt")
    candidate_count: int = Field(alias="candidateCount")
    best_model_version: str = Field(alias="bestModelVersion")
    promoted: bool
    active_model_version: str = Field(alias="activeModelVersion")
    best_validation_rmse: float = Field(alias="bestValidationRmse")
    best_validation_risk_level_accuracy: float = Field(
        alias="bestValidationRiskLevelAccuracy"
    )
    promotion_decision: dict = Field(alias="promotionDecision")


class ModelSelectionRunDetailRead(SchemaBase):
    version: str
    selection_id: str = Field(alias="selectionId")
    artifact_type: str = Field(alias="artifactType")
    artifact_path: str = Field(alias="artifactPath")
    dataset_version: str = Field(alias="datasetVersion")
    created_at: str = Field(alias="createdAt")
    comparison_policy: dict = Field(alias="comparisonPolicy")
    gate_policy: dict = Field(alias="gatePolicy")
    dataset_context: dict = Field(alias="datasetContext")
    candidate_count: int = Field(alias="candidateCount")
    best_model_version: str = Field(alias="bestModelVersion")
    promoted: bool
    promotion: dict
    promotion_decision: dict = Field(alias="promotionDecision")
    active_model_version: str = Field(alias="activeModelVersion")
    best_vs_active_comparison: dict = Field(alias="bestVsActiveComparison")
    family_rollups: list[dict] = Field(default_factory=list, alias="familyRollups")
    nested_estimation: dict = Field(default_factory=dict, alias="nestedEstimation")
    candidates: list[ModelSelectionCandidateRead]

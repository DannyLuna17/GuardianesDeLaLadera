from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.v1.dependencies.auth import require_admin
from app.db.session import get_db
from app.models import UserAccount
from app.schemas.admin import (
    AcknowledgeNotificationsRequest,
    AcknowledgeNotificationsResponse,
    AssignTrainingReleaseRequest,
    AssignTrainingReleaseResponse,
    AssignOutcomeLabelsRequest,
    AssignOutcomeLabelsResponse,
    CreateUserAccountRequest,
    EvaluateModelRequest,
    EvaluateModelResponse,
    EscalateTrainingReleaseRequest,
    EscalateTrainingReleaseResponse,
    ExportTrainingDatasetRequest,
    ExportTrainingDatasetResponse,
    ImportFieldValidationLabelsRequest,
    ImportFieldValidationLabelsResponse,
    ImportHistoricalLabelsRequest,
    ImportHistoricalLabelsResponse,
    ImportUngrdLabelsRequest,
    ImportUngrdLabelsResponse,
    JobExecutionRead,
    ModelArtifactDetailRead,
    ModelArtifactSummaryRead,
    ModelDriftDetailRead,
    ModelDriftSummaryRead,
    ModelEvaluationDetailRead,
    ModelEvaluationSummaryRead,
    ModelReviewTaskRead,
    ScanModelMonitoringRequest,
    ScanModelMonitoringResponse,
    ModelMonitoringDetailRead,
    ModelMonitoringSummaryRead,
    ModelPromotionHistoryEntryRead,
    ModelSelectionRunDetailRead,
    ModelSelectionRunSummaryRead,
    ModelShadowRunDetailRead,
    ModelShadowRunSummaryRead,
    NotificationDeliveryAttemptRead,
    NotificationDeliverySummaryRead,
    OutcomeLabelRead,
    OutcomeLabelReleaseQueueRead,
    OutcomeLabelReviewQueueRead,
    OpenModelReviewTasksRequest,
    OpenModelReviewTasksResponse,
    PromoteModelRequest,
    PromoteModelResponse,
    ResetUserPasswordRequest,
    RollbackModelRequest,
    RollbackModelResponse,
    ScanModelDriftRequest,
    ScanModelDriftResponse,
    ScanModelShadowRequest,
    ScanModelShadowResponse,
    NotificationEventRead,
    RetryNotificationDeliveryRequest,
    RetryNotificationDeliveryResponse,
    ReassignTrainingReleaseRequest,
    ReassignTrainingReleaseResponse,
    RequestTrainingReleaseRequest,
    RequestTrainingReleaseResponse,
    RefreshExplanationRequest,
    RefreshExplanationResponse,
    ReviewTrainingReleaseRequest,
    ReviewTrainingReleaseResponse,
    ReviewOutcomeLabelsRequest,
    ReviewOutcomeLabelsResponse,
    RetrainModelRequest,
    RetrainModelResponse,
    RunModernLabelsBenchmarkRequest,
    RunModernLabelsBenchmarkResponse,
    RunModernLabelsBenchmarkReviewRequest,
    RunModernLabelsBenchmarkReviewResponse,
    SchedulerStatusRead,
    SourceSyncEventRead,
    TrainingDatasetDetailRead,
    TrainingDatasetSummaryRead,
    TuneModelRequest,
    TuneModelResponse,
    TriggerNotificationAckScanRequest,
    TriggerNotificationAckScanResponse,
    TriggerNotificationDeliveryFailureScanRequest,
    TriggerNotificationDeliveryFailureScanResponse,
    TriggerNotificationDeliveryRetryScanRequest,
    TriggerNotificationDeliveryRetryScanResponse,
    TriggerTrainingReleaseReassignmentScanRequest,
    TriggerTrainingReleaseReassignmentScanResponse,
    TriggerTrainingReleaseSlaScanRequest,
    TriggerTrainingReleaseSlaScanResponse,
    TriggerIngestionRequest,
    TriggerIngestionResponse,
    TriggerPipelineRequest,
    TriggerPipelineResponse,
    TriggerRunRequest,
    TriggerRunResponse,
    UpdateModelReviewTaskRequest,
    UpdateUserAccountRequest,
    UpdateTrainingEligibilityRequest,
    UpdateTrainingEligibilityResponse,
    UpsertOutcomeLabelsRequest,
    UpsertOutcomeLabelsResponse,
    UserAccountAdminRead,
)
from app.services.auth import AuthService
from app.services.datasets import TrainingDatasetService
from app.services.ingestion import IngestionService
from app.services.labels import OutcomeLabelService
from app.services.notification_service import AdminNotificationService
from app.services.model_evaluations import ModelEvaluationService
from app.services.model_drift import ModelDriftService
from app.services.model_monitoring import ModelMonitoringService
from app.services.model_monitoring_cycle import ModelMonitoringCycleService
from app.services.model_review_tasks import ModelReviewTaskService
from app.services.model_selection import ModelSelectionService
from app.services.model_shadow import ModelShadowService
from app.services.models import ModelService
from app.services.pipeline import PipelineService
from app.services.runs import RunService
from app.services.training import TrainingService
from app.tasks.scheduler import BackendScheduler

router = APIRouter(prefix="/v1/admin", tags=["admin"])


def get_run_service(session: Session = Depends(get_db)) -> RunService:
    return RunService(session)


def get_ingestion_service(session: Session = Depends(get_db)) -> IngestionService:
    return IngestionService(session)


def get_pipeline_service(session: Session = Depends(get_db)) -> PipelineService:
    return PipelineService(session)


def get_training_service(session: Session = Depends(get_db)) -> TrainingService:
    return TrainingService(session)


def get_training_dataset_service(
    session: Session = Depends(get_db),
) -> TrainingDatasetService:
    return TrainingDatasetService(session)


def get_outcome_label_service(
    session: Session = Depends(get_db),
) -> OutcomeLabelService:
    return OutcomeLabelService(session)


def get_notification_service(
    session: Session = Depends(get_db),
) -> AdminNotificationService:
    return AdminNotificationService(session)


def get_model_service() -> ModelService:
    return ModelService()


def get_model_evaluation_service(
    session: Session = Depends(get_db),
) -> ModelEvaluationService:
    return ModelEvaluationService(session)


def get_model_drift_service(
    session: Session = Depends(get_db),
) -> ModelDriftService:
    return ModelDriftService(session)


def get_model_selection_service(
    session: Session = Depends(get_db),
) -> ModelSelectionService:
    return ModelSelectionService(session)


def get_model_shadow_service(
    session: Session = Depends(get_db),
) -> ModelShadowService:
    return ModelShadowService(session)


def get_model_monitoring_service() -> ModelMonitoringService:
    return ModelMonitoringService()


def get_model_monitoring_cycle_service(
    session: Session = Depends(get_db),
) -> ModelMonitoringCycleService:
    return ModelMonitoringCycleService(session)


def get_model_review_task_service(
    session: Session = Depends(get_db),
) -> ModelReviewTaskService:
    return ModelReviewTaskService(session)


def get_auth_admin_service(session: Session = Depends(get_db)) -> AuthService:
    return AuthService(session)


@router.post("/ingestion/trigger", response_model=TriggerIngestionResponse)
def trigger_ingestion(
    payload: TriggerIngestionRequest | None = None,
    _: object = Depends(require_admin),
    service: IngestionService = Depends(get_ingestion_service),
):
    sources = payload.sources if payload else None
    note = payload.note if payload else None
    return service.sync_sources(source_ids=sources, origin="manual", note=note)


@router.get("/ingestion/history", response_model=list[SourceSyncEventRead])
def list_ingestion_history(
    source_id: str | None = Query(default=None, alias="sourceId"),
    limit: int = Query(default=20, ge=1, le=100),
    _: object = Depends(require_admin),
    service: IngestionService = Depends(get_ingestion_service),
):
    return service.list_sync_events(source_id=source_id, limit=limit)


@router.post("/pipeline/trigger", response_model=TriggerPipelineResponse)
def trigger_pipeline(
    payload: TriggerPipelineRequest | None = None,
    _: object = Depends(require_admin),
    service: PipelineService = Depends(get_pipeline_service),
):
    sources = payload.sources if payload else None
    note = payload.note if payload else None
    return service.trigger_full_pipeline(sources=sources, note=note, origin="manual")


@router.post("/retrain", response_model=RetrainModelResponse)
def retrain_model(
    payload: RetrainModelRequest | None = None,
    admin_user: UserAccount = Depends(require_admin),
    service: TrainingService = Depends(get_training_service),
    review_service: ModelReviewTaskService = Depends(get_model_review_task_service),
):
    version = payload.version if payload else None
    alpha = payload.alpha if payload else 0.75
    model_family = payload.model_family if payload else "linear_ridge"
    knot_count = payload.knot_count if payload else None
    learning_rate = payload.learning_rate if payload else None
    estimator_count = payload.estimator_count if payload else None
    max_depth = payload.max_depth if payload else None
    min_leaf_size = payload.min_leaf_size if payload else None
    min_split_gain = payload.min_split_gain if payload else None
    early_stopping_rounds = payload.early_stopping_rounds if payload else None
    dataset_version = payload.dataset_version if payload else None
    review_task_id = payload.review_task_id if payload else None
    if review_task_id is not None:
        review_service.validate_action_guardrail(
            task_id=review_task_id,
            action_type="retraining",
            active_model_version=service.model_registry.active_version(),
            dataset_version=dataset_version,
        )
    response = service.retrain_seed_model(
        version=version,
        alpha=alpha,
        model_family=model_family,
        knot_count=knot_count,
        learning_rate=learning_rate,
        estimator_count=estimator_count,
        max_depth=max_depth,
        min_leaf_size=min_leaf_size,
        min_split_gain=min_split_gain,
        early_stopping_rounds=early_stopping_rounds,
        dataset_version=dataset_version,
        origin="manual",
    )
    if review_task_id is not None:
        review_service.record_governed_action(
            task_id=review_task_id,
            action_type="retraining",
            actor=admin_user.username,
            outcome={
                "model_version": response.model_version,
                "dataset_version": response.dataset_version,
                "job_id": response.job.id,
            },
        )
    return response


@router.post("/models/evaluate", response_model=EvaluateModelResponse)
def evaluate_model(
    payload: EvaluateModelRequest,
    _: object = Depends(require_admin),
    service: ModelEvaluationService = Depends(get_model_evaluation_service),
):
    return service.evaluate_model(
        model_version=payload.model_version,
        dataset_version=payload.dataset_version,
        version=payload.version,
        top_error_count=payload.top_error_count,
        origin="manual",
    )


@router.post("/models/drift-scan", response_model=ScanModelDriftResponse)
def scan_model_drift(
    payload: ScanModelDriftRequest | None = None,
    _: object = Depends(require_admin),
    service: ModelDriftService = Depends(get_model_drift_service),
):
    return service.scan_model_drift(
        model_version=payload.model_version if payload else None,
        dataset_version=payload.dataset_version if payload else None,
        version=payload.version if payload else None,
        evaluation_version=payload.evaluation_version if payload else None,
        top_error_count=payload.top_error_count if payload else 10,
        warning_validation_rmse_increase=payload.warning_validation_rmse_increase
        if payload
        else None,
        critical_validation_rmse_increase=payload.critical_validation_rmse_increase
        if payload
        else None,
        warning_accuracy_drop=payload.warning_accuracy_drop if payload else None,
        critical_accuracy_drop=payload.critical_accuracy_drop if payload else None,
        origin="manual",
    )


@router.post("/models/tune", response_model=TuneModelResponse)
def tune_model(
    payload: TuneModelRequest,
    admin_user: UserAccount = Depends(require_admin),
    service: ModelSelectionService = Depends(get_model_selection_service),
):
    return service.tune_model(
        dataset_version=payload.dataset_version,
        alphas=payload.alphas,
        model_family=payload.model_family,
        model_families=payload.model_families,
        selection_mode=payload.selection_mode,
        validation_strategy=payload.validation_strategy,
        validation_fold_count=payload.validation_fold_count,
        nested_outer_fold_count=payload.nested_outer_fold_count,
        knot_counts=payload.knot_counts,
        learning_rates=payload.learning_rates,
        estimator_counts=payload.estimator_counts,
        max_depths=payload.max_depths,
        min_leaf_sizes=payload.min_leaf_sizes,
        min_split_gains=payload.min_split_gains,
        early_stopping_rounds=payload.early_stopping_rounds,
        version=payload.version,
        version_prefix=payload.version_prefix,
        promote_best=payload.promote_best,
        promotion_reason=payload.promotion_reason,
        top_error_count=payload.top_error_count,
        min_validation_rmse_improvement=payload.min_validation_rmse_improvement,
        min_validation_rows=payload.min_validation_rows,
        require_labels_dataset_for_promotion=payload.require_labels_dataset_for_promotion,
        require_nested_estimation_for_promotion=payload.require_nested_estimation_for_promotion,
        min_calibration_gain=payload.min_calibration_gain,
        min_nested_outer_validation_rmse_improvement=payload.min_nested_outer_validation_rmse_improvement,
        min_nested_outer_selection_rate=payload.min_nested_outer_selection_rate,
        require_nested_temporal_latest_win_for_promotion=payload.require_nested_temporal_latest_win_for_promotion,
        min_nested_temporal_outer_bucket_count=payload.min_nested_temporal_outer_bucket_count,
        min_nested_temporal_latest_validation_rmse_improvement=payload.min_nested_temporal_latest_validation_rmse_improvement,
        nested_temporal_recent_window_size=payload.nested_temporal_recent_window_size,
        min_nested_temporal_recent_win_rate=payload.min_nested_temporal_recent_win_rate,
        min_nested_temporal_recent_average_validation_rmse_improvement=payload.min_nested_temporal_recent_average_validation_rmse_improvement,
        max_spatial_slice_validation_rmse_regression=payload.max_spatial_slice_validation_rmse_regression,
        max_temporal_slice_validation_rmse_regression=payload.max_temporal_slice_validation_rmse_regression,
        max_spatial_slice_regression_count=payload.max_spatial_slice_regression_count,
        max_temporal_slice_regression_count=payload.max_temporal_slice_regression_count,
        slice_regression_min_rows=payload.slice_regression_min_rows,
        stability_window_runs=payload.stability_window_runs,
        required_consistent_wins=payload.required_consistent_wins,
        stability_require_same_dataset_family=payload.stability_require_same_dataset_family,
        stability_require_same_dataset_taxonomy=payload.stability_require_same_dataset_taxonomy,
        stability_require_same_evaluation_cohort=payload.stability_require_same_evaluation_cohort,
        stability_max_time_window_gap_days=payload.stability_max_time_window_gap_days,
        stability_max_cohort_distance=payload.stability_max_cohort_distance,
        promoted_by=admin_user.username,
        origin="manual",
    )


@router.post(
    "/models/benchmark/modern-labels",
    response_model=RunModernLabelsBenchmarkResponse,
)
def run_modern_labels_benchmark(
    payload: RunModernLabelsBenchmarkRequest | None = None,
    admin_user: UserAccount = Depends(require_admin),
    service: ModelSelectionService = Depends(get_model_selection_service),
):
    return service.run_modern_labels_benchmark(
        dataset_version=payload.dataset_version if payload else None,
        auto_export_dataset=payload.auto_export_dataset if payload else True,
        dataset_export_version=payload.dataset_export_version if payload else None,
        label_sources=payload.label_sources if payload else None,
        max_labels=payload.max_labels if payload else 250,
        observed_after=payload.observed_after if payload else None,
        observed_before=payload.observed_before if payload else None,
        validation_strategy=payload.validation_strategy if payload else None,
        validation_fold_count=payload.validation_fold_count if payload else None,
        nested_outer_fold_count=payload.nested_outer_fold_count if payload else None,
        version=payload.version if payload else None,
        version_prefix=payload.version_prefix if payload else None,
        promote_best=payload.promote_best if payload else False,
        promotion_reason=payload.promotion_reason if payload else None,
        promoted_by=admin_user.username,
        origin="manual",
    )


@router.post(
    "/models/benchmark/modern-labels/review",
    response_model=RunModernLabelsBenchmarkReviewResponse,
)
def run_modern_labels_benchmark_review(
    payload: RunModernLabelsBenchmarkReviewRequest | None = None,
    admin_user: UserAccount = Depends(require_admin),
    service: ModelSelectionService = Depends(get_model_selection_service),
):
    return service.run_modern_labels_benchmark_with_review(
        dataset_version=payload.dataset_version if payload else None,
        auto_export_dataset=payload.auto_export_dataset if payload else True,
        dataset_export_version=payload.dataset_export_version if payload else None,
        label_sources=payload.label_sources if payload else None,
        max_labels=payload.max_labels if payload else 250,
        observed_after=payload.observed_after if payload else None,
        observed_before=payload.observed_before if payload else None,
        validation_strategy=payload.validation_strategy if payload else None,
        validation_fold_count=payload.validation_fold_count if payload else None,
        nested_outer_fold_count=payload.nested_outer_fold_count if payload else None,
        version=payload.version if payload else None,
        version_prefix=payload.version_prefix if payload else None,
        promote_best=payload.promote_best if payload else False,
        promotion_reason=payload.promotion_reason if payload else None,
        open_review_task=payload.open_review_task if payload else True,
        assigned_reviewer=payload.assigned_reviewer if payload else None,
        due_at=payload.due_at if payload else None,
        notes=payload.notes if payload else None,
        opened_by=admin_user.username,
        origin="manual",
    )


@router.post("/models/shadow-scan", response_model=ScanModelShadowResponse)
def scan_model_shadow(
    payload: ScanModelShadowRequest,
    _: object = Depends(require_admin),
    service: ModelShadowService = Depends(get_model_shadow_service),
):
    return service.scan_shadow_run(
        dataset_version=payload.dataset_version,
        model_versions=payload.model_versions,
        version=payload.version,
        max_candidates=payload.max_candidates,
        top_error_count=payload.top_error_count,
        origin="manual",
    )


@router.post("/models/monitoring-scan", response_model=ScanModelMonitoringResponse)
def scan_model_monitoring(
    payload: ScanModelMonitoringRequest | None = None,
    _: object = Depends(require_admin),
    service: ModelMonitoringCycleService = Depends(get_model_monitoring_cycle_service),
):
    return service.run_monitoring_cycle(
        dataset_version=payload.dataset_version if payload else None,
        drift_top_error_count=payload.drift_top_error_count if payload else 10,
        shadow_top_error_count=payload.shadow_top_error_count if payload else 5,
        shadow_max_candidates=payload.shadow_max_candidates if payload else 4,
        origin="manual",
    )


@router.post("/models/promote", response_model=PromoteModelResponse)
def promote_model(
    payload: PromoteModelRequest,
    admin_user: UserAccount = Depends(require_admin),
    service: ModelService = Depends(get_model_service),
    review_service: ModelReviewTaskService = Depends(get_model_review_task_service),
):
    if payload.review_task_id is not None:
        review_service.validate_action_guardrail(
            task_id=payload.review_task_id,
            action_type="promotion",
            candidate_model_version=payload.model_version,
            active_model_version=service.registry.active_version(),
        )
    response = service.promote_model(
        payload.model_version,
        promoted_by=admin_user.username,
        reason=payload.reason,
        source="manual",
    )
    if payload.review_task_id is not None:
        review_service.record_governed_action(
            task_id=payload.review_task_id,
            action_type="promotion",
            actor=admin_user.username,
            outcome={
                "model_version": response["modelVersion"],
                "active_model_version": response["activeModelVersion"],
                "previous_active_model_version": response["previousActiveModelVersion"],
            },
        )
    return response


@router.post("/models/rollback", response_model=RollbackModelResponse)
def rollback_model(
    payload: RollbackModelRequest | None = None,
    admin_user: UserAccount = Depends(require_admin),
    service: ModelService = Depends(get_model_service),
    review_service: ModelReviewTaskService = Depends(get_model_review_task_service),
):
    reason = payload.reason if payload else None
    review_task_id = payload.review_task_id if payload else None
    if review_task_id is not None:
        review_service.validate_action_guardrail(
            task_id=review_task_id,
            action_type="rollback",
            active_model_version=service.registry.active_version(),
        )
    response = service.rollback_model(
        rolled_back_by=admin_user.username,
        reason=reason,
        source="manual",
    )
    if review_task_id is not None:
        review_service.record_governed_action(
            task_id=review_task_id,
            action_type="rollback",
            actor=admin_user.username,
            outcome={
                "model_version": response["modelVersion"],
                "rolled_back_from_model_version": response["rolledBackFromModelVersion"],
            },
        )
    return response


@router.get("/model-evaluations", response_model=list[ModelEvaluationSummaryRead])
def list_model_evaluations(
    _: object = Depends(require_admin),
    service: ModelEvaluationService = Depends(get_model_evaluation_service),
):
    return service.list_evaluations()


@router.get("/model-evaluations/{version}", response_model=ModelEvaluationDetailRead)
def get_model_evaluation(
    version: str,
    _: object = Depends(require_admin),
    service: ModelEvaluationService = Depends(get_model_evaluation_service),
):
    return service.get_evaluation(version)


@router.get("/model-drift-reports", response_model=list[ModelDriftSummaryRead])
def list_model_drift_reports(
    _: object = Depends(require_admin),
    service: ModelDriftService = Depends(get_model_drift_service),
):
    return service.list_drift_reports()


@router.get("/model-drift-reports/{version}", response_model=ModelDriftDetailRead)
def get_model_drift_report(
    version: str,
    _: object = Depends(require_admin),
    service: ModelDriftService = Depends(get_model_drift_service),
):
    return service.get_drift_report(version)


@router.get("/model-selection-runs", response_model=list[ModelSelectionRunSummaryRead])
def list_model_selection_runs(
    _: object = Depends(require_admin),
    service: ModelSelectionService = Depends(get_model_selection_service),
):
    return service.list_selection_runs()


@router.get(
    "/model-selection-runs/{version}", response_model=ModelSelectionRunDetailRead
)
def get_model_selection_run(
    version: str,
    _: object = Depends(require_admin),
    service: ModelSelectionService = Depends(get_model_selection_service),
):
    return service.get_selection_run(version)


@router.get("/model-shadow-runs", response_model=list[ModelShadowRunSummaryRead])
def list_model_shadow_runs(
    _: object = Depends(require_admin),
    service: ModelShadowService = Depends(get_model_shadow_service),
):
    return service.list_shadow_runs()


@router.get("/model-shadow-runs/{version}", response_model=ModelShadowRunDetailRead)
def get_model_shadow_run(
    version: str,
    _: object = Depends(require_admin),
    service: ModelShadowService = Depends(get_model_shadow_service),
):
    return service.get_shadow_run(version)


@router.get(
    "/models/promotion-history",
    response_model=list[ModelPromotionHistoryEntryRead],
)
def list_model_promotion_history(
    model_version: str | None = Query(default=None, alias="modelVersion"),
    _: object = Depends(require_admin),
    service: ModelMonitoringService = Depends(get_model_monitoring_service),
):
    return service.list_promotion_history(model_version=model_version)


@router.get("/models/monitoring", response_model=list[ModelMonitoringSummaryRead])
def list_model_monitoring(
    _: object = Depends(require_admin),
    service: ModelMonitoringService = Depends(get_model_monitoring_service),
):
    return service.list_model_monitoring()


@router.get("/models/monitoring/{version}", response_model=ModelMonitoringDetailRead)
def get_model_monitoring(
    version: str,
    _: object = Depends(require_admin),
    service: ModelMonitoringService = Depends(get_model_monitoring_service),
):
    return service.get_model_monitoring(version)


@router.post(
    "/models/review-tasks/open-from-alerts",
    response_model=OpenModelReviewTasksResponse,
)
def open_model_review_tasks_from_alerts(
    payload: OpenModelReviewTasksRequest,
    admin_user: UserAccount = Depends(require_admin),
    service: ModelReviewTaskService = Depends(get_model_review_task_service),
):
    return service.open_review_tasks_from_alerts(
        notification_ids=payload.notification_ids,
        review_type=payload.review_type,
        opened_by=admin_user.username,
        assigned_reviewer=payload.assigned_reviewer,
        due_at=payload.due_at,
        notes=payload.notes,
    )


@router.get("/models/review-tasks", response_model=list[ModelReviewTaskRead])
def list_model_review_tasks(
    review_type: str | None = Query(default=None, alias="reviewType"),
    status: str | None = Query(default=None),
    assigned_reviewer: str | None = Query(default=None, alias="assignedReviewer"),
    source_notification_id: int | None = Query(
        default=None, alias="sourceNotificationId"
    ),
    active_model_version: str | None = Query(default=None, alias="activeModelVersion"),
    candidate_model_version: str | None = Query(
        default=None, alias="candidateModelVersion"
    ),
    limit: int = Query(default=100, ge=1, le=500),
    _: object = Depends(require_admin),
    service: ModelReviewTaskService = Depends(get_model_review_task_service),
):
    return service.list_review_tasks(
        review_type=review_type,
        status=status,
        assigned_reviewer=assigned_reviewer,
        source_notification_id=source_notification_id,
        active_model_version=active_model_version,
        candidate_model_version=candidate_model_version,
        limit=limit,
    )


@router.get("/models/review-tasks/{task_id}", response_model=ModelReviewTaskRead)
def get_model_review_task(
    task_id: int,
    _: object = Depends(require_admin),
    service: ModelReviewTaskService = Depends(get_model_review_task_service),
):
    return service.get_review_task(task_id)


@router.post(
    "/models/review-tasks/{task_id}/update",
    response_model=ModelReviewTaskRead,
)
def update_model_review_task(
    task_id: int,
    payload: UpdateModelReviewTaskRequest,
    admin_user: UserAccount = Depends(require_admin),
    service: ModelReviewTaskService = Depends(get_model_review_task_service),
):
    return service.update_review_task(
        task_id=task_id,
        updated_by=admin_user.username,
        status=payload.status,
        assigned_reviewer=payload.assigned_reviewer,
        due_at=payload.due_at,
        decision=payload.decision,
        notes=payload.notes,
        provided_fields=set(payload.model_fields_set),
    )


@router.get("/users", response_model=list[UserAccountAdminRead])
def list_user_accounts(
    _: object = Depends(require_admin),
    service: AuthService = Depends(get_auth_admin_service),
):
    return service.list_users()


@router.post("/users", response_model=UserAccountAdminRead)
def create_user_account(
    payload: CreateUserAccountRequest,
    admin_user: UserAccount = Depends(require_admin),
    service: AuthService = Depends(get_auth_admin_service),
):
    return service.create_user(
        username=payload.username,
        password=payload.password,
        role=payload.role,
        is_active=payload.is_active,
        created_by=admin_user.username,
    )


@router.patch("/users/{username}", response_model=UserAccountAdminRead)
def update_user_account(
    username: str,
    payload: UpdateUserAccountRequest,
    admin_user: UserAccount = Depends(require_admin),
    service: AuthService = Depends(get_auth_admin_service),
):
    return service.update_user(
        username=username,
        updated_by=admin_user.username,
        role=payload.role,
        is_active=payload.is_active,
    )


@router.post("/users/{username}/password-reset", response_model=UserAccountAdminRead)
def reset_user_password(
    username: str,
    payload: ResetUserPasswordRequest,
    admin_user: UserAccount = Depends(require_admin),
    service: AuthService = Depends(get_auth_admin_service),
):
    return service.reset_password(
        username=username,
        new_password=payload.new_password,
        rotated_by=admin_user.username,
    )


@router.post("/training-datasets/export", response_model=ExportTrainingDatasetResponse)
def export_training_dataset(
    payload: ExportTrainingDatasetRequest | None = None,
    _: object = Depends(require_admin),
    service: TrainingDatasetService = Depends(get_training_dataset_service),
):
    version = payload.version if payload else None
    source_mode = payload.source_mode if payload else "seed"
    run_ids = payload.run_ids if payload else None
    max_runs = payload.max_runs if payload else 5
    label_ids = payload.label_ids if payload else None
    label_sources = payload.label_sources if payload else None
    max_labels = payload.max_labels if payload else 100
    observed_after = payload.observed_after if payload else None
    observed_before = payload.observed_before if payload else None
    return service.export_dataset(
        version=version,
        source_mode=source_mode,
        run_ids=run_ids,
        max_runs=max_runs,
        label_ids=label_ids,
        label_sources=label_sources,
        max_labels=max_labels,
        observed_after=observed_after,
        observed_before=observed_before,
        origin="manual",
    )


@router.get("/training-datasets", response_model=list[TrainingDatasetSummaryRead])
def list_training_datasets(
    _: object = Depends(require_admin),
    service: TrainingDatasetService = Depends(get_training_dataset_service),
):
    return service.list_datasets()


@router.get("/training-datasets/{version}", response_model=TrainingDatasetDetailRead)
def get_training_dataset(
    version: str,
    sample_size: int = Query(default=5, alias="sampleSize", ge=1, le=20),
    _: object = Depends(require_admin),
    service: TrainingDatasetService = Depends(get_training_dataset_service),
):
    return service.get_dataset(version, sample_size=sample_size)


@router.post("/labels/upsert", response_model=UpsertOutcomeLabelsResponse)
def upsert_outcome_labels(
    payload: UpsertOutcomeLabelsRequest,
    _: object = Depends(require_admin),
    service: OutcomeLabelService = Depends(get_outcome_label_service),
):
    return service.upsert_labels(payload.labels)


@router.post(
    "/labels/import/historical-events", response_model=ImportHistoricalLabelsResponse
)
def import_historical_event_labels(
    payload: ImportHistoricalLabelsRequest | None = None,
    _: object = Depends(require_admin),
    service: OutcomeLabelService = Depends(get_outcome_label_service),
):
    municipality = payload.municipality if payload else None
    zone_id = payload.zone_id if payload else None
    event_ids = payload.event_ids if payload else None
    event_source = payload.event_source if payload else None
    status = payload.status if payload else "draft"
    max_events = payload.max_events if payload else 100
    severity_score_overrides = payload.severity_score_overrides if payload else None
    return service.import_historical_event_labels(
        municipality=municipality,
        zone_id=zone_id,
        event_ids=event_ids,
        event_source=event_source,
        status=status,
        max_events=max_events,
        severity_score_overrides=severity_score_overrides,
    )


@router.post("/labels/import/ungrd-records", response_model=ImportUngrdLabelsResponse)
def import_ungrd_record_labels(
    payload: ImportUngrdLabelsRequest | None = None,
    _: object = Depends(require_admin),
    service: OutcomeLabelService = Depends(get_outcome_label_service),
):
    municipality = payload.municipality if payload else None
    zone_id = payload.zone_id if payload else None
    record_ids = payload.record_ids if payload else None
    status = payload.status if payload else "draft"
    max_records = payload.max_records if payload else 50
    max_zones_per_record = payload.max_zones_per_record if payload else 2
    summary_score_overrides = payload.summary_score_overrides if payload else None
    return service.import_ungrd_record_labels(
        municipality=municipality,
        zone_id=zone_id,
        record_ids=record_ids,
        status=status,
        max_records=max_records,
        max_zones_per_record=max_zones_per_record,
        summary_score_overrides=summary_score_overrides,
    )


@router.post(
    "/labels/import/field-validations",
    response_model=ImportFieldValidationLabelsResponse,
)
def import_field_validation_labels(
    payload: ImportFieldValidationLabelsRequest,
    _: object = Depends(require_admin),
    service: OutcomeLabelService = Depends(get_outcome_label_service),
):
    return service.import_field_validation_labels(
        observations=payload.observations,
        severity_score_overrides=payload.severity_score_overrides,
    )


@router.post("/labels/review", response_model=ReviewOutcomeLabelsResponse)
def review_outcome_labels(
    payload: ReviewOutcomeLabelsRequest,
    admin_user: UserAccount = Depends(require_admin),
    service: OutcomeLabelService = Depends(get_outcome_label_service),
):
    return service.review_labels(
        label_ids=payload.label_ids,
        decision=payload.decision,
        reviewer_username=admin_user.username,
        review_notes=payload.review_notes,
    )


@router.post("/labels/assign", response_model=AssignOutcomeLabelsResponse)
def assign_outcome_labels(
    payload: AssignOutcomeLabelsRequest,
    admin_user: UserAccount = Depends(require_admin),
    service: OutcomeLabelService = Depends(get_outcome_label_service),
):
    return service.assign_labels(
        label_ids=payload.label_ids,
        reviewer_username=payload.reviewer_username,
        assigned_by=admin_user.username,
        review_due_at=payload.review_due_at,
        assignment_notes=payload.assignment_notes,
    )


@router.get("/labels/review-queue", response_model=OutcomeLabelReviewQueueRead)
def list_review_queue(
    assigned_reviewer: str | None = Query(default=None, alias="assignedReviewer"),
    ready_for_review: bool | None = Query(default=None, alias="readyForReview"),
    limit: int = Query(default=100, ge=1, le=500),
    _: object = Depends(require_admin),
    service: OutcomeLabelService = Depends(get_outcome_label_service),
):
    return service.list_review_queue(
        assigned_reviewer=assigned_reviewer,
        ready_for_review=ready_for_review,
        limit=limit,
    )


@router.post(
    "/labels/training-eligibility", response_model=UpdateTrainingEligibilityResponse
)
def update_training_eligibility(
    payload: UpdateTrainingEligibilityRequest,
    admin_user: UserAccount = Depends(require_admin),
    service: OutcomeLabelService = Depends(get_outcome_label_service),
):
    return service.update_training_eligibility(
        label_ids=payload.label_ids,
        training_eligibility_status=payload.training_eligibility_status,
        updated_by=admin_user.username,
        notes=payload.notes,
    )


@router.post(
    "/labels/training-eligibility/release-request",
    response_model=RequestTrainingReleaseResponse,
)
def request_training_release(
    payload: RequestTrainingReleaseRequest,
    admin_user: UserAccount = Depends(require_admin),
    service: OutcomeLabelService = Depends(get_outcome_label_service),
):
    return service.request_training_release(
        label_ids=payload.label_ids,
        release_criteria=payload.release_criteria,
        requested_by=admin_user.username,
        notes=payload.notes,
    )


@router.post(
    "/labels/training-eligibility/release-assign",
    response_model=AssignTrainingReleaseResponse,
)
def assign_training_release(
    payload: AssignTrainingReleaseRequest,
    admin_user: UserAccount = Depends(require_admin),
    service: OutcomeLabelService = Depends(get_outcome_label_service),
):
    return service.assign_training_release(
        label_ids=payload.label_ids,
        reviewer_username=payload.reviewer_username,
        assigned_by=admin_user.username,
        review_due_at=payload.review_due_at,
        assignment_notes=payload.assignment_notes,
    )


@router.post(
    "/labels/training-eligibility/release-escalate",
    response_model=EscalateTrainingReleaseResponse,
)
def escalate_training_release(
    payload: EscalateTrainingReleaseRequest,
    admin_user: UserAccount = Depends(require_admin),
    service: OutcomeLabelService = Depends(get_outcome_label_service),
):
    return service.escalate_training_release(
        label_ids=payload.label_ids,
        escalation_reason=payload.escalation_reason,
        escalated_by=admin_user.username,
        escalation_level=payload.escalation_level,
    )


@router.post(
    "/labels/training-eligibility/release-sla-scan",
    response_model=TriggerTrainingReleaseSlaScanResponse,
)
def run_training_release_sla_scan(
    payload: TriggerTrainingReleaseSlaScanRequest | None = None,
    _: object = Depends(require_admin),
    service: OutcomeLabelService = Depends(get_outcome_label_service),
):
    max_labels = payload.max_labels if payload else 100
    note = payload.note if payload else None
    return service.run_training_release_sla_scan(
        max_labels=max_labels,
        note=note,
        origin="manual",
    )


@router.post(
    "/labels/training-eligibility/release-reassign",
    response_model=ReassignTrainingReleaseResponse,
)
def reassign_training_release(
    payload: ReassignTrainingReleaseRequest,
    admin_user: UserAccount = Depends(require_admin),
    service: OutcomeLabelService = Depends(get_outcome_label_service),
):
    return service.reassign_training_release(
        label_ids=payload.label_ids,
        reviewer_username=payload.reviewer_username,
        reassigned_by=admin_user.username,
        reassignment_reason=payload.reassignment_reason,
        review_due_at=payload.review_due_at,
    )


@router.post(
    "/labels/training-eligibility/release-reassignment-scan",
    response_model=TriggerTrainingReleaseReassignmentScanResponse,
)
def run_training_release_reassignment_scan(
    payload: TriggerTrainingReleaseReassignmentScanRequest | None = None,
    _: object = Depends(require_admin),
    service: OutcomeLabelService = Depends(get_outcome_label_service),
):
    max_labels = payload.max_labels if payload else 100
    note = payload.note if payload else None
    return service.run_training_release_reassignment_scan(
        max_labels=max_labels,
        note=note,
        origin="manual",
    )


@router.post(
    "/labels/training-eligibility/release-review",
    response_model=ReviewTrainingReleaseResponse,
)
def review_training_release(
    payload: ReviewTrainingReleaseRequest,
    admin_user: UserAccount = Depends(require_admin),
    service: OutcomeLabelService = Depends(get_outcome_label_service),
):
    return service.review_training_release(
        label_ids=payload.label_ids,
        decision=payload.decision,
        reviewed_by=admin_user.username,
        notes=payload.notes,
    )


@router.get(
    "/labels/training-release-queue", response_model=OutcomeLabelReleaseQueueRead
)
def list_training_release_queue(
    assigned_reviewer: str | None = Query(default=None, alias="assignedReviewer"),
    overdue_only: bool = Query(default=False, alias="overdueOnly"),
    escalated_only: bool = Query(default=False, alias="escalatedOnly"),
    limit: int = Query(default=100, ge=1, le=500),
    _: object = Depends(require_admin),
    service: OutcomeLabelService = Depends(get_outcome_label_service),
):
    return service.list_training_release_queue(
        assigned_reviewer=assigned_reviewer,
        overdue_only=overdue_only,
        escalated_only=escalated_only,
        limit=limit,
    )


@router.get("/labels", response_model=list[OutcomeLabelRead])
def list_outcome_labels(
    zone_id: str | None = Query(default=None, alias="zoneId"),
    source: str | None = Query(default=None),
    status: str | None = Query(default=None),
    training_eligibility_status: str | None = Query(
        default=None, alias="trainingEligibilityStatus"
    ),
    training_release_status: str | None = Query(
        default=None, alias="trainingReleaseStatus"
    ),
    training_release_escalation_status: str | None = Query(
        default=None, alias="trainingReleaseEscalationStatus"
    ),
    limit: int = Query(default=100, ge=1, le=500),
    _: object = Depends(require_admin),
    service: OutcomeLabelService = Depends(get_outcome_label_service),
):
    return service.list_labels(
        zone_id=zone_id,
        source=source,
        status=status,
        training_eligibility_status=training_eligibility_status,
        training_release_status=training_release_status,
        training_release_escalation_status=training_release_escalation_status,
        limit=limit,
    )


@router.get("/notifications", response_model=list[NotificationEventRead])
def list_notifications(
    status: str | None = Query(default=None),
    severity: str | None = Query(default=None),
    target_username: str | None = Query(default=None, alias="targetUsername"),
    event_type: str | None = Query(default=None, alias="eventType"),
    channel: str | None = Query(default=None),
    delivery_status: str | None = Query(default=None, alias="deliveryStatus"),
    overdue_only: bool = Query(default=False, alias="overdueOnly"),
    limit: int = Query(default=100, ge=1, le=500),
    _: object = Depends(require_admin),
    service: AdminNotificationService = Depends(get_notification_service),
):
    return service.list_notifications(
        status=status,
        severity=severity,
        target_username=target_username,
        event_type=event_type,
        channel=channel,
        delivery_status=delivery_status,
        overdue_only=overdue_only,
        limit=limit,
    )


@router.get(
    "/notifications/delivery-summary", response_model=NotificationDeliverySummaryRead
)
def get_notification_delivery_summary(
    _: object = Depends(require_admin),
    service: AdminNotificationService = Depends(get_notification_service),
):
    return service.get_notification_delivery_summary()


@router.get(
    "/notifications/delivery-attempts",
    response_model=list[NotificationDeliveryAttemptRead],
)
def list_notification_delivery_attempts(
    notification_id: int | None = Query(default=None, alias="notificationId"),
    channel: str | None = Query(default=None),
    status: str | None = Query(default=None),
    target_username: str | None = Query(default=None, alias="targetUsername"),
    provider_name: str | None = Query(default=None, alias="providerName"),
    failure_classification: str | None = Query(
        default=None, alias="failureClassification"
    ),
    delivery_origin: str | None = Query(default=None, alias="deliveryOrigin"),
    limit: int = Query(default=100, ge=1, le=500),
    _: object = Depends(require_admin),
    service: AdminNotificationService = Depends(get_notification_service),
):
    return service.list_notification_delivery_attempts(
        notification_id=notification_id,
        channel=channel,
        status=status,
        target_username=target_username,
        provider_name=provider_name,
        failure_classification=failure_classification,
        delivery_origin=delivery_origin,
        limit=limit,
    )


@router.post(
    "/notifications/acknowledge", response_model=AcknowledgeNotificationsResponse
)
def acknowledge_notifications(
    payload: AcknowledgeNotificationsRequest,
    admin_user: UserAccount = Depends(require_admin),
    service: AdminNotificationService = Depends(get_notification_service),
):
    return service.acknowledge_notifications(
        notification_ids=payload.notification_ids,
        acknowledged_by=admin_user.username,
    )


@router.post(
    "/notifications/retry-delivery", response_model=RetryNotificationDeliveryResponse
)
def retry_notification_delivery(
    payload: RetryNotificationDeliveryRequest,
    admin_user: UserAccount = Depends(require_admin),
    service: AdminNotificationService = Depends(get_notification_service),
):
    return service.retry_notification_delivery(
        notification_ids=payload.notification_ids,
        triggered_by=admin_user.username,
        channels=payload.channels,
        note=payload.note,
        origin="manual",
    )


@router.post(
    "/notifications/retry-scan",
    response_model=TriggerNotificationDeliveryRetryScanResponse,
)
def run_notification_delivery_retry_scan(
    payload: TriggerNotificationDeliveryRetryScanRequest | None = None,
    _: object = Depends(require_admin),
    service: AdminNotificationService = Depends(get_notification_service),
):
    max_notifications = payload.max_notifications if payload else 100
    note = payload.note if payload else None
    return service.run_notification_delivery_retry_scan(
        max_notifications=max_notifications,
        note=note,
        origin="manual",
    )


@router.post(
    "/notifications/delivery-failure-scan",
    response_model=TriggerNotificationDeliveryFailureScanResponse,
)
def run_notification_delivery_failure_scan(
    payload: TriggerNotificationDeliveryFailureScanRequest | None = None,
    _: object = Depends(require_admin),
    service: AdminNotificationService = Depends(get_notification_service),
):
    max_notifications = payload.max_notifications if payload else 100
    note = payload.note if payload else None
    return service.run_notification_delivery_failure_scan(
        max_notifications=max_notifications,
        note=note,
        origin="manual",
    )


@router.post(
    "/notifications/ack-deadline-scan",
    response_model=TriggerNotificationAckScanResponse,
)
def run_notification_ack_scan(
    payload: TriggerNotificationAckScanRequest | None = None,
    _: object = Depends(require_admin),
    service: AdminNotificationService = Depends(get_notification_service),
):
    max_notifications = payload.max_notifications if payload else 100
    note = payload.note if payload else None
    return service.run_notification_ack_scan(
        max_notifications=max_notifications,
        note=note,
        origin="manual",
    )


@router.get("/models", response_model=list[ModelArtifactSummaryRead])
def list_models(
    _: object = Depends(require_admin),
    service: ModelService = Depends(get_model_service),
):
    return service.list_models()


@router.get("/models/{version}", response_model=ModelArtifactDetailRead)
def get_model(
    version: str,
    _: object = Depends(require_admin),
    service: ModelService = Depends(get_model_service),
):
    return service.get_model(version)


@router.post("/runs/trigger", response_model=TriggerRunResponse)
def trigger_run(
    payload: TriggerRunRequest | None = None,
    _: object = Depends(require_admin),
    service: RunService = Depends(get_run_service),
):
    note = payload.note if payload else None
    return service.trigger_run(note=note)


@router.post("/explanations/trigger", response_model=RefreshExplanationResponse)
def refresh_explanations(
    payload: RefreshExplanationRequest | None = None,
    _: object = Depends(require_admin),
    service: RunService = Depends(get_run_service),
):
    run_id = payload.run_id if payload else None
    return service.refresh_explanations(run_id=run_id)


@router.get("/jobs", response_model=list[JobExecutionRead])
def list_jobs(
    _: object = Depends(require_admin),
    service: RunService = Depends(get_run_service),
):
    return service.list_jobs()


@router.get("/scheduler/status", response_model=SchedulerStatusRead)
def scheduler_status(
    _: object = Depends(require_admin),
    service: RunService = Depends(get_run_service),
):
    scheduler = BackendScheduler()
    status = scheduler.status()
    jobs = service.list_jobs()
    last_job = jobs[0] if jobs else None
    return SchedulerStatusRead(
        enabled=status["enabled"],
        running=status["running"],
        timezone=status["timezone"],
        executionMode=status["execution_mode"],
        schedulerSources=status["scheduler_sources"],
        ingestionIntervalMinutes=status["ingestion_interval_minutes"],
        predictionIntervalMinutes=status["prediction_interval_minutes"],
        operationalPipelineIntervalMinutes=status[
            "operational_pipeline_interval_minutes"
        ],
        trainingReleaseSlaMonitorEnabled=status["training_release_sla_monitor_enabled"],
        trainingReleaseSlaIntervalMinutes=status[
            "training_release_sla_interval_minutes"
        ],
        trainingReleaseReassignmentMonitorEnabled=status[
            "training_release_reassignment_monitor_enabled"
        ],
        trainingReleaseReassignmentIntervalMinutes=status[
            "training_release_reassignment_interval_minutes"
        ],
        trainingReleaseAutoReassignReviewer=status[
            "training_release_auto_reassign_reviewer"
        ],
        notificationAckMonitorEnabled=status["notification_ack_monitor_enabled"],
        notificationAckMonitorIntervalMinutes=status[
            "notification_ack_monitor_interval_minutes"
        ],
        notificationDeliveryRetryMonitorEnabled=status[
            "notification_delivery_retry_monitor_enabled"
        ],
        notificationDeliveryRetryIntervalMinutes=status[
            "notification_delivery_retry_interval_minutes"
        ],
        notificationDeliveryFailureMonitorEnabled=status[
            "notification_delivery_failure_monitor_enabled"
        ],
        notificationDeliveryFailureIntervalMinutes=status[
            "notification_delivery_failure_interval_minutes"
        ],
        modelMonitoringCycleEnabled=status["model_monitoring_cycle_enabled"],
        modelMonitoringIntervalMinutes=status["model_monitoring_interval_minutes"],
        modelMonitoringDriftTopErrorCount=status[
            "model_monitoring_drift_top_error_count"
        ],
        modelMonitoringShadowTopErrorCount=status[
            "model_monitoring_shadow_top_error_count"
        ],
        modelMonitoringShadowMaxCandidates=status[
            "model_monitoring_shadow_max_candidates"
        ],
        modelMonitoringAlertsEnabled=status["model_monitoring_alerts_enabled"],
        notificationModelMonitoringUsernames=status[
            "notification_model_monitoring_usernames"
        ],
        jobs=status["jobs"],
        lastJob=last_job,
    )

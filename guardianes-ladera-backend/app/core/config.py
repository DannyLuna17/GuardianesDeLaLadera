from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Guardianes de la Ladera Backend"
    app_env: str = "development"
    real_data_only: bool = True
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    api_v1_prefix: str = "/v1"
    database_url: str = "sqlite:///./guardianes_ladera.db"
    run_db_migrations_on_startup: bool = True
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]
    )
    log_level: str = "INFO"
    seed_demo_data: bool = False
    model_version: str = "trained-spatial-seed-v1"
    model_artifacts_path: str | None = None
    active_model_manifest_path: str | None = None
    training_datasets_path: str | None = None
    model_evaluations_path: str | None = None
    model_selection_runs_path: str | None = None
    model_drift_reports_path: str | None = None
    model_shadow_runs_path: str | None = None
    model_promotion_min_validation_rmse_improvement: float = 0.0
    model_promotion_min_validation_rows: int = 1
    model_promotion_require_labels_dataset: bool = False
    model_promotion_min_calibration_gain: float = 0.0
    model_promotion_require_nested_estimation: bool = False
    model_promotion_min_nested_outer_validation_rmse_improvement: float = 0.0
    model_promotion_min_nested_outer_selection_rate: float = 0.0
    model_promotion_require_nested_temporal_latest_win: bool = False
    model_promotion_min_nested_temporal_outer_bucket_count: int = 0
    model_promotion_min_nested_temporal_latest_validation_rmse_improvement: (
        float
    ) = 0.0
    model_promotion_nested_temporal_recent_window_size: int = 0
    model_promotion_min_nested_temporal_recent_win_rate: float = 0.0
    model_promotion_min_nested_temporal_recent_average_validation_rmse_improvement: (
        float
    ) = 0.0
    model_promotion_max_spatial_slice_validation_rmse_regression: float = -1.0
    model_promotion_max_temporal_slice_validation_rmse_regression: float = -1.0
    model_promotion_max_spatial_slice_regression_count: int = -1
    model_promotion_max_temporal_slice_regression_count: int = -1
    model_promotion_slice_regression_min_rows: int = 1
    model_promotion_stability_window_runs: int = 1
    model_promotion_required_consistent_wins: int = 1
    model_promotion_stability_require_same_dataset_family: bool = True
    model_promotion_stability_require_same_dataset_taxonomy: bool = False
    model_promotion_stability_require_same_evaluation_cohort: bool = False
    model_promotion_stability_max_time_window_gap_days: int = 365
    model_promotion_stability_max_cohort_distance: int = 1
    model_drift_warning_validation_rmse_increase: float = 0.05
    model_drift_critical_validation_rmse_increase: float = 0.15
    model_drift_warning_accuracy_drop: float = 0.05
    model_drift_critical_accuracy_drop: float = 0.15
    enable_model_monitoring_cycle: bool = False
    model_monitoring_interval_minutes: int = 30
    model_monitoring_drift_top_error_count: int = 10
    model_monitoring_shadow_top_error_count: int = 5
    model_monitoring_shadow_max_candidates: int = 4
    enable_model_monitoring_alerts: bool = True
    notification_model_monitoring_usernames: Annotated[list[str], NoDecode] = Field(
        default_factory=list
    )
    jwt_secret_key: str = "guardianes-ladera-dev-secret-change-me-2026"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    seed_admin_username: str = "admin"
    seed_admin_password: str = "guardianes-admin"
    seed_admin_role: str = "admin"
    enable_scheduler: bool = False
    scheduler_timezone: str = "UTC"
    scheduler_execution_mode: str = "pipeline"
    scheduler_sources: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["IDEAM", "SGC", "UNGRD"]
    )
    ingestion_job_interval_minutes: int = 15
    prediction_job_interval_minutes: int = 10
    operational_pipeline_interval_minutes: int = 10
    enable_training_release_sla_monitor: bool = True
    training_release_sla_interval_minutes: int = 5
    training_release_auto_escalation_level: int = 2
    enable_training_release_reassignment_monitor: bool = True
    training_release_reassignment_interval_minutes: int = 7
    training_release_auto_reassign_reviewer: str = "admin"
    training_release_auto_reassign_due_in_hours: int = 24
    notification_delivery_channels_info: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["in_app"]
    )
    notification_delivery_channels_warning: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["in_app"]
    )
    notification_delivery_channels_critical: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["in_app"]
    )
    notification_release_ops_usernames: Annotated[list[str], NoDecode] = Field(
        default_factory=list
    )
    notification_escalation_include_requester: bool = True
    notification_resolution_copy_assigned_reviewer: bool = True
    notification_reassignment_copy_previous_reviewer: bool = True
    notification_ack_reminder_channels_warning: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["in_app"]
    )
    notification_ack_reminder_channels_critical: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["in_app"]
    )
    notification_stub_fail_channels: Annotated[list[str], NoDecode] = Field(
        default_factory=list
    )
    notification_ack_deadline_hours_info: int = 24
    notification_ack_deadline_hours_warning: int = 12
    notification_ack_deadline_hours_critical: int = 4
    enable_notification_ack_monitor: bool = True
    notification_ack_monitor_interval_minutes: int = 6
    notification_ack_reminder_max_count: int = 1
    notification_ack_reminder_escalate_after_count: int = 2
    enable_notification_delivery_retry_monitor: bool = True
    notification_delivery_retry_interval_minutes: int = 8
    notification_delivery_retry_backoff_minutes: int = 5
    notification_delivery_retry_max_attempts_per_channel: int = 3
    notification_retryable_failure_classifications: Annotated[list[str], NoDecode] = (
        Field(
            default_factory=lambda: [
                "transient_provider_error",
                "rate_limited",
            ]
        )
    )
    enable_notification_delivery_failure_monitor: bool = True
    notification_delivery_failure_interval_minutes: int = 9
    notification_delivery_failure_alert_after_attempts: int = 2
    notification_delivery_failure_watch_usernames: Annotated[list[str], NoDecode] = (
        Field(default_factory=list)
    )
    ingestion_transport_default: str = "auto"
    ideam_transport: str = "auto"
    sgc_transport: str = "auto"
    ungrd_transport: str = "auto"
    ideam_base_url: str | None = None
    sgc_base_url: str | None = None
    ungrd_base_url: str | None = None
    ideam_auth_header_name: str | None = None
    ideam_auth_token: str | None = None
    ideam_auth_query_param: str | None = None
    sgc_auth_header_name: str | None = None
    sgc_auth_token: str | None = None
    sgc_auth_query_param: str | None = None
    ungrd_auth_header_name: str | None = None
    ungrd_auth_token: str | None = None
    ungrd_auth_query_param: str | None = None
    provider_request_timeout_seconds: float = 30.0
    provider_request_retry_attempts: int = 3
    provider_request_retry_backoff_seconds: float = 1.5
    provider_cache_path: str | None = None
    structural_catalog_bundle_path: str | None = None
    ideam_station_catalog_url: str = (
        "https://visualizador.ideam.gov.co/gisserver/rest/services/CNE/"
        "CatalogoNacionalEstaciones/MapServer/0/query"
    )
    ideam_station_search_radius_degrees: float = 0.9
    ideam_station_limit_per_municipality: int = 3
    ideam_history_days: int = 7
    ideam_cache_max_age_minutes: int = 720
    sgc_page_size: int = 1000
    sgc_max_pages: int = 40
    sgc_cache_max_age_minutes: int = 180
    ungrd_page_size: int = 1000
    ungrd_max_pages: int = 10
    ungrd_cache_max_age_minutes: int = 180
    enable_llm_explanations: bool = False
    openai_api_key: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, list):
            return value
        if not value:
            return []
        return [item.strip() for item in value.split(",") if item.strip()]

    @field_validator("scheduler_sources", mode="before")
    @classmethod
    def parse_scheduler_sources(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if not value:
            return ["IDEAM", "SGC", "UNGRD"]
        return [item.strip() for item in value.split(",") if item.strip()]

    @field_validator(
        "notification_delivery_channels_info",
        "notification_delivery_channels_warning",
        "notification_delivery_channels_critical",
        "notification_release_ops_usernames",
        "notification_model_monitoring_usernames",
        "notification_ack_reminder_channels_warning",
        "notification_ack_reminder_channels_critical",
        "notification_retryable_failure_classifications",
        "notification_delivery_failure_watch_usernames",
        mode="before",
    )
    @classmethod
    def parse_notification_delivery_channels(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if not value:
            return ["in_app"]
        return [item.strip() for item in value.split(",") if item.strip()]

    @field_validator("notification_stub_fail_channels", mode="before")
    @classmethod
    def parse_notification_stub_fail_channels(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if not value:
            return []
        return [item.strip() for item in value.split(",") if item.strip()]

    @field_validator("scheduler_execution_mode")
    @classmethod
    def validate_scheduler_execution_mode(cls, value: str) -> str:
        normalized = value.lower().strip()
        allowed = {"split", "pipeline"}
        if normalized not in allowed:
            allowed_display = ", ".join(sorted(allowed))
            raise ValueError(
                f"Scheduler execution mode must be one of: {allowed_display}"
            )
        return normalized

    @field_validator(
        "ingestion_transport_default",
        "ideam_transport",
        "sgc_transport",
        "ungrd_transport",
    )
    @classmethod
    def validate_ingestion_transport(cls, value: str) -> str:
        normalized = value.lower().strip()
        allowed = {"seed", "http", "auto"}
        if normalized not in allowed:
            allowed_display = ", ".join(sorted(allowed))
            raise ValueError(f"Ingestion transport must be one of: {allowed_display}")
        return normalized

    @property
    def backend_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    @property
    def seed_data_path(self) -> Path:
        return self.backend_root / "app" / "data" / "frontend_seed.json"

    @property
    def resolved_model_artifacts_path(self) -> Path:
        if self.model_artifacts_path:
            return Path(self.model_artifacts_path).resolve()
        return self.backend_root / "app" / "ml" / "artifacts"

    @property
    def resolved_active_model_manifest_path(self) -> Path:
        if self.active_model_manifest_path:
            return Path(self.active_model_manifest_path).resolve()
        return self.resolved_model_artifacts_path / "active-model.json"

    @property
    def resolved_training_datasets_path(self) -> Path:
        if self.training_datasets_path:
            return Path(self.training_datasets_path).resolve()
        return self.backend_root / "app" / "ml" / "training_datasets"

    @property
    def resolved_model_evaluations_path(self) -> Path:
        if self.model_evaluations_path:
            return Path(self.model_evaluations_path).resolve()
        return self.backend_root / "app" / "ml" / "evaluations"

    @property
    def resolved_model_selection_runs_path(self) -> Path:
        if self.model_selection_runs_path:
            return Path(self.model_selection_runs_path).resolve()
        return self.backend_root / "app" / "ml" / "selection_runs"

    @property
    def resolved_model_drift_reports_path(self) -> Path:
        if self.model_drift_reports_path:
            return Path(self.model_drift_reports_path).resolve()
        return self.backend_root / "app" / "ml" / "drift_reports"

    @property
    def resolved_model_shadow_runs_path(self) -> Path:
        if self.model_shadow_runs_path:
            return Path(self.model_shadow_runs_path).resolve()
        return self.backend_root / "app" / "ml" / "shadow_runs"

    @property
    def resolved_provider_cache_path(self) -> Path:
        if self.provider_cache_path:
            return Path(self.provider_cache_path).resolve()
        return self.backend_root / ".provider_cache"

    @property
    def resolved_structural_catalog_bundle_path(self) -> Path:
        if self.structural_catalog_bundle_path:
            return Path(self.structural_catalog_bundle_path).resolve()
        return (
            self.backend_root
            / "app"
            / "data"
            / "official-structural"
            / "official_structural_bundle.json"
        )

    def transport_for_source(self, source_id: str) -> str:
        mapping = {
            "IDEAM": self.ideam_transport,
            "SGC": self.sgc_transport,
            "UNGRD": self.ungrd_transport,
        }
        return mapping.get(source_id, self.ingestion_transport_default)

    def source_base_url(self, source_id: str) -> str | None:
        mapping = {
            "IDEAM": self.ideam_base_url,
            "SGC": self.sgc_base_url,
            "UNGRD": self.ungrd_base_url,
        }
        value = mapping.get(source_id)
        return value.strip() if value else None

    def source_auth_header_name(self, source_id: str) -> str | None:
        mapping = {
            "IDEAM": self.ideam_auth_header_name,
            "SGC": self.sgc_auth_header_name,
            "UNGRD": self.ungrd_auth_header_name,
        }
        value = mapping.get(source_id)
        return value.strip() if value else None

    def source_auth_token(self, source_id: str) -> str | None:
        mapping = {
            "IDEAM": self.ideam_auth_token,
            "SGC": self.sgc_auth_token,
            "UNGRD": self.ungrd_auth_token,
        }
        value = mapping.get(source_id)
        return value.strip() if value else None

    def source_auth_query_param(self, source_id: str) -> str | None:
        mapping = {
            "IDEAM": self.ideam_auth_query_param,
            "SGC": self.sgc_auth_query_param,
            "UNGRD": self.ungrd_auth_query_param,
        }
        value = mapping.get(source_id)
        return value.strip() if value else None

    @property
    def configured_notification_channels(self) -> list[str]:
        channels = [
            *self.notification_delivery_channels_info,
            *self.notification_delivery_channels_warning,
            *self.notification_delivery_channels_critical,
            *self.notification_ack_reminder_channels_warning,
            *self.notification_ack_reminder_channels_critical,
        ]
        deduplicated: list[str] = []
        for channel in channels:
            normalized = channel.strip()
            if normalized and normalized not in deduplicated:
                deduplicated.append(normalized)
        return deduplicated

    _DEV_JWT_SECRET = "guardianes-ladera-dev-secret-change-me-2026"
    _DEV_ADMIN_PASSWORD = "guardianes-admin"

    def validate_production_secrets(self) -> None:
        if self.app_env.lower() in {"development", "docker", "local"}:
            return
        violations: list[str] = []
        if self.jwt_secret_key == self._DEV_JWT_SECRET:
            violations.append(
                "JWT_SECRET_KEY is still the default development value — "
                "set a strong, unique secret via environment variable"
            )
        if self.seed_admin_password == self._DEV_ADMIN_PASSWORD:
            violations.append(
                "SEED_ADMIN_PASSWORD is still the default development value — "
                "set a strong password via environment variable"
            )
        if violations:
            raise RuntimeError(
                "Production secrets are not configured: " + "; ".join(violations) + "."
            )

    def validate_real_data_runtime(self) -> None:
        if not self.real_data_only:
            return

        violations: list[str] = []
        if self.seed_demo_data:
            violations.append("SEED_DEMO_DATA must remain disabled")

        seed_sources = [
            source_id
            for source_id in ("IDEAM", "SGC", "UNGRD")
            if self.transport_for_source(source_id) == "seed"
        ]
        if self.ingestion_transport_default == "seed":
            violations.append("INGESTION_TRANSPORT_DEFAULT cannot use seed transport")
        if seed_sources:
            violations.append(
                f"seed transport is still configured for: {', '.join(seed_sources)}"
            )

        stub_channels = [
            channel
            for channel in self.configured_notification_channels
            if channel.lower().endswith("_stub")
        ]
        if stub_channels:
            violations.append(
                "stub notification channels are still configured: "
                + ", ".join(stub_channels)
            )

        if self.notification_stub_fail_channels:
            violations.append("NOTIFICATION_STUB_FAIL_CHANNELS must be empty")

        stub_retry_failures = [
            classification
            for classification in self.notification_retryable_failure_classifications
            if "stub" in classification.lower()
        ]
        if stub_retry_failures:
            violations.append(
                "stub retry classifications are still configured: "
                + ", ".join(stub_retry_failures)
            )

        if violations:
            raise RuntimeError(
                "REAL_DATA_ONLY is enabled but the runtime is still configured with "
                + "; ".join(violations)
                + "."
            )


@lru_cache
def get_settings() -> Settings:
    return Settings()

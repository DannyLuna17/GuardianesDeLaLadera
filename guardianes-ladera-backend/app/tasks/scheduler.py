from __future__ import annotations

from apscheduler.schedulers.background import BackgroundScheduler

from app.core.config import get_settings
from app.tasks.jobs import (
    run_ingestion_cycle,
    run_model_monitoring_cycle,
    run_notification_ack_cycle,
    run_notification_delivery_failure_cycle,
    run_notification_delivery_retry_cycle,
    run_operational_cycle,
    run_prediction_cycle,
    run_training_release_reassignment_cycle,
    run_training_release_sla_cycle,
)


class BackendScheduler:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.scheduler = BackgroundScheduler(timezone=self.settings.scheduler_timezone)
        self._configured = False

    def _configure_split_mode(self) -> None:
        self.scheduler.add_job(
            run_ingestion_cycle,
            "interval",
            minutes=self.settings.ingestion_job_interval_minutes,
            id="ingestion_cycle",
            replace_existing=True,
            kwargs={
                "sources": self.settings.scheduler_sources,
                "note": "Automated scheduler ingestion sync.",
            },
        )
        self.scheduler.add_job(
            run_prediction_cycle,
            "interval",
            minutes=self.settings.prediction_job_interval_minutes,
            id="prediction_cycle",
            replace_existing=True,
            kwargs={"note": "Automated scheduler prediction run."},
        )

    def _configure_pipeline_mode(self) -> None:
        self.scheduler.add_job(
            run_operational_cycle,
            "interval",
            minutes=self.settings.operational_pipeline_interval_minutes,
            id="operational_pipeline_cycle",
            replace_existing=True,
            kwargs={
                "sources": self.settings.scheduler_sources,
                "note": "Automated scheduler operational pipeline.",
            },
        )

    def _configure_release_sla_monitor(self) -> None:
        if not self.settings.enable_training_release_sla_monitor:
            return
        self.scheduler.add_job(
            run_training_release_sla_cycle,
            "interval",
            minutes=self.settings.training_release_sla_interval_minutes,
            id="training_release_sla_cycle",
            replace_existing=True,
            kwargs={"note": "Automated scheduler release SLA scan."},
        )

    def _configure_release_reassignment_monitor(self) -> None:
        if not self.settings.enable_training_release_reassignment_monitor:
            return
        self.scheduler.add_job(
            run_training_release_reassignment_cycle,
            "interval",
            minutes=self.settings.training_release_reassignment_interval_minutes,
            id="training_release_reassignment_cycle",
            replace_existing=True,
            kwargs={"note": "Automated scheduler release reassignment scan."},
        )

    def _configure_notification_ack_monitor(self) -> None:
        if not self.settings.enable_notification_ack_monitor:
            return
        self.scheduler.add_job(
            run_notification_ack_cycle,
            "interval",
            minutes=self.settings.notification_ack_monitor_interval_minutes,
            id="notification_ack_deadline_cycle",
            replace_existing=True,
            kwargs={"note": "Automated scheduler notification ack scan."},
        )

    def _configure_notification_delivery_retry_monitor(self) -> None:
        if not self.settings.enable_notification_delivery_retry_monitor:
            return
        self.scheduler.add_job(
            run_notification_delivery_retry_cycle,
            "interval",
            minutes=self.settings.notification_delivery_retry_interval_minutes,
            id="notification_delivery_retry_cycle",
            replace_existing=True,
            kwargs={"note": "Automated scheduler notification delivery retry scan."},
        )

    def _configure_notification_delivery_failure_monitor(self) -> None:
        if not self.settings.enable_notification_delivery_failure_monitor:
            return
        self.scheduler.add_job(
            run_notification_delivery_failure_cycle,
            "interval",
            minutes=self.settings.notification_delivery_failure_interval_minutes,
            id="notification_delivery_failure_cycle",
            replace_existing=True,
            kwargs={"note": "Automated scheduler notification delivery failure scan."},
        )

    def _configure_model_monitoring_cycle(self) -> None:
        if not self.settings.enable_model_monitoring_cycle:
            return
        self.scheduler.add_job(
            run_model_monitoring_cycle,
            "interval",
            minutes=self.settings.model_monitoring_interval_minutes,
            id="model_monitoring_cycle",
            replace_existing=True,
            kwargs={"note": "Automated scheduler model monitoring cycle."},
        )

    def configure(self) -> None:
        if self._configured:
            return
        if self.settings.scheduler_execution_mode == "pipeline":
            self._configure_pipeline_mode()
        else:
            self._configure_split_mode()
        self._configure_release_sla_monitor()
        self._configure_release_reassignment_monitor()
        self._configure_notification_ack_monitor()
        self._configure_notification_delivery_retry_monitor()
        self._configure_notification_delivery_failure_monitor()
        self._configure_model_monitoring_cycle()
        self._configured = True

    def start(self) -> None:
        self.configure()
        if not self.scheduler.running:
            self.scheduler.start()

    def shutdown(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)

    def status(self) -> dict:
        self.configure()
        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append(
                {
                    "id": job.id,
                    "trigger": str(job.trigger),
                    "next_run_time": getattr(job, "next_run_time", None),
                }
            )
        return {
            "enabled": self.settings.enable_scheduler,
            "running": self.scheduler.running,
            "timezone": self.settings.scheduler_timezone,
            "execution_mode": self.settings.scheduler_execution_mode,
            "scheduler_sources": self.settings.scheduler_sources,
            "ingestion_interval_minutes": self.settings.ingestion_job_interval_minutes,
            "prediction_interval_minutes": self.settings.prediction_job_interval_minutes,
            "operational_pipeline_interval_minutes": self.settings.operational_pipeline_interval_minutes,
            "training_release_sla_monitor_enabled": self.settings.enable_training_release_sla_monitor,
            "training_release_sla_interval_minutes": self.settings.training_release_sla_interval_minutes,
            "training_release_reassignment_monitor_enabled": self.settings.enable_training_release_reassignment_monitor,
            "training_release_reassignment_interval_minutes": self.settings.training_release_reassignment_interval_minutes,
            "training_release_auto_reassign_reviewer": self.settings.training_release_auto_reassign_reviewer,
            "notification_ack_monitor_enabled": self.settings.enable_notification_ack_monitor,
            "notification_ack_monitor_interval_minutes": self.settings.notification_ack_monitor_interval_minutes,
            "notification_delivery_retry_monitor_enabled": self.settings.enable_notification_delivery_retry_monitor,
            "notification_delivery_retry_interval_minutes": self.settings.notification_delivery_retry_interval_minutes,
            "notification_delivery_failure_monitor_enabled": self.settings.enable_notification_delivery_failure_monitor,
            "notification_delivery_failure_interval_minutes": self.settings.notification_delivery_failure_interval_minutes,
            "model_monitoring_cycle_enabled": self.settings.enable_model_monitoring_cycle,
            "model_monitoring_interval_minutes": self.settings.model_monitoring_interval_minutes,
            "model_monitoring_drift_top_error_count": self.settings.model_monitoring_drift_top_error_count,
            "model_monitoring_shadow_top_error_count": self.settings.model_monitoring_shadow_top_error_count,
            "model_monitoring_shadow_max_candidates": self.settings.model_monitoring_shadow_max_candidates,
            "model_monitoring_alerts_enabled": self.settings.enable_model_monitoring_alerts,
            "notification_model_monitoring_usernames": self.settings.notification_model_monitoring_usernames,
            "jobs": jobs,
        }

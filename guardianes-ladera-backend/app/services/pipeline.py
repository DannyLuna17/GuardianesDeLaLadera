from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.models import JobExecution
from app.schemas.admin import JobExecutionRead, TriggerPipelineResponse
from app.services.ingestion import IngestionService
from app.services.runs import RunService


class PipelineService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.ingestion_service = IngestionService(session)
        self.run_service = RunService(session)

    @staticmethod
    def _job_read(job: JobExecutionRead | JobExecution) -> JobExecutionRead:
        if isinstance(job, JobExecutionRead):
            return job
        return JobExecutionRead(
            id=job.id,
            jobType=job.job_type,
            status=job.status,
            startedAt=job.started_at,
            completedAt=job.completed_at,
            details=job.details or {},
        )

    def trigger_full_pipeline(
        self,
        sources: list[str] | None = None,
        note: str | None = None,
        origin: str = "manual",
    ) -> TriggerPipelineResponse:
        started_at = datetime.now(timezone.utc).replace(microsecond=0)
        pipeline_job = JobExecution(
            job_type="pipeline_run",
            status="running",
            started_at=started_at,
            details={"sources": sources or ["IDEAM", "SGC", "UNGRD"], "origin": origin, "note": note},
        )
        self.session.add(pipeline_job)
        self.session.flush()

        try:
            ingestion_response = self.ingestion_service.sync_sources(
                source_ids=sources,
                origin=origin,
                note=note or "Pipeline ingestion stage.",
            )
            run_response = self.run_service.trigger_run(
                note=note or "Pipeline prediction stage.",
                origin=origin,
                generate_explanations=False,
            )
            explanation_response = self.run_service.refresh_explanations(
                run_id=run_response.run.id,
                origin=origin,
            )
        except Exception as exc:
            completed_at = datetime.now(timezone.utc).replace(microsecond=0)
            pipeline_job.status = "failed"
            pipeline_job.completed_at = completed_at
            pipeline_job.details = {
                **(pipeline_job.details or {}),
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
            self.session.commit()
            raise

        completed_at = datetime.now(timezone.utc).replace(microsecond=0)
        pipeline_job.status = "completed"
        pipeline_job.completed_at = completed_at
        pipeline_job.details = {
            "sources": sources or ["IDEAM", "SGC", "UNGRD"],
            "origin": origin,
            "note": note,
            "ingestion_job_id": ingestion_response.job.id,
            "prediction_job_id": run_response.job.id,
            "explanation_job_id": explanation_response.job.id,
            "run_id": run_response.run.id,
        }
        self.session.commit()
        self.session.refresh(pipeline_job)

        return TriggerPipelineResponse(
            job=self._job_read(pipeline_job),
            ingestion=ingestion_response,
            run=run_response,
            explanations=explanation_response,
        )

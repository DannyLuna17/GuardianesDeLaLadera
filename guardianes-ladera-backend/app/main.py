from __future__ import annotations

from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import router as api_router
from app.core.config import get_settings
from app.core.exceptions import ApiError, api_error_handler
from app.core.logging import configure_logging
from app.core.rate_limit import login_limiter
from app.db.bootstrap import init_database, seed_demo_data
from app.db.session import session_scope
from app.services.structural_catalog import ensure_real_data_structural_catalog

CACHEABLE_PATHS = {
    "/v1/dashboard/bootstrap",
    "/v1/dashboard/summary",
    "/v1/municipalities",
    "/v1/source-catalog",
    "/v1/source-status",
}


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    settings.validate_production_secrets()
    settings.validate_real_data_runtime()
    init_database()
    if settings.seed_demo_data:
        with session_scope() as session:
            seed_demo_data(session)
    with session_scope() as session:
        ensure_real_data_structural_catalog(session, for_api=False)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    login_limiter.reset()
    app = FastAPI(title=settings.app_name, lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Accept"],
    )

    @app.middleware("http")
    async def attach_request_id(request: Request, call_next):
        request.state.request_id = str(uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        if request.url.path in CACHEABLE_PATHS:
            response.headers["Cache-Control"] = "public, max-age=30, stale-while-revalidate=60"
        return response

    app.add_exception_handler(ApiError, api_error_handler)
    app.include_router(api_router)

    return app


app = create_app()

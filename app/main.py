"""FastAPI app: lifespan (model loading), routes and static HTML page."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from app.model_loader import ModelLoader, ModelNotReadyError
from app.schemas import Health, ModelInfo, PredictRequest, PredictResponse, Status
from app.settings import Settings, get_settings

STATIC_DIR = Path(__file__).parent / "static"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("app")


def create_app(settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loader = ModelLoader(settings or get_settings())
        app.state.loader = loader
        # Load in the background: the process answers /health immediately and
        # /ready stays 503 until an approved model can predict.
        task = asyncio.create_task(loader.load_with_retry())
        yield
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        logger.info("Shutdown complete")

    app = FastAPI(title="SMS spam detector", version="1.0.0", lifespan=lifespan)

    def loader_of(request: Request) -> ModelLoader:
        return request.app.state.loader

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health", response_model=Health)
    def health(request: Request) -> Health:
        """Liveness: the process is running (also reports the image's commit)."""
        return Health(status="ok", git_sha=loader_of(request).settings.git_sha)

    @app.get("/ready", response_model=Status, responses={503: {"model": Status}})
    def ready(request: Request):
        """Readiness: an approved model is loaded and a test prediction succeeds."""
        loader = loader_of(request)
        if loader.is_ready():
            return Status(status="ready", model_version=loader.current.version)
        return JSONResponse(
            status_code=503,
            content=Status(status="not ready", detail=loader.last_error).model_dump(),
        )

    @app.get("/model-info", response_model=ModelInfo, responses={503: {"model": ModelInfo}})
    def model_info(request: Request):
        loader = loader_of(request)
        current = loader.current
        if current is None:
            s = loader.settings
            empty = ModelInfo(
                model_name=s.model_name,
                alias=s.model_alias,
                version=None,
                run_id=None,
                data_version=None,
                loaded_at=None,
            )
            return JSONResponse(status_code=503, content=empty.model_dump())
        return ModelInfo(
            model_name=current.model_name,
            alias=current.alias,
            version=current.version,
            run_id=current.run_id,
            data_version=current.data_version,
            loaded_at=current.loaded_at,
        )

    @app.post("/predict", response_model=PredictResponse, responses={503: {"model": Status}})
    def predict(body: PredictRequest, request: Request):
        try:
            label, prob, current = loader_of(request).predict(body.text)
        except ModelNotReadyError as exc:
            return JSONResponse(
                status_code=503, content=Status(status="not ready", detail=str(exc)).model_dump()
            )
        return PredictResponse(
            label=label,
            spam_probability=prob,
            model_name=current.model_name,
            model_version=current.version,
        )

    return app


app = create_app()

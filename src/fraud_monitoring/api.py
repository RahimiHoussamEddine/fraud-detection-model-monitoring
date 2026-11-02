"""FastAPI adapter for the frozen fraud-ranking artifact."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from starlette.responses import JSONResponse

from fraud_monitoring.service import ArtifactValidationError, FraudScoringService

app = FastAPI(
    title="Fraud Review Ranking API",
    version="1.0.0",
    description=(
        "Returns a model ranking score for manual review prioritization. "
        "The score is not a calibrated probability or a definitive fraud finding."
    ),
)


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    features: dict[str, float]


@lru_cache(maxsize=4)
def _service_for(path: str) -> FraudScoringService:
    return FraudScoringService(path)


async def _service() -> FraudScoringService:
    return _service_for(os.environ.get("FRAUD_ARTIFACT_DIR", "artifacts/v1"))


ServiceDependency = Annotated[FraudScoringService, Depends(_service)]


@app.get("/health", tags=["operations"])
async def health(deep: bool = False) -> JSONResponse:
    try:
        service = await _service()
        smoke = service.smoke_check() if deep else None
    except (ArtifactValidationError, FileNotFoundError, ValueError, OSError):
        return JSONResponse(
            status_code=503,
            content={
                "status": "unavailable",
                "artifact_loaded": False,
                "message": "The validated model artifact is not available.",
            },
        )
    return JSONResponse(
        content={
            "status": "ok",
            "artifact_loaded": True,
            "artifact_version": service.manifest["artifact_version"],
            "selected_model": service.manifest["selected_model"],
            "smoke_checks": smoke,
        }
    )


@app.post("/v1/score", tags=["review"])
async def score(request: ScoreRequest, service: ServiceDependency) -> dict:
    try:
        value = service.score(request.features)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "ranking_score": value,
        "interpretation": (
            "Use only to prioritize manual review; this is not a calibrated probability."
        ),
    }

import logging
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware

from api.config import settings
from api.schemas import MetarDataInput
from api.htm_service import HTMOrchestrator
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MetarMind HTM Visualizer API",
    description="Backend API to feed SDR topologies to the Next.js frontend.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.CLIENT_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("Booting up HTM Orchestrator Singleton...")
_orchestrator = HTMOrchestrator()


def get_orchestrator() -> HTMOrchestrator:
    return _orchestrator


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/v1/encode")
async def encode_params(
    payload: MetarDataInput, orchestrator: HTMOrchestrator = Depends(get_orchestrator)
):
    try:
        response = orchestrator.encode(payload)

        return {"status": "success", "data": response}

    except Exception as e:
        logger.error(f"Failed to encode telemetry: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

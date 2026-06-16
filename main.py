import logging

from fastapi import FastAPI

from src.api.routes import router
from src.services.finbert_service import preload_models

app = FastAPI(title="MLPOS FinBERT API")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.on_event("startup")
def startup_preload_models() -> None:
    logger.info("Starting FinBERT API startup preload")
    try:
        preload_models()
    except Exception:
        logger.exception("FinBERT startup preload failed")
        raise
    logger.info("FinBERT startup preload completed successfully")


app.include_router(router)

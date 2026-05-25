from fastapi import FastAPI

from src.api.routes import router   


app = FastAPI(
    title="MLPOS FinBERT API"
)

app.include_router(router)

from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="RAG Automated Response")

app.include_router(router, prefix="/api")

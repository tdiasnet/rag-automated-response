from fastapi import APIRouter
from pydantic import BaseModel
from src.pipelines.rag_pipeline import RAGPipeline
from src.retriever.faiss_retriever import FaissRetriever

router = APIRouter()

retriever = FaissRetriever("data/index.faiss", "data/docs.pkl")
pipeline = RAGPipeline(retriever)

class QueryInput(BaseModel):
    query: str

@router.post("/ask")
def ask_question(data: QueryInput):
    results = pipeline.retrieve(data.query)
    return {"answers": results}

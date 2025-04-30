from sentence_transformers import SentenceTransformer
from src.retriever.faiss_retriever import FaissRetriever

class RAGPipeline:
    def __init__(self, retriever: FaissRetriever, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        self.retriever = retriever

    def retrieve(self, query: str, top_k: int = 5):
        query_embedding = self.encoder.encode(query)
        return self.retriever.search(query_embedding, top_k)

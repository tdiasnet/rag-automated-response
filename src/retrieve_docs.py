# retrieval.py — Retrieves relevant documents using GTR-T5 embeddings and cosine similarity
# retrieval.py — Recupera documentos relevantes usando embeddings do GTR-T5 e similaridade cosseno

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Knowledge base (can be expanded later with Azure integration)
# Base de conhecimento (pode ser expandida depois com integração à Azure)
DOCUMENTS = [
    "RAG combines document retrieval with generative models to improve answer quality.",
    "Transformers like BERT are good for NLP tasks but lack retrieval mechanisms.",
    "RAG uses models like DPR to retrieve documents and generate contextual answers.",
    "Models like GPT-4 can be paired with retrieval to boost response accuracy.",
    "RAG systems are ideal for large and frequently updated knowledge bases."
]

# Load the Sentence Transformer model
# Carrega o modelo Sentence Transformer
model = SentenceTransformer("sentence-transformers/gtr-t5-large")

def retrieve_docs(question: str, top_k: int = 1) -> str:
    # Encode all documents and the question
    # Codifica todos os documentos e a pergunta
    doc_embeddings = model.encode(DOCUMENTS)
    question_embedding = model.encode([question])

    # Calculate cosine similarity and return top-k documents
    # Calcula a similaridade cosseno e retorna os top-k documentos
    similarities = cosine_similarity(question_embedding, doc_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]

    return " ".join([DOCUMENTS[i] for i in top_indices])

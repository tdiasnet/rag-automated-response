import faiss
import numpy as np

def retrieve_docs(query, index, top_k=5):
    """Recupera os documentos mais relevantes para uma consulta"""
    query_vector = query_to_vector(query)  # Função para converter a consulta em vetor
    distances, indices = index.search(query_vector, top_k)
    return indices

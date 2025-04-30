import faiss
import numpy as np
import pickle

class FaissRetriever:
    def __init__(self, index_path: str, store_path: str):
        self.index = faiss.read_index(index_path)
        with open(store_path, "rb") as f:
            self.docs = pickle.load(f)

    def search(self, query_embedding, top_k=5):
        D, I = self.index.search(np.array([query_embedding]), top_k)
        return [self.docs[i] for i in I[0]]

import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

docs = ["Doc1: AI is cool.", "Doc2: Python is great.", "Doc3: FAISS is fast."]
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = encoder.encode(docs)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

faiss.write_index(index, "data/index.faiss")
with open("data/docs.pkl", "wb") as f:
    pickle.dump(docs, f)

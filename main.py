import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Carregar o modelo pré-treinado para gerar embeddings
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Função para gerar embeddings
def generate_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.numpy()

# Exemplos de documentos
documents = [
    "A agricultura de precisão usa tecnologias para otimizar o uso de recursos.",
    "Máquinas pesadas como tratores são essenciais para a construção.",
    "O SAP S/4HANA é um sistema de gestão empresarial integrado."
]

# Gerar embeddings para os documentos
document_embeddings = generate_embeddings(documents)

# Construir o índice FAISS
index = faiss.IndexFlatL2(document_embeddings.shape[1])
index.add(np.array(document_embeddings))

# Função para buscar documentos
def search(query, k=1):
    query_embedding = generate_embeddings([query])
    _, indices = index.search(query_embedding, k)
    return [documents[i] for i in indices[0]]

# retrieval.py — Recupera documentos relevantes usando sentence-transformers
# retrieval.py — Retrieves relevant documents using sentence-transformers

from sentence_transformers import SentenceTransformer, util

# Documentos da base de conhecimento simulada
DOCUMENTS = [
    "O RAG combina recuperação de documentos com modelos generativos para gerar respostas melhores.",
    "Transformers como BERT são úteis para tarefas de NLP, mas não possuem mecanismo de recuperação.",
    "A arquitetura RAG usa modelos como DPR para buscar documentos e depois gerar respostas com base neles.",
    "Modelos como GPT-4 podem ser combinados com técnicas de recuperação de informações para melhorar a precisão.",
    "Sistemas RAG são úteis quando a base de conhecimento é grande e constantemente atualizada."
]

# Carrega modelo treinado para similaridade semântica
# Load a pretrained model for semantic similarity
model = SentenceTransformer('all-MiniLM-L6-v2')  # Rápido e eficiente

# Função para recuperar os documentos mais similares
# Function to retrieve the most similar documents
def retrieve_docs(question: str, top_k: int = 1) -> str:
    # Codifica documentos e pergunta
    # Encode the documents and the input question
    doc_embeddings = model.encode(DOCUMENTS, convert_to_tensor=True)
    question_embedding = model.encode(question, convert_to_tensor=True)

    # Calcula a similaridade entre a pergunta e todos os documentos
    # Compute the similarity between the question and all documents
    similarities = util.cos_sim(question_embedding, doc_embeddings)[0]

    # Seleciona os índices dos documentos mais similares
    # Select the most similar documents' indices
    top_indices = similarities.argsort(descending=True)[:top_k]

    # Retorna os documentos mais similares concatenados
    # Return the top-k similar documents concatenated
    return " ".join([DOCUMENTS[i] for i in top_indices])

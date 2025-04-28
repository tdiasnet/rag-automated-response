from transformers import RagTokenForGeneration, RagTokenizer

def load_model():
    """Carrega o modelo RAG pré-treinado"""
    model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq")
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
    return model, tokenizer

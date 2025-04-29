# Load the RAG model and tokenizer from Hugging Face / Carrega o modelo RAG e o tokenizador do Hugging Face
from transformers import RagTokenForGeneration, RagTokenizer

def load_model():
    """
    Load RAG model and tokenizer from pretrained weights
    Carrega o modelo RAG e o tokenizador a partir de pesos pré-treinados
    """
    print("🔍 Loading RAG model and tokenizer...")
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")
    model = RagTokenForGeneration.from_pretrained("facebook/rag-token-base")
    return model, tokenizer

# model.py — Loads the SentenceTransformer model (GTR-T5-Large) and ensures GPU usage if available.
# model.py — Carrega o modelo SentenceTransformer (GTR-T5-Large) e garante uso de GPU se disponível.

from sentence_transformers import SentenceTransformer
import torch

# Define model name
# Define o nome do modelo
MODEL_NAME = 'sentence-transformers/gtr-t5-large'

# Check if GPU is available
# Verifica se a GPU está disponível
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the model to the appropriate device
# Carrega o modelo para o dispositivo apropriado
print(f"🔍 Loading model: {MODEL_NAME}")
print(f"💻 Using device: {device.upper()}")

model = SentenceTransformer(MODEL_NAME, device=device)

def embed_text(texts):
    """
    🇺🇸 Encodes a list of texts into dense embeddings using the GTR-T5 model.
    🇧🇷 Codifica uma lista de textos em embeddings densos usando o modelo GTR-T5.
    
    Args:
        texts (List[str]): List of input texts.
        
    Returns:
        np.ndarray: Matrix of embeddings.
    """
    return model.encode(texts, convert_to_tensor=True)

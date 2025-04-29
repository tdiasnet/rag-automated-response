# Clean input text or documents before using them in the pipeline
# Limpa texto ou documentos antes de usá-los no pipeline

import re

def clean_text(text):
    """
    Apply basic text preprocessing: remove extra spaces and special characters.
    Aplica pré-processamento básico de texto: remove espaços extras e caracteres especiais.
    """
    text = re.sub(r"\s+", " ", text)  # Remove multiple spaces / Remove espaços múltiplos
    text = re.sub(r"[^\w\s.,?!-]", "", text)  # Remove unwanted characters / Remove caracteres indesejados
    return text.strip()

import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

def clean_text(text):
    """Limpa o texto removendo stopwords e caracteres especiais."""
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    stop_words = set(stopwords.words('english'))
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

import logging

def setup_logging():
    """Configura o logger para o projeto"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

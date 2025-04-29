import os

# Configura o diretório de cache local
os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.getcwd(), ".cache_hf")
os.environ["HF_HOME"] = os.path.join(os.getcwd(), ".cache_hf")
os.environ["TORCH_HOME"] = os.path.join(os.getcwd(), ".cache_hf")

# Imports locais
from model import load_model
from inference import generate_response
from retrieval import retrieve_docs  # Ainda será integrado futuramente
from preprocess import clean_text     # Ainda será integrado futuramente
from prompt_engineering import create_prompt


def main():
    print("🔧 Carregando modelo...")
    try:
        model, tokenizer = load_model()
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        return

    # Exemplo de pergunta
    question = "Como o RAG pode ser usado?"
    print(f"❓ Pergunta: {question}")

    # Simulando contexto (substituir depois pelo retrieve_docs)
    context = "O RAG usa recuperação de informações para melhorar as respostas geradas."
    print(f"📄 Contexto simulado: {context}")

    # Monta o prompt
    prompt = create_prompt(question, context)
    print(f"🧠 Prompt gerado:\n{prompt}")

    # Gera resposta
    try:
        response = generate_response(prompt, model, tokenizer)
        print(f"\n✅ Resposta:\n{response}")
    except Exception as e:
        print(f"Erro durante inferência: {e}")


if __name__ == "__main__":
    main()

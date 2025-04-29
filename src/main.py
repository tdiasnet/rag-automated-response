# Project entry point / Ponto de entrada do projeto
# This script loads the model, simulates retrieval, and generates an answer / 
# Este script carrega o modelo, simula a recuperação e gera uma resposta

import os

# Set local cache to keep model data within the project / 
# Define o cache local para manter os dados do modelo dentro do projeto
os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.getcwd(), ".cache_hf")
os.environ["HF_HOME"] = os.path.join(os.getcwd(), ".cache_hf")
os.environ["TORCH_HOME"] = os.path.join(os.getcwd(), ".cache_hf")

# Internal modules (logic separation) / Módulos internos (separação da lógica)
from model import load_model
from inference import generate_response
from retrieval import retrieve_docs  # Placeholder for future implementation
from preprocess import clean_text    # A ser integrado
from prompt_engineering import create_prompt


def main():
    # Step 1: Load model and tokenizer / Etapa 1: Carregar modelo e tokenizador
    print("🔧 Loading model / Carregando modelo...")
    model, tokenizer = load_model()

    # Step 2: Define user question / Etapa 2: Definir pergunta do usuário
    question = "Como o RAG pode ser usado?"
    print(f"❓ User question / Pergunta: {question}")

    # Step 3: Simulate document retrieval / Etapa 3: Simular recuperação de contexto
    context = "O RAG usa recuperação de informações para melhorar as respostas geradas."
    print(f"📄 Context (simulated) / Contexto (simulado): {context}")

    # Step 4: Build prompt with question and context / Etapa 4: Criar prompt com pergunta e contexto
    prompt = create_prompt(question, context)
    print(f"🧠 Prompt:\n{prompt}")

    # Step 5: Generate answer using the model / Etapa 5: Gerar resposta com o modelo
    response = generate_response(prompt, model, tokenizer)
    print(f"\n✅ Response / Resposta:\n{response}")


if __name__ == "__main__":
    main()

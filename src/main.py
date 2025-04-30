# src/main.py

# Import local modules
# Importa os módulos locais
from model import load_model
from inference import generate_response
from retrieval import retrieve_docs
from preprocess import clean_text
from prompt_engineering import create_prompt

# Import system modules
# Importa módulos do sistema
import os

# Set cache directory for Hugging Face and PyTorch
# Define o diretório de cache para Hugging Face e PyTorch
cache_dir = os.path.join(os.getcwd(), ".cache_hf")
os.environ["HF_HOME"] = cache_dir   # Cache for Hugging Face models and datasets
os.environ["TORCH_HOME"] = cache_dir  # Cache for PyTorch model files

# Main entry point of the application
# Ponto de entrada principal da aplicação
def main():
    # Load pretrained model and tokenizer
    # Carrega o modelo pré-treinado e o tokenizador
    model, tokenizer = load_model()

    # Example user question
    # Exemplo de pergunta do usuário
    question = "Como o RAG pode ser usado?"

    # Retrieve relevant documents based on the question (simulated for now)
    # Recupera documentos relevantes com base na pergunta (simulado por enquanto)
    context = retrieve_docs(question)

    # Create prompt to feed the model
    # Cria o prompt para alimentar o modelo
    prompt = create_prompt(question, context)

    # Generate answer from the model using the prompt
    # Gera uma resposta do modelo utilizando o prompt
    response = generate_response(prompt, model, tokenizer)

    # Display the generated response
    # Exibe a resposta gerada
    print(f"Resposta: {response}")

# Run the main function when the script is executed
# Executa a função principal quando o script é executado
if __name__ == "__main__":
    main()

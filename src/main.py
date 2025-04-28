from model import load_model
from inference import generate_response
from retrieval import retrieve_docs
from preprocess import clean_text
from prompt_engineering import create_prompt

def main():
    model, tokenizer = load_model()
    question = "Como o RAG pode ser usado?"
    
    # Simulando recuperação de documentos
    context = "O RAG usa recuperação de informações para melhorar as respostas geradas."
    prompt = create_prompt(question, context)
    
    # Gerando resposta
    response = generate_response(prompt, model, tokenizer)
    
    print(f"Resposta: {response}")

if __name__ == "__main__":
    main()

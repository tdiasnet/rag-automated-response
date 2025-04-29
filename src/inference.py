# Run inference using the RAG model and tokenizer
# Executa a inferência usando o modelo RAG e o tokenizador

import torch

def generate_response(prompt, model, tokenizer):
    """
    Generate response from model based on input prompt
    Gera uma resposta do modelo com base no prompt de entrada
    """
    print("🧮 Tokenizing prompt... / Tokenizando o prompt...")
    inputs = tokenizer(prompt, return_tensors="pt")

    print("🤖 Generating response... / Gerando resposta...")
    with torch.no_grad():
        output_ids = model.generate(**inputs)

    response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    return response

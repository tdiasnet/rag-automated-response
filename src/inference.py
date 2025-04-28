from model import load_model

def generate_response(question, model, tokenizer):
    """Gera uma resposta para uma pergunta usando o modelo RAG"""
    inputs = tokenizer(question, return_tensors="pt")
    outputs = model.generate(input_ids=inputs["input_ids"])
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

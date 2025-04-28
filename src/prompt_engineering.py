def create_prompt(question, context):
    """Cria um prompt estruturado para o modelo RAG"""
    prompt = f"Pergunta: {question}\nContexto: {context}\nResposta:"
    return prompt

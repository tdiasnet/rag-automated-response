# Build the prompt from user question and retrieved context
# Constrói o prompt a partir da pergunta do usuário e do contexto recuperado

def create_prompt(question, context):
    """
    Create a prompt combining context and user question.
    Cria um prompt combinando contexto e pergunta do usuário.
    """
    prompt = f"Contexto:\n{context}\n\nPergunta:\n{question}\n\nResposta:"
    return prompt

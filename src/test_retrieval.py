from retrieval import retrieve_docs

pergunta = "Como o RAG pode melhorar sistemas de busca?"
resposta = retrieve_docs(pergunta, top_k=2)
print("Documentos relevantes:\n", resposta)

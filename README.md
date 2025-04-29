# Sistema de Resposta Automatizada com RAG (Retrieval-Augmented Generation)

Este projeto é um sistema de resposta automática baseado em arquitetura RAG (Retrieval-Augmented Generation). Ele combina **busca vetorial** e **modelos de linguagem (LLMs)** para fornecer respostas mais precisas e contextualizadas a partir de uma base de dados própria.

## Objetivos
- Integrar embeddings e FAISS para recuperação eficiente de documentos.
- Utilizar LLMs para gerar respostas baseadas nas informações recuperadas.
- Implementar boas práticas de engenharia de software: ambientes isolados, versionamento e cache controlado.

## Tecnologias e Bibliotecas
- Python 3.12+
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [Sentence Transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [PyTorch](https://pytorch.org/)
- [Datasets (Hugging Face)](https://huggingface.co/docs/datasets/)

## Estrutura do Projeto
```bash
├── app/
│   ├── __init__.py
│   ├── load_data.py
│   ├── embed_data.py
│   ├── build_index.py
│   ├── retrieval.py
│   ├── generation.py
│   └── main.py
├── requirements.txt
├── README.md
├── .gitignore
├── venv/
└── .cache_hf/

# 🤖 RAG-Automated-Response
### Context-Aware Question Answering with Retrieval-Augmented Generation (RAG)  
### Respostas Automáticas com Geração Aumentada por Recuperação de Contexto (RAG)

---

## 🧠 Project Summary / Resumo do Projeto

### EN:
This project demonstrates how to build a simple question answering system using a Retrieval-Augmented Generation (RAG) architecture. It integrates traditional search (retrieval) with generative AI (RAG model) to generate more accurate and context-aware responses.

### PT-BR:
Este projeto demonstra como construir um sistema simples de perguntas e respostas usando a arquitetura de Geração Aumentada por Recuperação (RAG). Ele integra busca tradicional (retrieval) com IA generativa (modelo RAG) para gerar respostas mais precisas e com contexto.

---

## 🛠️ Technologies Used / Tecnologias Utilizadas

### EN:
- **Core ML and NLP Libraries**: `torch`, `transformers`, `sentence-transformers`, `scikit-learn`, `numpy`, `pandas`
- **Retrieval and Similarity**: `faiss-cpu` (or `faiss-gpu` for GPU support)
- **Web Framework**: `FastAPI` and `uvicorn` for building scalable APIs
- **Other**: `MLflow`, `Redis`, `PostgreSQL`, `huggingface-hub`, `safetensors`

### PT-BR:
- **Bibliotecas principais de ML e PLN**: `torch`, `transformers`, `sentence-transformers`, `scikit-learn`, `numpy`, `pandas`
- **Recuperação e Similaridade**: `faiss-cpu` (ou `faiss-gpu` para suporte à GPU)
- **Framework Web**: `FastAPI` e `uvicorn` para construir APIs escaláveis
- **Outras**: `MLflow`, `Redis`, `PostgreSQL`, `huggingface-hub`, `safetensors`

---

## 🗂️ Project Structure / Estrutura do Projeto

### EN:
- **app/**: Contains the FastAPI application for the project.
- **src/**: Source code for model training, data processing, and utility functions.
- **data/**: Datasets and data files used for model training.
- **models/**: Pretrained and custom-trained models.
- **notebooks/**: Jupyter notebooks for experimentation and prototyping.

### PT-BR:
- **app/**: Contém a aplicação FastAPI para o projeto.
- **src/**: Código fonte para treinamento de modelos, processamento de dados e funções utilitárias.
- **data/**: Conjuntos de dados e arquivos de dados usados para treinamento de modelos.
- **models/**: Modelos pré-treinados e treinados sob demanda.
- **notebooks/**: Notebooks Jupyter para experimentação e prototipagem.

---

## 📦 Installation / Instalação

### EN:
1. Clone the repository:
    ```bash
    git clone <repo_url>
    cd <project_directory>
    ```

2. Create a virtual environment:
    ```bash
    python -m venv .venv
    ```

3. Activate the virtual environment:
    - **Windows**:
        ```bash
        .venv\Scripts\activate
        ```
    - **Linux/macOS**:
        ```bash
        source .venv/bin/activate
        ```

4. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### PT-BR:
1. Clone o repositório:
    ```bash
    git clone <repo_url>
    cd <diretório_do_projeto>
    ```

2. Crie um ambiente virtual:
    ```bash
    python -m venv .venv
    ```

3. Ative o ambiente virtual:
    - **Windows**:
        ```bash
        .venv\Scripts\activate
        ```
    - **Linux/macOS**:
        ```bash
        source .venv/bin/activate
        ```

4. Instale as dependências necessárias:
    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 Running the API / Executando a API

### EN:
To run the FastAPI application, use the following command:
```bash
uvicorn app.main:app --reload


## 📁 Folder Structure / Estrutura de Pastas


```bash
├── app/                       # FastAPI application for serving the API / Aplicação FastAPI para servir a API
│   ├── __init__.py
│   ├── main.py                # FastAPI main app file / Arquivo principal da aplicação FastAPI
│   ├── api/                   # API routes and endpoints / Rotas e endpoints da API
│   │   └── v1/                # Version 1 of the API / Versão 1 da API
│   │       └── endpoints.py   # Endpoints for question answering and other features / Endpoints para perguntas e respostas
│   └── models/                # Pydantic models for request/response validation / Modelos Pydantic para validação de requisições/respostas
│       └── qa_model.py        # Model for the Question Answering data / Modelo para os dados de Pergunta e Resposta
│   
├── src/                       # Source code for training, data processing, and utilities / Código fonte para treinamento, processamento de dados e utilitários
│   ├── __init__.py
│   ├── load_data.py           # Script for loading datasets / Script para carregar os conjuntos de dados
│   ├── embed_data.py          # Script for text embedding generation / Script para geração de embeddings de texto
│   ├── build_index.py         # FAISS index building script / Script para construção do índice FAISS
│   ├── retrieval.py           # Retrieval system for question answering / Sistema de recuperação para perguntas e respostas
│   ├── generation.py          # Model for generating answers using RAG / Modelo para gerar respostas usando RAG
│   └── utils.py               # Utility functions for the project / Funções utilitárias para o projeto
│   
├── data/                      # Data folder for storing datasets and processed files / Pasta de dados para armazenar conjuntos de dados e arquivos processados
│   ├── raw/                   # Raw datasets / Conjuntos de dados brutos
│   └── processed/             # Processed datasets / Conjuntos de dados processados

├── models/                     # Pretrained and custom models / Modelos pré-treinados e modelos personalizados
│   └── rag_model/             # Trained RAG model / Modelo RAG treinado
│   
├── notebooks/                  # Jupyter notebooks for experimentation and prototyping / Notebooks Jupyter para experimentação e prototipagem
│   └── exploration.ipynb      # Exploration of the models and data / Exploração dos modelos e dados
│   
├── requirements.txt            # List of dependencies / Lista de dependências
├── README.md                   # Project documentation / Documentação do projeto
├── .gitignore                  # Git ignore file / Arquivo .gitignore
├── venv/                       # Virtual environment folder / Pasta do ambiente virtual
└── .cache_hf/                  # Hugging Face cache directory / Diretório de cache do Hugging Face


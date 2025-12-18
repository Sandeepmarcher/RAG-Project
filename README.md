# RAG-Project

# Document-Based Question Answering using RAG

This project implements a Retrieval-Augmented Generation (RAG) system that allows users to ask natural language questions over documents and receive accurate, context-aware answers.

## Tech Stack
- Python
- Flask
- LangChain
- FAISS
- OpenAI GPT
- Sentence Transformers

## Features
- PDF document ingestion
- Semantic search using vector embeddings
- Context-grounded LLM responses
- Reduced hallucination
- Scalable document retrieval

## How It Works
1. Documents are split into chunks
2. Chunks converted to embeddings
3. Stored in FAISS vector database
4. User query retrieves relevant chunks
5. LLM generates answer using retrieved context

## Run Instructions
```bash
pip install -r requirements.txt
python app.py

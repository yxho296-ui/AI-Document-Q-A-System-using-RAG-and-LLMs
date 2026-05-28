# Retrieval-Augmented Generation (RAG) Question Answering System

# Overview

This project is a Retrieval-Augmented Generation (RAG) system that allows users to ask questions based on the content of a PDF document. It combines semantic search using vector embeddings with a Large Language Model (LLM) to generate context-aware responses.

The system retrieves the most relevant text chunks from a document and uses them as context for the LLM to generate accurate answers.

# Features

- Load and process PDF documents
- Split documents into smaller text chunks
- Convert text into embeddings using HuggingFace models
- Store and search embeddings using FAISS vector database
- Perform semantic search to retrieve relevant context
- Generate responses using LLM APIs (Groq / OpenAI)

# How It Works

1. Load PDF document
2. Split document into smaller chunks
3. Convert chunks into vector embeddings
4. Store embeddings in FAISS index
5. Convert user query into embeddings
6. Retrieve top relevant chunks using similarity search
7. Send retrieved context to LLM
8. Generate final answer based on context

# Tech Stack

- Python
- LangChain
- FAISS (Vector Database)
- HuggingFace Sentence Transformers
- Groq / OpenAI API
- PyPDF


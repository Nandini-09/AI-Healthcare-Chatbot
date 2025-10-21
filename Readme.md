#  Medical Chatbot using LLaMA and GEMINI Models

This project implements an end-to-end Retrieval-Augmented Generation (RAG) system for a medical chatbot. It utilizes the open-source **LLaMA 2 7B Chat** model, leverages a **Pinecone** vector database or a **FAISS** local store for efficient document retrieval, and is built on the **LangChain** framework. The project includes a full evaluation workflow with metrics like **Cosine Similarity, BLEU, and ROUGE**, and a user-friendly frontend built with **Streamlit** .

## Project Structure


* **`llama.ipynb`**: Contains the full RAG pipeline development, including:
    * Setup and dependencies (`pinecone`, `langchain`, `ctransformers`).
    * PDF data loading and text chunking.
    * HuggingFace Embeddings (`sentence-transformers/all-MiniLM-L6-v2`) for vector creation.
    * Pinecone index initialization and vector storage.
    * Loading the **LLaMA 2 7B Chat** GGUF model via `CTransformers`.
    * Setting up the `RetrievalQA` chain.
    * A custom evaluation framework that calculates **Cosine Similarity, BLEU, ROUGE-1, and ROUGE-L** scores against a set of reference answers.
* **`app.py`**: The Streamlit application for the chatbot frontend.
    * Uses **Google GenerativeAI Embeddings** and the **Gemini-Pro** model.
    * Persists the vector store locally using **FAISS**.
    * Features a custom conversational prompt for a "highly knowledgeable and empathetic healthcare assistant."
    * Includes a function to calculate and display the quality metrics (**Cosine Similarity, BLEU, ROUGE**) in the live chat response.
* **`data/`**: Directory for the medical PDF documents (e.g., `Medical_book.pdf`).
* **`model/`**: Directory to store the local LLaMA 2 model file (e.g., `llama-2-7b-chat.ggmlv3.q4_0.bin`).


### 1. Prerequisites

You will need a working Python environment (Python 3.10+ recommended).

### 2. Dependencies

Install the required Python packages.


# 🧠 RAG Application with Ollama + ChromaDB

This project is a **Retrieval-Augmented Generation (RAG) application** built with:
- [Ollama](https://ollama.ai/) → Runs local LLMs (e.g., Llama3).
- [ChromaDB](https://www.trychroma.com/) → Vector database to store document embeddings.
- [LangChain](https://www.langchain.com/) → Orchestration layer (loading docs, creating embeddings, retrieval).
- Python (for glue code).

It lets you:
1. Load documents into a vector database.
2. Ask questions in natural language.
3. Get answers grounded in your documents.

With this setup, you can load your documents, embed them, and query them with a local LLM (like Llama 3), without relying on cloud services.

---

## 🚀 Features

- Fully local RAG pipeline (no external APIs).
- Supports custom embeddings (mxbai-embed-large).
- Uses ChromaDB to persist embeddings.
- Interactive Q&A in terminal (rag_chain.py).
- Configurable models and settings via .env.

---

## 📂 Project Structure

rag-app/
│── load_data.py      # Loads documents into ChromaDB with embeddings
│── rag_chain.py      # Interactive Q&A chain using retriever + LLM
│── requirements.txt  # Python dependencies
│── .env              # Configurations (models, db path, etc.)
│── chroma_data/      # Persistent vector DB (auto-created)
│── vector_search.py  # This script is a utility to directly test the Chroma vector database
│── README.md  
│── .gitignore  

---

## ⚙️ Setup Instructions

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/rag-app.git
cd rag-app
```

### 2. Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows
```

### 3. Install dependencies
```bash
pip3 install -r requirements.txt
```

### 4. Run Ollama
Make sure Ollama is installed and the required models are available:
```bash
ollama list
```

If not installed:
```bash
ollama pull llama3:8b
ollama pull mxbai-embed-large
```

Run Ollama server (in a separate terminal):
```bash
ollama serve
```

### 5. Environment Variables

Create a .env file in project root:
```env
# Model names (must match ollama list)

# Embedding & Chat Models
EMBEDDINGS_MODEL=mxbai-embed-large
CHAT_MODEL=llama3:8b

# Chroma vectorstore settings
CHROMA_DIR=./chroma_data
COLLECTION_NAME=docs

# Retrieval settings
TOP_K=5
```

### 6. Load Documents into ChromaDB
Run:
```bash
python3 load_data.py
```
This script will read your local documents (you can customize it) → embed them → store in ChromaDB.

### 7. Run RAG Q&A
Start the interactive chain:
```bash
python3 rag_chain.py
```
Example:
```
[User]: What is a vector store?
[Assistant]: A vector store or [vector database]...
```
Type exit to quit.

### 8. vector_search.py
This script is a utility to directly test the Chroma vector database.  
It performs a similarity search using the specified embedding model and prints out the most relevant documents with their similarity scores.  

Use this script if you want to:
- Debug embeddings stored in Chroma
- Verify that documents are being embedded and retrieved correctly
- Run quick experiments without invoking the full RAG pipeline

## 🛠️ Tech Stack

- [Ollama](https://ollama.ai/) → Runs open-source LLMs locally.
- [ChromaDB](https://www.trychroma.com/) → Stores embeddings for retrieval.
- [LangChain](https://www.langchain.com/) → Glue code for embeddings → retriever → LLM chain.
- Python → Core language for orchestration.

## 📌 Notes

- This project was inspired by a [Microsoft blog.](https://devblogs.microsoft.com/cosmosdb/build-a-rag-application-with-langchain-and-local-llms-powered-by-ollama/)
- I extended/customized it to learn how RAG works in practice by replacing cosmos vector db with chroma db
- You can replace ChromaDB with CosmosDB or Pinecone if you want a cloud vector DB.
- Data lives in chroma_data/ — delete this folder to reset the DB.


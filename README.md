# PDF-RAG-Assistant

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![RAG](https://img.shields.io/badge/RAG-Enabled-success)
![License](https://img.shields.io/badge/License-MIT-green)

**PDF-RAG-Assistant** is an intelligent document understanding system that enables users to query PDF documents using natural language.

The system is built using a Retrieval-Augmented Generation (RAG) pipeline, combining semantic search (FAISS), cross-encoder reranking, and a local LLM (Ollama) to generate accurate and context-grounded answers.

---

## Project Highlights

- End-to-end RAG pipeline for document question answering  
- Semantic retrieval using FAISS vector database  
- Cross-encoder reranking for improved answer relevance  
- Local LLM inference using Ollama (`llama3:8b`)  
- Context-aware question answering with hallucination control  
- Document summarization capability  
- Optimized with caching and session state  

---

## Table of Contents

- System Architecture  
- Retrieval Pipeline  
- Repository Structure  
- Design Insights  
- Known Limitations  
- Installation  
- Running the Application  

---

## System Architecture

PDF → Text Extraction → Chunking → Embeddings → FAISS Index  
User Query → Query Embedding → Similarity Search → Re-ranking → LLM → Answer

---

## Retrieval Pipeline

### Stage 1 — Dense Retrieval (FAISS)
- Text chunks are converted into embeddings using SentenceTransformers  
- Stored in a FAISS vector index  
- Query is embedded and top-K similar chunks are retrieved  

### Stage 2 — Cross-Encoder Re-ranking
- Retrieved chunks are re-ranked using a CrossEncoder model  
- Produces higher-quality, semantically aligned context  

### Final Step — LLM Generation
- Context is passed to a local LLM (Ollama)  
- Model generates an answer grounded in retrieved context  

---

## Repository Structure

PDF-RAG-Assistant  
│  
├── app.py  
├── requirements.txt  
├── README.md  
└── .gitignore  

---

## Design Insights

- FAISS ensures fast similarity search  
- CrossEncoder improves precision  
- Context truncation prevents overflow  
- Session state avoids recomputation  

---

## Known Limitations

- Requires Ollama running locally  
- Long context may be truncated  
- Scanned PDFs may fail  
- Performance depends on hardware  

---

## Installation

pip install -r requirements.txt

---

## Running the Application

streamlit run app.py

Open in browser:
http://localhost:8501

---

## Usage

1. Upload a PDF  
2. Wait for processing  
3. Ask questions  
4. View answers with sources  
5. Optionally summarize document  

---

## Author
Himanshu Dixit  

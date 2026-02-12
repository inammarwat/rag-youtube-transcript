# 🎥 YouTube RAG Pipeline with Gemini

A modular **Retrieval-Augmented Generation (RAG)** system that ingests YouTube video transcripts, performs semantic chunking and embedding, and enables question answering using **Google Gemini**.

This project demonstrates a complete end-to-end LLM pipeline including:

- Data ingestion  
- Text preprocessing & chunking  
- Embedding generation  
- Vector search (FAISS)  
- Retrieval-Augmented Generation (RAG)  

---

## 🚀 Project Overview

Large Language Models (LLMs) struggle with long-context documents and external knowledge integration.  
This project implements a **RAG pipeline** over YouTube transcripts to provide accurate, context-aware answers.

The system workflow:

1. Extracts transcripts from YouTube
2. Cleans and preprocesses text
3. Splits text into semantic chunks
4. Generates embeddings
5. Stores embeddings in FAISS vector store
6. Retrieves relevant chunks
7. Uses Gemini to generate context-aware answers

---

## 🏗️ Project Structure

```
youtube-rag-gemini/
│
├── notebooks/
│   ├── 01_data_ingestion.ipynb
│   ├── 02_chunking_analysis.ipynb
│   ├── 03_embedding_retrieval.ipynb
│   └── 04_rag_pipeline.ipynb
│
├── data/
│   └── transcripts/
│
├── main.py
├── requirements.txt
├── .env
└── README.md
```

---

## 🧠 System Architecture

```
YouTube Video
      ↓
Transcript Extraction
      ↓
Text Cleaning
      ↓
Chunking
      ↓
Embedding Generation
      ↓
FAISS Vector Store
      ↓
Retriever
      ↓
Gemini LLM
      ↓
Final Answer
```

---

## 🛠️ Tech Stack

- Python 3.10+
- Google Gemini API
- LangChain
- FAISS (Vector Database)
- youtube-transcript-api
- python-dotenv
- uv (package manager)

---


## 🧪 Example Usage

**Input Query:**
```
What is the main idea of the video?
```

**Output:**
```
The video explains ...
```

(Answer generated using retrieved transcript context.)

---

## 📊 Key Features

- Modular RAG pipeline
- Error-handled transcript ingestion
- Persistent transcript storage
- Semantic text chunking
- Vector similarity search
- Context-aware Gemini responses
- Reproducible project structure

---

## 🔐 Security Notes

- `.env` is excluded from version control
- API keys are never stored in source code
- Vector stores are not committed to GitHub

---

## 📌 Future Improvements

- Add Streamlit web interface
- Add RAG evaluation metrics
- Support multiple videos
- Hybrid retrieval (BM25 + embeddings)
- Convert into REST API
- Extend to Agentic RAG system

---

## 🎓 Academic Value

This project demonstrates practical understanding of:

- Retrieval-Augmented Generation (RAG)
- Vector databases
- Embedding models
- Prompt engineering
- LLM integration
- Modular pipeline design

It can be extended into research on:
- Agentic AI systems  
- Educational assistants  
- Multimodal RAG  
- Domain-specific knowledge systems  

---

## 👤 Author

**Inam Ullah Khan**  
MSc Data Science  
Al-Farabi Kazakh National University  

Research Interests:
- Retrieval-Augmented Generation
- Agentic AI Systems
- Generative AI
- Applied NLP

---

⭐ If you found this project useful, consider giving it a star!

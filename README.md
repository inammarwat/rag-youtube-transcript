# NeuralTranscript  
## A RAG-Based Semantic Search & Q&A System for YouTube Content

NeuralTranscript is an end-to-end Retrieval-Augmented Generation (RAG) system designed to perform semantic search and context-aware question answering over long-form YouTube transcripts.

The system integrates:

- Transcript ingestion
- Semantic chunking
- Vector embedding
- FAISS indexing
- Retrieval-based context injection
- Grounded response generation using Google Gemini

---

## 🚀 Project Motivation

Large Language Models (LLMs) struggle with long documents due to context window limitations and hallucination risks.

NeuralTranscript addresses this by:

- Converting transcripts into dense vector embeddings
- Performing similarity-based retrieval
- Injecting only relevant context into the LLM
- Generating grounded and reliable answers

---

## 🏗️ System Architecture

```
YouTube Video
      ↓
Transcript Extraction
      ↓
Semantic Chunking
      ↓
Embedding Generation
      ↓
FAISS Vector Index
      ↓
Retriever (Top-k Search)
      ↓
Context Injection
      ↓
Gemini LLM
      ↓
Final Answer
```

---

## 📂 Project Structure

```
NeuralTranscript/
│
├── notebooks/
│   ├── 01_data_ingestion.ipynb
│   ├── 02_semantic_chunking.ipynb
│   ├── 03_vector_indexing.ipynb
│   └── 04_rag_query_engine.ipynb
│
├── data/
│   └── transcripts/
│   └── chunked_docs.pkl
│   └── faiss_index/
│   └── index.faiss
│   └── index.pkl
│
├
├── main.py
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

- Python 3.10+
- LangChain
- FAISS
- Google Gemini API
- youtube-transcript-api
- python-dotenv
- uv

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

## ▶️ Running the Full Pipeline

### 1️⃣ Install Dependencies

```bash
uv venv
.venv\Scripts\activate
uv pip install -r requirements.txt
```

---

### 2️⃣ Add Gemini API Key

Create `.env` file:

```
GEMINI_API_KEY=your_api_key_here
```

---

### 3️⃣ Run Application

```bash
python main.py
```

---

## 🎓 Academic Contribution

This project demonstrates applied expertise in:

- Retrieval-Augmented Generation (RAG)
- Vector databases and semantic search
- Embedding-based knowledge indexing
- Prompt engineering
- Grounded LLM response generation

---

## 👤 Author

Engr. Inam Ullah Khan  
MSc Data Science  
Al-Farabi Kazakh National University  

Research Interests:
- Agentic AI Systems
- Retrieval-Augmented Generation
- Generative AI
- Applied NLP

---

⭐ If you found this project useful, consider giving it a star!

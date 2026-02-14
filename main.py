"""
NeuralTranscript - Main Application
RAG-based Semantic Search & Q&A for YouTube Content
"""

import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings # Import the embedding model
from langchain_google_genai import ChatGoogleGenerativeAI

# -----------------------------
# Load Environment Variables
# -----------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment.")

# -----------------------------
# Load Transcript
# -----------------------------
TRANSCRIPT_PATH = "data/transcripts/Gfr50f6ZBvo.txt"

if not os.path.exists(TRANSCRIPT_PATH):
    raise FileNotFoundError("Transcript file not found. Run ingestion first.")

with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
    transcript_text = f.read()

# -----------------------------
# Chunking
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_text(transcript_text)

print(f"Total chunks created: {len(chunks)}")

# -----------------------------
# Embedding Model
# -----------------------------
embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'} # Use 'cuda' if you have a GPU in Colab
)

# -----------------------------
# Vector Store
# -----------------------------
vectorstore = FAISS.from_texts(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -----------------------------
# LLM Initialization
# -----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    google_api_key=GEMINI_API_KEY
)

# -----------------------------
# Query Loop
# -----------------------------
print("\nNeuralTranscript RAG System Ready.")
print("Type 'exit' to quit.\n")

while True:
    query = input("Enter your question: ")

    if query.lower() == "exit":
        break

    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Use the following context to answer the question.
    If the answer is not in the context, say "I don't know."

    Context:
    {context}

    Question:
    {query}
    """

    response = llm.invoke(prompt)

    print("\nAnswer:")
    print(response.content)
    print("\n" + "-"*60 + "\n")

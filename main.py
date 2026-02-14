"""
PROJECT: NeuralTranscript: Semantic Search & Q&A for YouTube Content
MODULE: main.py (System Entry Point)
-------------------------------------------------------------------------
DESCRIPTION:
The master controller for the NeuralTranscript pipeline. It orchestrates 
four primary stages: Ingestion, Chunking, Indexing, and RAG Querying.

AUTHOR: Engr. Inam Ullah Khan
-------------------------------------------------------------------------
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Import our custom modules (ensure your .ipynb files are exported as .py)
# Or, if they remain in notebooks, we wrap the core logic in this file.
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# --- 1. SETUP LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 2. CONFIGURATION ---
load_dotenv()
INDEX_PATH = "data/faiss_index"

def run_pipeline(youtube_url: str, user_query: str):
    """
    Executes the full NeuralTranscript pipeline.
    """
    try:
        logger.info("🚀 Starting NeuralTranscript Pipeline...")

        # STEP 1: Ingestion & Chunking (Logic from Notebooks 01 & 02)
        # For a professional main.py, you would ideally have these in separate .py files
        # Here we assume the Vector Store (Notebook 03) is already built or we build it.
        
        if not os.path.exists(INDEX_PATH):
            logger.warning("⚠️ FAISS Index not found. Please run the Indexing Notebook first.")
            return

        # STEP 2: Load Vector Store
        logger.info("📂 Loading Semantic Index...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_db = FAISS.load_local(
            INDEX_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )

        # STEP 3: Build RAG Chain (Modern LCEL Standard)
        logger.info("🤖 Initializing Gemini 2.5 Flash RAG Chain...")
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
        
        prompt = ChatPromptTemplate.from_template("""
        You are an AI Research Assistant. Use the provided transcript context to answer the question accurately.
        If the answer isn't in the context, state that you don't know.
        
        CONTEXT: {context}
        QUESTION: {question}
        ANSWER:
        """)

        rag_chain = (
            {"context": vector_db.as_retriever(search_kwargs={"k": 5}), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # STEP 4: Execution
        logger.info(f"❓ Querying: {user_query}")
        response = rag_chain.invoke(user_query)
        
        print("\n" + "="*50)
        print(f"✨ NEURAL TRANSCRIPT RESPONSE:\n\n{response}")
        print("="*50 + "\n")

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")

# --- 3. MAIN ENTRY POINT ---
if __name__ == "__main__":
    # Example Usage
    video_url = "https://www.youtube.com/watch?v=-HzgcbRXUK8" # Lex Fridman & Demis Hassabis
    query = "What are the three stages of AI evolution according to Demis?"

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    
    run_pipeline(video_url, query)
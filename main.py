import os
from typing import List

from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# --------------------------------------------------
# Configuration
# --------------------------------------------------
VIDEO_ID = "Gfr50f6ZBvo"
VECTOR_DB_PATH = "vectorstore/faiss_index"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 4

# --------------------------------------------------
# Environment setup
# --------------------------------------------------
load_dotenv()


# --------------------------------------------------
# Data ingestion
# --------------------------------------------------
def load_youtube_transcript(video_id: str) -> str:
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return " ".join(segment["text"] for segment in transcript)


# --------------------------------------------------
# Chunking
# --------------------------------------------------
def chunk_text(text: str) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.create_documents([text])


# --------------------------------------------------
# Embeddings & vector store
# --------------------------------------------------
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def build_or_load_vectorstore(docs: List[Document]):
    embeddings = get_embeddings()

    if os.path.exists(VECTOR_DB_PATH):
        return FAISS.load_local(
            VECTOR_DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    vectorstore = FAISS.from_documents(docs, embeddings)
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)
    vectorstore.save_local(VECTOR_DB_PATH)
    return vectorstore


# --------------------------------------------------
# Gemini LLM
# --------------------------------------------------
def initialize_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.2
    )


# --------------------------------------------------
# RAG pipeline
# --------------------------------------------------
def build_rag_pipeline(vectorstore):
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": TOP_K}
    )

    llm = initialize_llm()

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )


# --------------------------------------------------
# Main execution
# --------------------------------------------------
def main():
    print("Loading transcript...")
    text = load_youtube_transcript(VIDEO_ID)

    print("Chunking text...")
    documents = chunk_text(text)

    print("Loading or building vector store...")
    vectorstore = build_or_load_vectorstore(documents)

    print("Initializing RAG pipeline...")
    qa_chain = build_rag_pipeline(vectorstore)

    print("\nRAG system ready. Ask a question (type 'exit' to quit).\n")

    while True:
        query = input("Question: ").strip()
        if query.lower() == "exit":
            break

        response = qa_chain(query)

        print("\nAnswer:\n")
        print(response["result"])

        print("\nSources:\n")
        for i, doc in enumerate(response["source_documents"], start=1):
            print(f"[Source {i}] {doc.page_content[:200]}...\n")


# --------------------------------------------------
# Entry point
# --------------------------------------------------
if __name__ == "__main__":
    main()

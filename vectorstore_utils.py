import os
import streamlit as st
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Retrieve OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def create_or_load_vectorstore(force_recreate=False):
    """
    Create or load a FAISS vector store from local PDFs or an existing index.
    """
    if not force_recreate and "vector_store" in st.session_state and st.session_state['vector_store']:
        return st.session_state['vector_store']

    vector_store = build_vectorstore_from_folder("data")
    return vector_store

def build_vectorstore_from_folder(data_folder: str):
    """
    Reads local PDFs from a folder, splits text, embeds with OpenAI, and stores in a FAISS index in memory.
    """
    if not OPENAI_API_KEY:
        st.error("Missing OPENAI_API_KEY in .env.")
        return None

    pdf_texts = []
    for filename in os.listdir(data_folder):
        if filename.endswith(".pdf"):
            path = os.path.join(data_folder, filename)
            try:
                with open(path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    text = "".join(page.extract_text() or "" for page in reader.pages)
                    pdf_texts.append(text)
            except Exception as e:
                st.warning(f"Error processing {filename}: {e}")

    if not pdf_texts:
        st.warning("No PDF files found or failed to extract text from PDFs.")
        return None

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = [chunk for txt in pdf_texts for chunk in text_splitter.split_text(txt)]

    # Create embeddings using the updated OpenAIEmbeddings class
    try:
        embeddings = OpenAIEmbeddings()  # Automatically uses the API key from the environment
        vector_store = FAISS.from_texts(docs, embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating embeddings or vector store: {e}")
        return None

def similarity_search_docs(vector_store, query, k=3):
    """
    Perform a similarity search on the vector store.
    """
    if vector_store is None:
        st.warning("Vector store is not available.")
        return []
    return vector_store.similarity_search(query, k=k)

"""
RAG Pipeline - 2026 Best Practices (Render Optimized)
Uses OpenAI embeddings for fast startup - no local model loading required
"""

import os
from typing import List, Dict, Optional, Tuple
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
import PyPDF2
from dotenv import load_dotenv

load_dotenv()

# Configuration
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "finance_docs"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


class RAGPipeline:
    """
    2026 RAG Pipeline with:
    - ChromaDB for persistent vector storage
    - OpenAI embeddings (fast, no local model loading)
    - Hybrid search with metadata filtering
    - Simple relevance scoring
    """
    
    def __init__(self):
        """Initialize RAG pipeline with ChromaDB and OpenAI embeddings"""
        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        
        # Use default embeddings (no API key required, works offline)
        # This avoids OpenAI quota issues and is free
        try:
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        except Exception as e:
            print(f"Embedding init error: {e}")
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def add_pdf(self, pdf_path: str, metadata: Optional[Dict] = None) -> int:
        """
        Process and add a PDF to the vector store
        
        Args:
            pdf_path: Path to PDF file
            metadata: Optional metadata to attach to chunks
            
        Returns:
            Number of chunks added
        """
        # Extract text from PDF
        text = ""
        try:
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text() or ""
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}"
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
            return 0
        
        if not text.strip():
            return 0
        
        # Split into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Prepare documents for ChromaDB
        filename = os.path.basename(pdf_path)
        base_metadata = metadata or {}
        base_metadata["source"] = filename
        
        ids = []
        documents = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{filename}_{i}"
            chunk_metadata = {**base_metadata, "chunk_index": i}
            
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append(chunk_metadata)
        
        # Add to collection (upsert to handle re-indexing)
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        
        return len(chunks)
    
    def add_text(self, text: str, source: str = "user_input", metadata: Optional[Dict] = None) -> int:
        """Add raw text to the vector store"""
        chunks = self.text_splitter.split_text(text)
        
        base_metadata = metadata or {}
        base_metadata["source"] = source
        
        ids = []
        documents = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{source}_{i}"
            chunk_metadata = {**base_metadata, "chunk_index": i}
            
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append(chunk_metadata)
        
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        
        return len(chunks)
    
    def hybrid_search(
        self,
        query: str,
        n_results: int = 10,
        where_filter: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Perform hybrid search with semantic similarity and optional metadata filtering
        
        Args:
            query: Search query
            n_results: Number of results to retrieve
            where_filter: Optional ChromaDB where filter for metadata
            
        Returns:
            List of search results with documents, metadata, and scores
        """
        # Semantic search
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        if not results["documents"][0]:
            return []
        
        # Format results
        search_results = []
        for i, doc in enumerate(results["documents"][0]):
            search_results.append({
                "document": doc,
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else 0,
                "score": 1 - results["distances"][0][i] if results["distances"] else 1
            })
        
        return search_results
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_reranking: bool = False,  # Disabled for faster performance
        where_filter: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Full retrieval pipeline: search -> return top results
        
        Args:
            query: User query
            top_k: Number of final results
            use_reranking: Whether to use reranking (disabled for Render)
            where_filter: Optional metadata filter
            
        Returns:
            List of retrieved documents
        """
        results = self.hybrid_search(query, n_results=top_k, where_filter=where_filter)
        
        if not results:
            return []
        
        # Sort by score (higher is better)
        results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
        
        return results[:top_k]
    
    def get_context_for_llm(
        self,
        query: str,
        top_k: int = 5,
        max_context_length: int = 3000
    ) -> Tuple[str, List[str]]:
        """
        Get formatted context for LLM prompt
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            max_context_length: Maximum context length in characters
            
        Returns:
            Tuple of (formatted_context, list_of_sources)
        """
        results = self.retrieve(query, top_k=top_k)
        
        if not results:
            return "", []
        
        # Build context with source tracking
        context_parts = []
        sources = set()
        current_length = 0
        
        for i, result in enumerate(results):
            doc = result["document"]
            source = result["metadata"].get("source", "Unknown")
            sources.add(source)
            
            # Check length limit
            if current_length + len(doc) > max_context_length:
                break
            
            context_parts.append(f"[Source: {source}]\n{doc}")
            current_length += len(doc)
        
        formatted_context = "\n\n---\n\n".join(context_parts)
        
        return formatted_context, list(sources)
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the vector store"""
        return {
            "total_documents": self.collection.count(),
            "collection_name": COLLECTION_NAME,
            "embedding_model": "text-embedding-3-small" if OPENAI_API_KEY else "default"
        }
    
    def index_data_folder(self, folder_path: str = "data") -> int:
        """Index all PDFs in the data folder"""
        total_chunks = 0
        
        if not os.path.exists(folder_path):
            return 0
        
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(".pdf"):
                pdf_path = os.path.join(folder_path, filename)
                chunks = self.add_pdf(pdf_path)
                total_chunks += chunks
                print(f"Indexed {filename}: {chunks} chunks")
        
        return total_chunks


# Singleton instance
_rag_pipeline = None


def get_rag_pipeline() -> RAGPipeline:
    """Get or create RAG pipeline singleton"""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline


def initialize_rag() -> Dict:
    """Initialize RAG and index documents if needed"""
    pipeline = get_rag_pipeline()
    stats = pipeline.get_collection_stats()
    
    # If collection is empty, index the data folder
    if stats["total_documents"] == 0:
        print("Indexing documents from data folder...")
        chunks = pipeline.index_data_folder("data")
        stats["indexed_chunks"] = chunks
        stats["total_documents"] = pipeline.collection.count()
    
    return stats

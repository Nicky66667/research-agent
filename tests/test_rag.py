import pytest
import tempfile # create temp files and dir
import os
from rag.chunker import RecursiveChunker
from rag.pipeline import RAGPipeline

def test_chunk():

    chunker = RecursiveChunker(chunk_size=100, overlap=20)
    text = "This is paragraph one.\n\nThis is paragraph two.\n\nThis is paragraph three."
    chunks = chunker.split(text)

    assert len(chunks) > 0 #ensure list is not empty
    assert all (isinstance(c,str) for c in chunks) # ensure all elements are strings
    print(f"Chunker test passed : {len(chunks)} chunks")

def test_rag_pipeline():
    # Initialize RAG pipeline
    rag = RAGPipeline(collection_name="test_collection")

    # Test metadata
    metadata = {
        "source_id": "test_pdf_001",
        "title": "AI RAG Research Note",
        "authors": "Test Researcher",
        "year": "2025"
    }

    # Test real PDF file
    pdf_path = "arXiv-sample-1.pdf"

    # Test add PDF
    new_chunks = rag.add_pdf(pdf_path, metadata)
    assert new_chunks > 0  # Should ingest chunks from real PDF

    # Test duplicate paper skip
    duplicate_chunks = rag.add_pdf(pdf_path, metadata)
    assert duplicate_chunks == 0

    # Test semantic query
    results = rag.query("chunk overlap context retrieval", top_k=3)

    # Assertions same style as your test_chunk
    assert rag.collection.count() > 0
    assert len(results) > 0
    assert all(isinstance(item, dict) for item in results)
    assert all("text" in item and "score" in item for item in results)
    assert all(0 <= item["score"] <= 1 for item in results)

    print(f"RAGPipeline test passed: {new_chunks} chunks ingested, {len(results)} results returned")


if __name__ == "__main__":
    test_rag_pipeline()
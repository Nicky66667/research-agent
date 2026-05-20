# Vector database for semantic search
import chromadb

# PDF text extraction tool
import pymupdf

from langchain_openai import OpenAIEmbeddings
from .chunker import RecursiveChunker
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL

class RAGPipeline:

    def __init__(self,collection_name:str = "research_papers"):
        #  ChromaDB in‑memory client for dev
        # Use PersistentClient(path=".chroma") for persistence
        self.client = chromadb.Client()

        # Collection = database table
        # hnsw:space=cosine: use cosine similarity for vector distance
        self.collection = self.client.get_or_create_collection(
            name = collection_name,
            metadata = {"hnsw:space":"cosine"}
        )

        # OpenAI Embedding model: converts text to 1536-dim vectors
        self.embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL)

        self.chunker = RecursiveChunker(chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

        # loaded paper IDs to avoid duplicate processing
        self.loaded_ids = set() # create an empty set

    def add_pdf(self, pdf_path: str, metadata:dict):
        """
        Parse PDF → split chunks → embed → store in ChromaDB
        Return number of new chunks added (0 if skipped)
        """
        paper_id = metadata.get("source_id", pdf_path) # get(key, default value) -> if no key exist, return pdf_path

        # check duplication: skip processed papers
        if paper_id in self.loaded_ids:
            return 0

        # step 1: use PyMuPDF to extract pdf text

        doc = pymupdf.open(pdf_path) # open pdf
        pages = [page.get_text() for page in doc] # extract page texts
        # Join pages with double newlines
        text = "\n\n".join(pages)

        # step 2: chunks

        chunks = self.chunker.split(text)
        if not chunks:
            return 0

        # step 3: Batch embed by one API call instead of loop

        embeddings = self.embedding.embed_documents(chunks)
        """
        .embed_documents() return 
        [
            [0.1, 0.2, ...],
            [0.3, 0.5, ...]
        ]
        """

        # step 4: Save to ChromaDB with metadata for source tracking

        ids = [f"{paper_id}_chunk_{i}" for i in range(len(chunks))] # Create unique IDs for each chunk, e.g. paper123_chunk_0
        metas = [{**metadata, "chunk_index":i} for i in range(len(chunks))] # Attach metadata + chunk index

        # store chunks, embeddings, and metadata into vector DB
        self.collection.add(
            documents = chunks,
            embeddings = embeddings,
            ids = ids,
            metadatas = metas
        )

        self.loaded_ids.add(paper_id) # Mark paper as already ingested
        print(f"[RAG] Ingested {len(chunks)} chunks from: {metadata.get('title', paper_id)}") # log ingestion result
        return len(chunks) # Return number of stored chunks

    def query(self, query:str, top_k:int = 5) -> list[dict]:
        """
        Retrieve most relevant chunks.
        Return list of dicts with metadata, sorted by similarity descending.
        """

        if self.collection.count()==0:
            return []

        # 1. Convert query to embedding, then find top_k nearest neighbors

        query_emb = self.embedding.embed_query(query) # Convert query text (from user) into embedding vector
        # Query vector database for similar documents
        results = self.collection.query(
            query_embeddings = [query_emb],
            n_results = min(top_k, self.collection.count()), # Limit number of returned results
            include = ["documents", "metadatas", "distances"] # Return text, metadata, and similarity scores
        )

        # Convert distance to similarity: 0 = identical, 2 = opposite
        # (1 - distance) maps to 0~1 similarity score

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        # Format search results into ‑readable answer list
        output = []

        for doc, meta, dist in zip(docs, metas, dists): # zip -> [(doc1, meta1, dist1),...]
            item = {
                "text": doc,
                "source": meta.get("source_id", "unknown"),
                "title": meta.get("title", ""),
                "authors": meta.get("authors", ""),
                "year": meta.get("year", ""),
                "score": round(1 - dist, 4)
            }
            output.append(item)

        return output
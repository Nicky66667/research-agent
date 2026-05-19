import os
from dotenv import load_dotenv

# Chunking by embeddings → Retrieve Top‑K results → Feed to GPT‑4o‑mini → Run agent reasoning for up to 10 steps

load_dotenv() # read .env

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]

# RAG variables
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 128
TOP_K = 5
EMBEDDING_MODEL = "text-embedding-3-small"

# Agent variables
LLM_MODEL = "gpt-4o-mini"
MAX_STEPS = 10
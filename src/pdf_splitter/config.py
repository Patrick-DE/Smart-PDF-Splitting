# src/pdf_splitter/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
PDF_PATH = "data/Dokument_2025-08-18_190506_ocred.pdf"
OUTPUT_DIR = "output_documents"
USE_MODEL = "ollama"  # "gemini" or "ollama"
OLLAMA_MODEL_NAME = "llama3.1:8b"
GEMINI_MODEL_NAME = "gemini-1.5-flash"

# --- MongoDB and Embedding Setup ---
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "pdf_splitter_memory"
COLLECTION_NAME = "split_decisions"
VECTOR_INDEX_NAME = "vector_index"
# src/pdf_splitter/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
PDF_PATH = "data/Dokument_2025-08-18_190506_ocred.pdf"
OUTPUT_DIR = "output_documents"
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_HOST = "http://127.0.0.1:11434"

# --- MongoDB and Embedding Setup ---
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "pdf_splitter_memory"
COLLECTION_NAME = "split_decisions"
VECTOR_INDEX_NAME = "vector_index"
# src/pdf_splitter/tools.py
import os
import re
from typing import List
from datetime import datetime
import json
from PyPDF2 import PdfReader, PdfWriter
from pymongo import MongoClient
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
import ollama
from . import config


# Minimal Ollama embeddings adapter exposing a small API compatible with
# the existing vectorstore usage (embed_documents, embed_query). Keeps the
# rest of the code unchanged and avoids depending on langchain_ollama.
class OllamaEmbeddingsAdapter:
    def __init__(self, client: ollama.Client, model: str):
        self.client = client
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        resp = self.client.embeddings(model=self.model, input=texts)
        embeddings: List[List[float]] = []
        for item in resp:
            if isinstance(item, dict) and "embedding" in item:
                embeddings.append(item["embedding"])
            else:
                embeddings.append(item)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        resp = self.client.embeddings(model=self.model, input=[text])
        first = resp[0]
        if isinstance(first, dict) and "embedding" in first:
            return first["embedding"]
        return first


# Initialize embeddings adapter with the Ollama client (host from config)
_OLLAMA_CLIENT = ollama.Client(host=config.OLLAMA_HOST)
EMBEDDINGS = OllamaEmbeddingsAdapter(_OLLAMA_CLIENT, model=getattr(config, "OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"))


# --- Tool Definitions ---
def read_consecutive_pages(current_page_index: int) -> str:
    """Read current and next page text from the configured PDF path."""
    try:
        print(f"[DEBUG] Reading PDF: {config.PDF_PATH}, pages: {current_page_index}, {current_page_index + 1}")
        reader = PdfReader(config.PDF_PATH)
        content = ""
        if 0 <= current_page_index < len(reader.pages):
            page_text = reader.pages[current_page_index].extract_text() or ""
            print(f"[DEBUG] Extracted text (page {current_page_index}): {page_text}")
            content += f"--- Page {current_page_index + 1} Content ---\n{page_text}\n\n"
        else:
            return f"Error: Current page index {current_page_index} is out of bounds."

        if current_page_index + 1 < len(reader.pages):
            next_page_text = reader.pages[current_page_index + 1].extract_text() or ""
            print(f"[DEBUG] Extracted text (page {current_page_index + 1}): {next_page_text}")
            content += f"--- Page {current_page_index + 2} Content ---\n{next_page_text}"
        else:
            content += "--- End of Document ---"
        return content
    except Exception as e:
        return f"Error reading PDF pages: {e}"


# --- Helper functions used for filename/date/company normalization ---
def normalize_date(d: str) -> str:
    if not d:
        return "unknown_date"
    d = d.strip()
    formats = ["%d.%m.%Y", "%d/%m/%Y", "%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%Y.%m.%d"]
    for fmt in formats:
        try:
            parsed = datetime.strptime(d, fmt)
            return parsed.strftime("%Y%m%d")
        except Exception:
            continue
    digits = re.sub(r"\D", "", d)
    if len(digits) >= 8:
        if digits.startswith("20") or digits.startswith("19"):
            return digits[:8]
        ddmmyyyy = digits[:8]
        return ddmmyyyy[4:8] + ddmmyyyy[2:4] + ddmmyyyy[0:2]
    return re.sub(r"[^0-9A-Za-z_-]", "", d)


def shorten_company(name: str) -> str:
    if not name:
        return "UnknownCompany"
    n = name.strip()
    # remove common legal suffixes
    n = re.sub(r"\bGmbH\b\.?", "", n, flags=re.IGNORECASE).strip()
    mapping = {
        "deutsche bahn": "DB",
        "deutsche bank": "DBank",
        "stadtwerke neu isenburg": "SWNI",
    }
    lower = n.lower()
    for k, v in mapping.items():
        if k in lower:
            return v
    parts = [p for p in re.split(r"[\s,\.-]+", n) if p]
    if len(parts) == 1:
        return parts[0]
    initials = "".join(p[0].upper() for p in parts[:3])
    return initials


def extract_metadata(current_page_text: str, next_page_text: str = "") -> str:
    """Call the local Ollama model to extract date/company/title and return normalized JSON."""
    try:
        system_prompt = (
            "You are a metadata extraction assistant. Given the text of a document "
            "(current page and next page), extract the document date, the company/sender, "
            "and the document title/subject. Return ONLY a single JSON object with keys: "
            "date, company, title. Date must be in YYYYMMDD if available; if not, return an empty string. "
            "Company should be a short identifier (examples: 'Deutsche Bahn' -> 'DB', "
            "'Deutsche Bank' -> 'DBank', 'Stadtwerke Neu Isenburg' -> 'SWNI'), and strip legal suffixes like GmbH. "
            "Title should be concise and short only a few words max (no explanation). Do not include any additional text."
        )

        user_content = "CURRENT PAGE:\n" + current_page_text + "\n\nNEXT PAGE:\n" + next_page_text

        resp = _OLLAMA_CLIENT.chat(
            model=getattr(config, "OLLAMA_MODEL"),
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}],
            stream=False,
        )
        message = resp.get("message", {})
        content = message.get("content") or ""

        # Try to parse content as JSON; if the model returned text around JSON,
        # extract the first JSON object substring.
        parsed = None
        try:
            parsed = json.loads(content)
        except Exception:
            m = re.search(r"\{[\s\S]*\}", content)
            if m:
                candidate = m.group(0)
                try:
                    parsed = json.loads(candidate)
                except Exception:
                    parsed = None

        if isinstance(parsed, dict):
            date_raw = parsed.get("date", "") or parsed.get("datum", "")
            company_raw = parsed.get("company", "") or parsed.get("sender", "")
            title_raw = parsed.get("title", "") or parsed.get("subject", "")

            normalized = {
                "date": normalize_date(date_raw),
                "company": shorten_company(company_raw),
                "title": (title_raw or "").strip()
            }
            return json.dumps(normalized)

        # If nothing parsed, return raw content as a string (caller should handle)
        return content
    except Exception as e:
        return json.dumps({"error": str(e)})


def search_for_similar_cases(current_page_text: str, next_page_text: str) -> str:
    """Search the MongoDB vectorstore for similar past cases."""
    if not config.MONGO_URI:
        return "MongoDB URI not configured. Cannot search for similar cases."
    try:
        client = MongoClient(config.MONGO_URI)
        collection = client[config.DB_NAME][config.COLLECTION_NAME]
        vector_store = MongoDBAtlasVectorSearch(
            collection=collection, embedding=EMBEDDINGS, index_name=config.VECTOR_INDEX_NAME
        )
        combined_query = f"Current Page:\n{current_page_text}\n\nNext Page:\n{next_page_text}"
        docs = vector_store.similarity_search_with_score(combined_query, k=1)
        if docs:
            doc, score = docs[0]
            past_decision = doc.metadata.get('decision', 'unknown')
            return f"Found a similar case with score {score:.2f}. The past human decision was: '{past_decision}'."
        return "No similar cases found."
    except Exception as e:
        return f"Failed to search for similar cases: {e}"


def ask_human_for_confirmation(question: str) -> str:
    """Prompt the human operator for a yes/no confirmation."""
    print("\n--- HUMAN-IN-THE-LOOP REQUIRED ---")
    print(f"Agent asks: {question}")
    while True:
        response = input("Is this the start of a new document? (yes/no): ").lower()
        if response in ["yes", "y"]:
            return "Human feedback: 'yes', this is a new document."
        if response in ["no", "n"]:
            return "Human feedback: 'no', this is not a new document."
        print("Invalid input. Please enter 'yes' or 'no'.")


def save_document(page_indices: List[int], metadata) -> str:
    """Save pages with a filename based on LLM-provided metadata (normalized)."""
    if not page_indices:
        return "Error: No page indices provided to save."
    try:
        md = {}
        if isinstance(metadata, str):
            try:
                md = json.loads(metadata)
            except Exception:
                md = {"title": metadata}
        elif isinstance(metadata, dict):
            md = metadata

        date_raw = md.get("date") or md.get("datum") or md.get("date_raw") or ""
        company_raw = md.get("company") or md.get("sender") or md.get("from") or ""
        title_raw = md.get("title") or md.get("subject") or md.get("heading") or ""

        norm_date = normalize_date(date_raw)
        short_company = shorten_company(company_raw)
        sanitized_company = re.sub(r'[\W_]+', '_', short_company)
        sanitized_title = re.sub(r'[\W_]+', '_', (title_raw or "").strip()) or "untitled"
        filename = f"{norm_date}-{sanitized_company}-{sanitized_title}.pdf"
        output_path = os.path.join(config.OUTPUT_DIR, filename)

        pdf_writer = PdfWriter()
        pdf_reader = PdfReader(config.PDF_PATH)
        for page_num in page_indices:
            pdf_writer.add_page(pdf_reader.pages[page_num])

        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        with open(output_path, "wb") as f:
            pdf_writer.write(f)

        return f"Successfully saved document to: {output_path}"
    except Exception as e:
        return f"Error saving PDF to {locals().get('output_path', '<unknown>')}: {e}"
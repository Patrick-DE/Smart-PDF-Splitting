# src/pdf_splitter/tools.py
import os
import re
from typing import List
from PyPDF2 import PdfReader, PdfWriter
from langchain.agents import tool
from pymongo import MongoClient
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_ollama import OllamaEmbeddings
from . import config

from . import config
# Initialize embeddings model
EMBEDDINGS = OllamaEmbeddings(model="nomic-embed-text")

# --- Tool Definitions ---
# Move all functions decorated with @tool here:
# - read_consecutive_pages
# - search_for_similar_cases
# - ask_human_for_confirmation
# - save_document

@tool
def read_consecutive_pages(current_page_index: int) -> str:
    """
    Reads and returns the text content of the current page and the next page from the PDF.

    Args:
        current_page_index (int): The index of the current page to read.

    Returns:
        str: Combined text content of the current and next page, or an error message if out of bounds.
    """
    content = ""
    try:
        reader = PdfReader(config.PDF_PATH)
        # Read current page
        if 0 <= current_page_index < len(reader.pages):
            content += f"--- Page {current_page_index + 1} Content ---\n{reader.pages[current_page_index].extract_text()}\n\n"
        else:
            return f"Error: Current page index {current_page_index} is out of bounds."

        # Read next page
        if current_page_index + 1 < len(reader.pages):
            content += f"--- Page {current_page_index + 2} Content ---\n{reader.pages[current_page_index + 1].extract_text()}"
        else:
            content += "--- End of Document ---"
        return content
    except Exception as e:
        return f"Error reading PDF pages: {e}"


@tool
def search_for_similar_cases(current_page_text: str, next_page_text: str) -> str:
    """
    Searches the vector store for similar past cases to aid in document splitting decisions.

    Args:
        current_page_text (str): Text content of the current page.
        next_page_text (str): Text content of the next page.

    Returns:
        str: Description of the most similar past case and its decision, or a message if none found.
    """
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

@tool
def ask_human_for_confirmation(question: str) -> str:
    """
    Asks the human user for a 'yes' or 'no' confirmation when the agent is unsure about a document split.

    Args:
        question (str): The question to present to the human user.

    Returns:
        str: Human feedback indicating whether this is a new document or not.
    """
    print("\n--- HUMAN-IN-THE-LOOP REQUIRED ---")
    print(f"Agent asks: {question}")
    while True:
        response = input("Is this the start of a new document? (yes/no): ").lower()
        if response in ["yes", "y"]:
            return "Human feedback: 'yes', this is a new document."
        if response in ["no", "n"]:
            return "Human feedback: 'no', this is not a new document."
        print("Invalid input. Please enter 'yes' or 'no'.")

@tool
def save_document(page_indices: List[int], company: str, date: str, title: str) -> str:
    """
    Saves the specified pages as a new PDF document with a dynamically generated name based on company, date, and title.

    Args:
        page_indices (List[int]): List of page indices to save as a new document.
        company (str): Name of the company or sender.
        date (str): Date associated with the document (e.g., letter date).
        title (str): Title or subject of the document.

    Returns:
        str: Success message with output path, or error message if saving fails.
    """
    if not page_indices:
        return "Error: No page indices provided to save."
    try:
        # Sanitize filename components
        sanitized_company = re.sub(r'[\W_]+', '_', company)
        sanitized_title = re.sub(r'[\W_]+', '_', title)
        filename = f"{date}-{sanitized_company}-{sanitized_title}.pdf"
        output_path = os.path.join(config.OUTPUT_DIR, filename)

        pdf_writer = PdfWriter()
        pdf_reader = PdfReader(config.PDF_PATH)
        for page_num in page_indices:
            pdf_writer.add_page(pdf_reader.pages[page_num])
        
        with open(output_path, "wb") as f:
            pdf_writer.write(f)
        
        return f"Successfully saved document to: {output_path}"
    except Exception as e:
        return f"Error saving PDF to {output_path}: {e}"
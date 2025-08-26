# src/pdf_splitter/main.py

import os
import sys
from PyPDF2 import PdfReader
from . import config
from .ollama_agent import OllamaPDFSplitterAgent

if __name__ == "__main__":
    # 1. Initial Setup
    print(f"Attempting to connect to Ollama at: {config.OLLAMA_HOST}")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(config.PDF_PATH):
        print(f"Error: PDF file not found at '{config.PDF_PATH}'. Please update the path.")
        sys.exit(1)
    if not config.MONGO_URI:
        print("Warning: MONGO_URI not set. The agent cannot learn from feedback.")

    try:
        # 2. PDF and Agent Setup
        reader = PdfReader(config.PDF_PATH)
        total_pages = len(reader.pages)
        print(f"Processing PDF with {total_pages} pages...")

        agent = OllamaPDFSplitterAgent(config)

        # 3. Initial State
        state = {
            "current_page_index": 0,
            "total_pages": total_pages,
            "current_document_pages": [0],
        }
        messages = [{"role": "user", "content": "Start processing the PDF. Please split it into logical documents."}]

        # 4. Run agent loop
        os.system('cls' if os.name == 'nt' else 'clear')
        while state["current_page_index"] < state["total_pages"]:
            result = agent.run(messages, state)
            messages = result["messages"]
            state = result["state"]
            print(f"Current Page Index: {state['current_page_index']}")
            print(f"Pages for current doc: {state['current_document_pages']}")
            print("Last message:", messages[-1]["content"])
            print("\n" + "="*50 + "\n")

        print("PDF splitting process complete!")
    except FileNotFoundError:
        print(f"Error: The file '{config.PDF_PATH}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

    except FileNotFoundError:
        print(f"Error: The file '{config.PDF_PATH}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

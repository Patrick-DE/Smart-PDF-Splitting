# src/pdf_splitter/ollama_agent.py
"""
PDF splitting agent using native Ollama Python library with tool/function calling (llama3.1:8b).
"""

import ollama
from typing import List, Dict, Any
from . import config
from .tools import read_consecutive_pages, search_for_similar_cases, ask_human_for_confirmation, save_document, extract_metadata
from .base_agent import BasePDFSplitterAgent

ollama_tools = [
    {
        "name": "read_consecutive_pages",
        "description": "Reads and returns the text content of the current page and the next page from the PDF.",
        "parameters": {
            "type": "object",
            "properties": {
                "current_page_index": {"type": "integer", "description": "The index of the current page to read."}
            },
            "required": ["current_page_index"]
        }
    },
    {
        "name": "search_for_similar_cases",
        "description": "Searches the vector store for similar past cases to aid in document splitting decisions.",
        "parameters": {
            "type": "object",
            "properties": {
                "current_page_text": {"type": "string"},
                "next_page_text": {"type": "string"}
            },
            "required": ["current_page_text", "next_page_text"]
        }
    },
    {
        "name": "ask_human_for_confirmation",
        "description": "Asks the human user for a 'yes' or 'no' confirmation when the agent is unsure about a document split.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string"}
            },
            "required": ["question"]
        }
    },
    {
        "name": "save_document",
        "description": "Saves the specified pages as a new PDF document. Accepts 'page_indices' and a 'metadata' object (JSON) returned by extract_metadata.",
        "parameters": {
            "type": "object",
            "properties": {
                "page_indices": {"type": "array", "items": {"type": "integer"}},
                "metadata": {"type": "object"}
            },
            "required": ["page_indices", "metadata"]
        }
    }
    ,
    {
        "name": "extract_metadata",
        "description": "Extracts date, company, and title metadata from the current and next page text. Returns a JSON string.",
        "parameters": {
            "type": "object",
            "properties": {
                "current_page_text": {"type": "string"},
                "next_page_text": {"type": "string"}
            },
            "required": ["current_page_text", "next_page_text"]
        }
    }
]

class OllamaPDFSplitterAgent(BasePDFSplitterAgent):
    def __init__(self, config):
        super().__init__(tools=ollama_tools, model_name=config.OLLAMA_MODEL, config=config)
        self.client = ollama.Client(host=config.OLLAMA_HOST)

    def run(self, messages: List[Dict[str, Any]], state: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self.build_prompt(state)
        # METADATA #
        # If we are at the start of a new document (no collected pages yet),
        # proactively extract metadata once so the model doesn't need to call
        # extract_metadata repeatedly for each page. Store result in state.
        # try:
        #     if not state.get("current_document_pages"):
        #         if not state.get("metadata_extracted_for_current_doc", False):
        #             idx = state.get("current_page_index", 0)
        #             combined = read_consecutive_pages(idx)
        #             # split the combined content produced by read_consecutive_pages
        #             parts = __import__("re").split(r"--- Page \d+ Content ---\n", combined)
        #             current_text = parts[1].strip() if len(parts) > 1 else combined
        #             next_text = parts[2].strip() if len(parts) > 2 else ""
        #             metadata_raw = extract_metadata(current_text, next_text)
        #             # append as a tool message so the LLM sees the extracted metadata
        #             messages.append({
        #                 "role": "tool",
        #                 "tool_call_id": "pre_extract",
        #                 "name": "extract_metadata",
        #                 "content": str(metadata_raw)
        #             })
        #             state["current_metadata"] = metadata_raw
        #             state["metadata_extracted_for_current_doc"] = True
        # except Exception as e:
        #     print(f"Pre-extraction failed: {e}")

        ollama_request = {
            "model": self.model_name,
            "messages": [{"role": "system", "content": prompt}] + messages,
            "tools": self.tools,
            "stream": False,
        }
        # Call Ollama with a small retry and robust error handling. Ollama's client
        # may raise an exception if the server returns invalid JSON (500 with
        # an empty body), so catch exceptions, log details, and return a safe
        # message to the controller instead of crashing.
        try:
            response = self.client.chat(**ollama_request)
        except Exception as e:
            # Attempt one quick retry in case of a transient server hiccup
            try:
                response = self.client.chat(**ollama_request)
            except Exception as e2:
                import traceback
                print("Ollama chat request failed on retry:", e2)
                traceback.print_exc()
                # Append a tool-like message so the agent loop can continue and
                # so the LLM has context about the failure.
                messages.append({
                    "role": "tool",
                    "tool_call_id": "ollama_error",
                    "name": "ollama_error",
                    "content": f"Ollama chat failed: {e2}"
                })
                return {"messages": messages, "state": state}
        tool_calls = response.get("message", {}).get("tool_calls") or []

        # Ensure messages list is properly updated
        response_message = response.get("message", {})
        if response_message:
            messages.append(response_message)

        for call in tool_calls:
            name = call.get("function", {}).get("name")
            args = call.get("function", {}).get("arguments", {})
            
            if not name:
                continue

            # Dynamically call the tool function by name. Tools are plain
            # Python functions (no LangChain @tool decorator) so we call them
            # directly with kwargs provided by the model.
            tool_function = globals().get(name)
            if tool_function:
                try:
                    # METADATA # If saving a document and model didn't supply metadata, inject previously extracted metadata
                    # if name == "save_document":
                    #     if isinstance(args, dict) and "metadata" not in args:
                    #         if state.get("current_metadata"):
                    #             args = dict(args)
                    #             args["metadata"] = state.get("current_metadata")

                    result = tool_function(**args)

                    # METADATA # If we just saved a document, reset metadata flags so next document is fresh
                    # if name == "save_document":
                    #     state["current_document_pages"] = []
                    #     state["metadata_extracted_for_current_doc"] = False
                    #     state["current_metadata"] = None

                    # Append tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.get("id"),
                        "name": name,
                        "content": str(result)
                    })
                except Exception as e:
                    error_message = f"Error calling tool {name}: {e}"
                    print(error_message)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.get("id"),
                        "name": name,
                        "content": error_message
                    })
            else:
                unknown_tool_message = f"Unknown tool: {name}"
                print(unknown_tool_message)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "name": name,
                    "content": unknown_tool_message
                })

        state = self.update_state(state, tool_calls)
        return {"messages": messages, "state": state}

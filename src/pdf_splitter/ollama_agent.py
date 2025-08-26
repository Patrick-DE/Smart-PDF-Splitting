# src/pdf_splitter/ollama_agent.py
"""
PDF splitting agent using native Ollama Python library with tool/function calling (llama3.1:8b).
"""

import ollama
from typing import List, Dict, Any
from . import config
from .tools import read_consecutive_pages, search_for_similar_cases, ask_human_for_confirmation, save_document
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
        "description": "Saves the specified pages as a new PDF document with a dynamically generated name based on company, date, and title.",
        "parameters": {
            "type": "object",
            "properties": {
                "page_indices": {"type": "array", "items": {"type": "integer"}},
                "company": {"type": "string"},
                "date": {"type": "string"},
                "title": {"type": "string"}
            },
            "required": ["page_indices", "company", "date", "title"]
        }
    }
]

class OllamaPDFSplitterAgent(BasePDFSplitterAgent):
    def __init__(self, config):
        super().__init__(tools=ollama_tools, model_name=config.OLLAMA_MODEL, config=config)
        self.client = ollama.Client(host=config.OLLAMA_HOST)

    def run(self, messages: List[Dict[str, Any]], state: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self.build_prompt(state)
        ollama_request = {
            "model": self.model_name,
            "messages": [{"role": "system", "content": prompt}] + messages,
            "tools": self.tools,
            "stream": False,
        }
        response = self.client.chat(**ollama_request)
        tool_calls = response.get("message", {}).get("tool_calls") or []

        # Ensure messages list is properly updated
        response_message = response.get("message", {})
        if response_message:
            messages.append(response_message)
            print(f"[DEBUG] Response message: {response_message.thinking}")

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
                    result = tool_function(**args)
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

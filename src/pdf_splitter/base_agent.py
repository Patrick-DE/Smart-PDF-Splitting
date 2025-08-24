# src/pdf_splitter/base_agent.py
"""
Abstract base class for PDF splitting agents (Ollama, LangChain, etc.).
Provides common agent logic and interface to reduce code duplication.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BasePDFSplitterAgent(ABC):
    def __init__(self, tools: List[Dict[str, Any]], model_name: str, config: Any):
        self.tools = tools
        self.model_name = model_name
        self.config = config

    @abstractmethod
    def run(self, messages: List[Dict[str, Any]], state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the agent with the given messages and state.
        Args:
            messages: List of message dicts (role/content).
            state: Dict with current_page_index, total_pages, current_document_pages, etc.
        Returns:
            Updated state with new messages and any tool call results.
        """
        pass

    def build_prompt(self, state: Dict[str, Any]) -> str:
        """
        Build the agent prompt from the current state.
        """
        return f"""
You are a highly specialized AI agent tasked with splitting a large PDF containing multiple business letters into distinct documents, one per letter. Your operation is precise, methodical, and you must strictly follow the workflow and decision logic below.

### CONTEXT
- Total Pages in PDF: {state['total_pages']}
- Current Position: You are analyzing pages {state['current_page_index']} and {state['current_page_index']+1}.
- Collected Pages: {state['current_document_pages']}
---
### CORE WORKFLOW
1. Use the `read_consecutive_pages` tool to retrieve the text for the current and next page. Call `extract_metadata` ONLY when you are at the start of a new document (the first page of a document). Do not call `extract_metadata` on subsequent pages of the same document.
2. Analyze both pages (consider text topic, key themes, logos, page numbers, and visual elements) to determine if a new document begins on the next page.
3. Apply the Decision-Making Logic below. Always respond with a structured tool call when required, not just a description.
---
### DECISION-MAKING LOGIC
- High Confidence (New Document): If the next page clearly starts a new document, you must call the `save_document` tool immediately. Do not narrate your intent—make the tool call directly.
- High Confidence (Same Document): If the next page is clearly a continuation, do not call any tools. Respond with a brief rationale and proceed.
- Uncertainty / Low Confidence: If unsure, use `search_for_similar_cases` and/or `ask_human_for_confirmation` as needed.
---
### CRITICAL FINAL STEP
After the last page, you must call `extract_metadata` then `save_document` to save the last set of pages. Always use structured tool calls when required.

IMPORTANT: You have access to the following tools. You must use the exact tool name and argument structure as defined below:

1. read_consecutive_pages: Use this tool to read the text of two consecutive pages. Call it with 'current_page_index' (an integer, e.g., 0 for the first page).
2. extract_metadata: Use this tool to extract metadata from the current and next page text. Call it with 'current_page_text' and 'next_page_text' (both strings).
3. search_for_similar_cases: Use this tool to search for similar cases. Call it with 'current_page_text' and 'next_page_text' (both strings).
4. ask_human_for_confirmation: Use this tool to ask for human feedback. Call it with 'question' (a string).
5. save_document: Use this tool to save a document. Call it with 'page_indices' (list of integers) and 'metadata' (an object containing date, company, and title). The controller may pre-populate 'metadata' when it has already been extracted.

Do NOT use any other tool names or argument structures. Do NOT use 'page_numbers', 'pdf_file_path', or any other arguments. Only use the tools and arguments exactly as defined above.

"""

    def update_state(self, state: Dict[str, Any], tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Update the agent state after tool calls.
        Args:
            state: Current agent state.
            tool_calls: List of tool call dicts.
        Returns:
            Updated state dict.
        """
        # Example: increment page index, update collected pages, etc.
        # This should be customized in subclasses if needed.
        return state

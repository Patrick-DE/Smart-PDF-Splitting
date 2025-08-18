# src/pdf_splitter/main.py
import os
import sys
from PyPDF2 import PdfReader
from langchain_core.messages import HumanMessage


from . import config
from . import tools
from .agent import AgentState, create_agent, agent_node, tool_node, should_continue
# from .llm_setup import get_llm # We'll create this helper

def get_llm():
    if config.USE_MODEL == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=config.OLLAMA_MODEL_NAME, base_url="http://localhost:11434")
    elif config.USE_MODEL == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        import os
        api_key = os.getenv("GEMINI_API_KEY")
        return ChatGoogleGenerativeAI(model=config.GEMINI_MODEL_NAME, google_api_key=api_key)
    else:
        raise ValueError(f"Unknown USE_MODEL: {config.USE_MODEL}")

if __name__ == "__main__":
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(config.PDF_PATH):
        print(f"Error: PDF file not found at '{config.PDF_PATH}'. Please update the path.")
        sys.exit(1)
    if not config.MONGO_URI:
        print("Warning: MONGO_URI not set. The agent cannot learn from feedback.")

    try:
        reader = PdfReader(config.PDF_PATH)
        total_pages = len(reader.pages)
        print(f"Processing PDF with {total_pages} pages...")

        # Import tool functions from tools.py
        from .tools import read_consecutive_pages, search_for_similar_cases, ask_human_for_confirmation, save_document
        tools_list = [read_consecutive_pages, search_for_similar_cases, ask_human_for_confirmation, save_document]

        # Simple ToolExecutor implementation (since langgraph.prebuilt.ToolExecutor is not available)
        class ToolExecutor:
            def __init__(self, tools):
                self.tools = {tool.name: tool for tool in tools}
            def invoke(self, tool_call):
                tool_fn = self.tools.get(tool_call['name'])
                if tool_fn:
                    return tool_fn(*tool_call.get('args', []))
                return f"Tool '{tool_call['name']}' not found."

        tool_executor = ToolExecutor(tools_list)
        llm = get_llm()
        agent = create_agent(llm, tools_list)

        # Build the graph
        from langgraph.graph import StateGraph
        from .agent import agent_node, tool_node, should_continue
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", lambda state: agent_node(state, agent, tools_list))
        workflow.add_node("tools", lambda state: tool_node(state, tool_executor))
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", should_continue)
        workflow.add_edge("tools", "agent")
        app = workflow.compile()

        # Initial state
        initial_state = {
            "messages": [HumanMessage(content="Please use the read_consecutive_pages tool to read the first two pages and decide if a split is needed.")],
            "current_page_index": 0,
            "total_pages": total_pages,
            "current_document_pages": [],
        }

        os.system('cls' if os.name == 'nt' else 'clear') # Clear console for better readability
        # Stream the execution
        for event in app.stream(initial_state):
            print("--- Current State ---")
            print(f"Processing Page Index: {event.get('agent', {}).get('current_page_index')}")
            print(f"Pages for current doc: {event.get('agent', {}).get('current_document_pages')}")
            print("\n--- Agent's Last Action ---")
            last_message = event.get('agent', {}).get('messages', [])[-1]
            # Dump all tool calls for this step
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                print(f"Tool Calls: {[tc['name'] for tc in last_message.tool_calls]}")
                for tc in last_message.tool_calls:
                    print(f"Tool Call: {tc['name']}")
                    print(f"Args: {tc['args']}")
            else:
                print("Continuing to next page...")
            print("\n" + "="*50 + "\n")

        print("PDF splitting process complete!")

    except FileNotFoundError:
        print(f"Error: The file '{config.PDF_PATH}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
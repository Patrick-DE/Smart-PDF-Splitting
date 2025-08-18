# src/pdf_splitter/agent.py
from typing import List, TypedDict
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from . import tools # Import your tools

# --- ToolMessage for tool outputs ---
class ToolMessage(BaseMessage):
    def __init__(self, tool_call_id, content):
        super().__init__(content=content)
        self.tool_call_id = tool_call_id
# src/pdf_splitter/agent.py
from typing import List, TypedDict
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from . import tools # Import your tools

# --- Agent State and Graph ---
class AgentState(TypedDict):
    messages: List[BaseMessage]
    current_page_index: int
    total_pages: int
    current_document_pages: List[int]

def create_agent(llm, tools):
    """Creates the agent with a detailed system prompt."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert AI agent for splitting multi-page PDF documents.
        Your goal is to iterate through a PDF page by page and decide where each new document begins.

        Workflow:
        1. You are currently at page {current_page_index} of {total_pages} total pages.
        2. Use the `read_consecutive_pages` tool to get the text for the current and next pages.
        3. Analyze the text. A new document often starts with a clear title, company name, date, or a significant change in formatting/topic from the previous page.
        4. **Decision Making**:
           - If you are highly confident it's a new document, call the `save_document` tool. The `page_indices` should be the list of pages for the *previous* document you have collected. Then, start a new list for the current page.
           - If you are highly confident it is *not* a new document, continue to the next page.
           - If you are unsure, FIRST use `search_for_similar_cases` with the text from both pages.
           - If the search result is clear, follow its guidance.
           - If the search is inconclusive, you MUST use `ask_human_for_confirmation`.
        5. **Saving a Document**: Before calling `save_document`, you must determine the company, date (YYYYMMDD), and a short title from the document's content. If any part is missing, use "UnknownCompany", "YYYYMMDD", or "Untitled".
        6. **Final Step**: After processing the last page, you must make one final call to `save_document` to save the last collected set of pages.
        """),
        ("placeholder", "{messages}")
    ])
    return prompt | llm.bind_tools(tools)

def agent_node(state: AgentState, agent, tools):
    """The main agent node that reasons and calls tools."""
    print(f"[DEBUG] agent_node: current_page_index={state['current_page_index']}, total_pages={state['total_pages']}, current_document_pages={state['current_document_pages']}")
    result = agent.invoke({
        "messages": state["messages"],
        "current_page_index": state["current_page_index"],
        "total_pages": state["total_pages"]
    })
    print(f"[DEBUG] agent_node: result.tool_calls={getattr(result, 'tool_calls', None)}")
    # Preserve all state keys
    new_state = dict(state)
    new_state["messages"] = [result]
    return new_state

def tool_node(state: AgentState, tool_executor):
    """Executes tools and processes their outputs."""
    print(f"[DEBUG] tool_node: current_page_index={state['current_page_index']}, current_document_pages={state['current_document_pages']}")
    tool_calls = state["messages"][-1].tool_calls
    print(f"[DEBUG] tool_node: tool_calls={tool_calls}")
    tool_messages = []
    
    if not tool_calls:
        # If no tool is called, it means we are just continuing
        state["current_document_pages"].append(state["current_page_index"])
        state["current_page_index"] += 1
        print(f"[DEBUG] tool_node: incremented current_page_index to {state['current_page_index']}")
        return state

    for tool_call in tool_calls:
        output = tool_executor.invoke(tool_call)
        print(f"[DEBUG] tool_node: tool_call={tool_call['name']}, output={output}")
        tool_messages.append(ToolMessage(tool_call_id=tool_call['id'], content=str(output)))

        # State updates based on tool calls
        if tool_call['name'] == 'save_document':
            # After saving, start a new document with the current page
            state["current_document_pages"] = [state["current_page_index"]]
            state["current_page_index"] += 1
        elif tool_call['name'] == 'ask_human_for_confirmation':
            # If human says yes, it's a new doc. Save the old one.
            if 'yes' in str(output).lower():
                # This logic is now handled by the agent's next turn based on the tool output.
                # The agent will receive the human feedback and decide to call save_document.
                pass

    state["messages"].extend(tool_messages)
    return state

def should_continue(state: AgentState):
    """Determines if the agent should continue or end."""
    if state["current_page_index"] >= state["total_pages"]:
        return END
    return "agent"
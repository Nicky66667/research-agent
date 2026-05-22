from typing import Annotated, List # use annotated and list type hints
from typing_extensions import TypedDict # TypedDict defines dictionary field types
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # add_messages is a built-in LangGraph reducer:
    # when node update, new messages are appended instead of overwritten
    messages: Annotated[list, add_messages]

    original_query: str  # Original user query
    sub_queries: List[str]  # List of decomposed sub-queries
    retrieved_docs: List[dict]  # Retrieved document chunks
    final_report: str  # Final generated report
    step_count: int  # Current execution step count (prevents infinite loops)
    current_step: str  # Current step description (for UI display)
    loaded_papers: List[str]  # Loaded paper IDs (used for deduplication)


from langchain_core.messages import HumanMessage
from agents import build_graph

def test_agent_run():
     graph = build_graph(db_path=":memory:") # use SQLite,auto-clean after test
     config = {"configuration": {"thread_id": "test-agent-1"}}

     result = graph.invoke({ "messages":[HumanMessage(content='What is RAG in NLP')]},config = {"configurable": {"thread_id": "test-agent-1"}})

     assert "final_report" in result
     assert len(result["final_report"]) > 90

     print("\n[Agent Test] Final report preview", result["final_report"][:200])



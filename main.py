from langchain_core.messages import HumanMessage
from agents import build_graph

def run_agent(query:str,thread_id:str="test-1"):
    """
    takes a question and runs the full AI graph

    thread_id: identifies one research session.
    Same thread_id = shared memory across multiple calls.
    """

    graph = build_graph()

    # Config tells the graph which memory "slot" to use
    config = {"configurable":{"thread_id":thread_id}}

    # print a line to divide before show the question
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}\n")

    # run the graph
    result = graph.invoke(
        {"messages":[HumanMessage(content=query)]},
        config = config
    )

    # print the final report from the result
    print("\n[FINAL REPORT]")
    print(result.get("final_report","No report generated"))

    return result

if __name__ == "__main__":
    run_agent("What are the key innovations in transformer attention mechanisms?")
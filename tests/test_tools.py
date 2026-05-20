from tools.arxiv_tool import arxiv_search, rag_query

def test_arxiv_search():
    """
    Test if arXiv search returns results. download PDF(take about 10‑30 seconds).
    """

    result = arxiv_search.invoke({"query":"transformer attention mechanism","max_resulkts":1})

    assert isinstance(result, str)
    assert "Title:" in result
    print("\n[arxiv Test Result]\n",result[:500])

def test_rag_query_after_arxiv():
    """First search arXiv to load one paper, then test RAG retrieval."""

    # first load paper
    arxiv_search.invoke({"query":"attention is all you need","max_results":1})

    # search again
    result = rag_query.invoke({"query":"what is multi-head attention?","top_k":3})

    assert isinstance(result, str)

    if "empty" not in result:
        assert "Relevance:" in result
    print("\n[RAG Query Result]\n", result[:500])



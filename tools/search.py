from langchain_core.tools import tool
from tavily import TavilyClient
from config.settings import TAVILY_API_KEY

_tavily = TavilyClient(api_key = TAVILY_API_KEY)

@tool
def web_search(query:str) -> str:
    """
    Use Tavily for web search, then format results into text readable for LLM

    Args:
        query：Search query in natural language
    """

    try:
        results = _tavily.search(
            query=query,
            search_depth = "basic",
            max_results = 5
        )

        if not results.get("results"):
            return f"No web results found for: {query}"

        formatted = []

        """
        Format each result to:
        Source: https://...
        Title: xxx
        Content: xxx（up to 400 characters）
        """
        for r in results["results"]:
            formatted.append(
                f"Source:{r.get('url', 'unknown')}\n"
                f"Title:{r.get('title', '')}\n"
                f"Content:{r.get('content','')[:400]}..."
            )

        return "\n\n---\n\n".join(formatted)

    except Exception as e:
        return f"Web search failed: {str(e)}"


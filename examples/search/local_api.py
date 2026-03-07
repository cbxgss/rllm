import logging
import os
from typing import Any

from examples.dualrag.server.local.api_local import FastApiRetriever
from rllm.tools.tool_base import Tool, ToolOutput

logger = logging.getLogger(__name__)


class LocalRetrievalTool(Tool):
    """
    A tool for dense search using the local retrieval server.

    This tool connects to a locally running dense retrieval server
    and performs dense retrieval using E5 embeddings on the indexed Wikipedia corpus.
    """

    NAME = "local_search"
    DESCRIPTION = "Search for information using a dense retrieval server with Wikipedia corpus"

    def __init__(
        self,
        name: str = NAME,
        description: str = DESCRIPTION,
        server_url: str = None,
        timeout: float = 30.0,
        max_topk: int = 10,
    ):
        """
        Initialize the Local Retrieval Tool.

        Args:
            name: Tool name
            description: Tool description
            server_url: URL of the local retrieval server (if None, checks search_url env var)
            timeout: Request timeout in seconds
            max_topk: Maximum number of results to return
        """
        # Use environment variable if server_url not provided
        if server_url is None:
            search_url = os.getenv("search_url", "127.0.0.1")
            server_url = f"http://{search_url}:8011"

        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.max_topk = max_topk

        # Initialize the retriever from api_local.py
        self.retriever = FastApiRetriever(self.server_url)

        super().__init__(name=name, description=description)

        # Test server connection
        self._test_connection()

    def _test_connection(self):
        """Test connection to the retrieval server."""
        try:
            corpus_len = self.retriever.corpus_len()
            logger.info(f"Successfully connected to retrieval server at {self.server_url}, corpus size: {corpus_len}")
        except Exception as e:
            logger.warning(f"Could not connect to retrieval server: {e}")

    @property
    def json(self):
        """Return tool JSON schema for LLM function calling."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query to retrieve relevant documents"},
                    },
                    "required": ["query"],
                },
            },
        }

    def _format_search_results(self, results: list[dict[str, Any]]) -> str:
        """Format search results for LLM consumption."""
        if not results:
            return "No relevant documents found."

        formatted_results = []
        for i, result in enumerate(results[: self.max_topk], 1):
            # Extract key information
            doc_id = result.get("id", f"doc_{i}")
            content = result.get("content", "")
            score = result.get("score", 0.0)

            formatted_result = f"[Document {i}]\n{content}\n"
            formatted_results.append(formatted_result)

        return "\n".join(formatted_results)

    def forward(self, query: str, top_k: int = 10) -> ToolOutput:
        """
        Execute a search query using the dense retrieval server.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            ToolOutput: Search results or error message
        """
        try:
            # Use provided parameters or defaults
            top_k = top_k or self.max_topk

            # Call the retriever
            result = self.retriever.retrieve(
                source="wiki",
                query=query,
                topk=10,
            )

            # Transform result from api_local format to local_retrieval_tool format
            idxs = result.get("idxs", [])
            docs = result.get("docs", [])
            scores = result.get("scores", [])

            # Build results list
            results = []
            for idx, doc, score in zip(idxs, docs, scores):
                results.append({
                    "id": idx,
                    "content": doc,
                    "score": score,
                })

            if not results:
                return ToolOutput(name=self.name, output="No relevant documents found for the query.")

            # Format results
            formatted_output = self._format_search_results(results)

            # Create metadata for potential downstream use
            metadata = {"query": query, "num_results": len(results), "retriever_type": "dense", "server_url": self.server_url}

            return ToolOutput(name=self.name, output=formatted_output, metadata=metadata)

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return ToolOutput(name=self.name, error=f"Retrieval failed: {str(e)}")


# Convenience function for tool registry
def create_local_retrieval_tool(server_url: str = None, max_topk: int = 10) -> LocalRetrievalTool:
    """
    Create a LocalRetrievalTool instance with specified configuration.

    Args:
        server_url: URL of the dense retrieval server (if None, uses search_url env var or 127.0.0.1:8011)
        max_topk: Maximum number of results to return

    Returns:
        LocalRetrievalTool instance
    """
    return LocalRetrievalTool(server_url=server_url, max_topk=max_topk)

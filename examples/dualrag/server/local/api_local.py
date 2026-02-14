import os
from dotenv import load_dotenv
from typing import Any
import requests
import aiohttp
from tenacity import retry, stop_never, wait_random_exponential, stop_after_attempt


class FastApiRetriever:
    def __init__(self, url: str):
        self.url = url

    @retry(reraise=True, stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, min=1, max=10))
    def corpus_len(self) -> int:
        with requests.Session() as session:
            session.trust_env = False
            response = session.get(f"{self.url}/corpus_len/")
            results = response.json()
            return results

    @retry(reraise=True, stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, min=1, max=10))
    async def acorpus_len(self) -> int:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.url}/corpus_len/") as response:
                results = await response.json()
                return results

    @retry(reraise=True, stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, min=1, max=10))
    def retrieve(self, source: str, query: str, topk: int = 3) -> dict:
        with requests.Session() as session:
            session.trust_env = False
            data = {
                "source": source,
                "query": query,
                "topk": topk,
            }
            response = session.post(f"{self.url}/retrieve/", json=data)
            results = response.json()
            return results

    @retry(reraise=True, stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, min=1, max=10))
    async def aretrieve(self, source: str, query: str, topk: int = 10) -> dict:
        async with aiohttp.ClientSession() as session:
            data = {
                "source": source,
                "query": query,
                "topk": topk,
            }
            async with session.post(f"{self.url}/retrieve/", json=data) as response:
                results = await response.json()
                return results

    @retry(reraise=True, stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, min=1, max=10))
    def retrieve_batch(self, sources: list[str], queries: list[str], topk: int = 3) -> list[dict]:
        with requests.Session() as session:
            session.trust_env = False
            data = {
                "sources": sources,
                "queries": queries,
                "topk": topk,
            }
            response = session.post(f"{self.url}/retrieve_batch/", json=data)
            results = response.json()
            return results

    @retry(reraise=True, stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, min=1, max=10))
    async def aretrieve_batch(self, sources: list[str], queries: list[str], topk: int = 3) -> list[dict]:
        async with aiohttp.ClientSession() as session:
            data = {
                "sources": sources,
                "queries": queries,
                "topk": topk,
            }
            async with session.post(f"{self.url}/retrieve_batch/", json=data) as response:
                results = await response.json()
                return results


@retry(reraise=True, stop=stop_never, wait=wait_random_exponential(min=1, max=10))
async def search(query: str, timeout: int = 180) -> dict[str, Any]:
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)
    search_url = os.getenv("search_url", "127.0.0.1")
    retriever = FastApiRetriever(f"http://{search_url}:8011")
    result = await retriever.aretrieve("wiki", query, topk=10)
    return result

if __name__ == "__main__":
    search_url = os.getenv("search_url", "127.0.0.1")
    async def main():
        retriever = FastApiRetriever(f"http://{search_url}:8011")
        corpus_len = await retriever.acorpus_len()
        print(f"corpus_len: {corpus_len}")
        result = await retriever.aretrieve("wiki", "What is the second highest-grossing Kannada movie of all time?", topk=5)
        print(f"retrieve result: {result}")
        # sources = ["wiki", "wiki", "wiki"]
        # queries = ["What is RAG?", "Explain FAISS.", "Describe asyncio in Python."]
        # batch_result = await retriever.aretrieve_batch(sources, queries, topk=5)
        # print(f"retrieve batch result: {batch_result}")
        """
        return:
        {
            "idxs": [12345, 67890, 23456, 34567, 45678],
            "docs": [
                "Document content 1...",
                "Document content 2...",
                "Document content 3...",
                "Document content 4...",
                "Document content 5..."
            ],
            "scores": [0.95, 0.93, 0.90, 0.89, 0.85]
        }
        """


    import asyncio
    asyncio.run(main())

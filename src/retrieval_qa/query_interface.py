from .vector_store import VectorStoreManager
from .config import DEFAULT_TOP_K
from typing import Dict, List, Optional

class QueryInterface:
    def __init__(self, vector_store_id: str, api_key: Optional[str] = None):
        self.vector_store_id = vector_store_id
        self.vector_store_manager = VectorStoreManager(api_key)

    def search_only(self, query: str, top_k: int = DEFAULT_TOP_K) -> Dict:
        """
        Standalone vector search that returns raw results
        """
        return self.vector_store_manager.search_vector_store(
            self.vector_store_id, query, top_k
        )

    def search_with_llm(self, query: str, model: str = "gpt-4o-mini") -> Dict:
        """
        Search with LLM using Responses API
        """
        return self.vector_store_manager.query_with_llm(
            self.vector_store_id, query, model
        )

    def batch_query(self, queries: List[str], use_llm: bool = True, model: str = "gpt-4o-mini") -> List[Dict]:
        """
        Batc query with multiple questions
        """
        return self.vector_store_manager.batch_search(
            self.vector_store_id, queries, use_llm, model
        )
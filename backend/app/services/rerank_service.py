from typing import Any, Dict, List, Optional

from flashrank import Ranker, RerankRequest

from backend.app.core.logging import logger


class RerankService:
    """Local neural cross-encoder reranker using FlashRank TinyBERT."""

    def __init__(self, model_name: str = "ms-marco-TinyBERT-L-2-v2"):
        """Initializes rerank service with specified FlashRank model."""
        self.model_name = model_name
        self._ranker: Optional[Ranker] = None

    @property
    def ranker(self) -> Optional[Ranker]:
        """Initializes and returns cached FlashRank Ranker instance."""
        try:
            if self._ranker is None:
                logger.info(f"RerankService: Initializing FlashRank model '{self.model_name}'...")
                self._ranker = Ranker(model_name=self.model_name)
            return self._ranker
        except Exception as e:
            logger.error(f"RerankService: Failed to initialize FlashRank model '{self.model_name}': {e}", exc_info=True)
            return None

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 4,
    ) -> List[Dict[str, Any]]:
        """Reranks candidate document chunks using cross-encoder scoring."""
        try:
            if not documents:
                return []
            if len(documents) <= 1:
                return documents[:top_k]

            ranker_client = self.ranker
            if ranker_client is None:
                logger.warning("RerankService: Ranker unavailable, returning unranked candidates.")
                return documents[:top_k]

            passages = []
            for i, doc in enumerate(documents):
                text_val = doc.get("content") or doc.get("text") or ""
                passages.append({
                    "id": i,
                    "text": text_val,
                    "meta": doc,
                })

            rerank_request = RerankRequest(query=query, passages=passages)
            results = ranker_client.rerank(rerank_request)

            reranked_docs: List[Dict[str, Any]] = []
            for r in results[:top_k]:
                doc_meta = r.get("meta", {})
                doc_meta["rerank_score"] = float(r.get("score", 0.0))
                reranked_docs.append(doc_meta)

            logger.info(
                f"RerankService: Reranked {len(documents)} candidates down to top {len(reranked_docs)}."
            )
            return reranked_docs
        except Exception as e:
            logger.warning(f"RerankService: Reranking failed ({e}), returning original order.")
            return documents[:top_k] if documents else []


rerank_service = RerankService()

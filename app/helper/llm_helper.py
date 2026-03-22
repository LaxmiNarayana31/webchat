from google import genai
from langchain.llms.base import LLM
from langchain_core.embeddings import Embeddings
from typing import Any, List
from pydantic import Field


# ---------------- Embeddings ----------------
class GoogleGeminiEmbeddings(Embeddings):
    def __init__(self, api_key):
        super().__init__()
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-embedding-001"
        self.output_dim = 768

    def embed_documents(self, texts):
        response = self.client.models.embed_content(
            model=self.model,
            contents=texts,   # batch call
            config={
                "task_type": "RETRIEVAL_DOCUMENT",
                "output_dimensionality": self.output_dim
            }
        )
        return [emb.values for emb in response.embeddings]

    def embed_query(self, query):
        response = self.client.models.embed_content(
            model=self.model,
            contents=query,
            config={
                "task_type": "RETRIEVAL_QUERY",
                "output_dimensionality": self.output_dim
            }
        )
        return response.embeddings[0].values


# ---------------- LLM ----------------
class GoogleGeminiLLM(LLM):
    client: Any = None
    models: List[str] = ["gemini-3.1-flash-lite-preview", "gemini-3-flash-preview"]

    def __init__(self, api_key):
        super().__init__()
        # Use object.__setattr__ to set attributes in a Pydantic model (LLM)
        object.__setattr__(self, 'client', genai.Client(api_key=api_key))

    def _call(self, prompt, stop=None):
        errors = []

        for model_name in self.models:
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                return response.text
            except Exception as e:
                errors.append(f"{model_name}: {str(e)}")
                continue

        raise Exception(f"All Gemini models failed:\n" + "\n".join(errors))

    @property
    def _identifying_params(self):
        return {"model_name": "Gemini Multi-Model Fallback"}

    @property
    def _llm_type(self):
        return "gemini_fallback"
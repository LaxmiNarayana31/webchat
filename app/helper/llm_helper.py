import google.generativeai as genai
from langchain.llms.base import LLM
from langchain_core.embeddings import Embeddings


class GoogleGeminiEmbeddings(Embeddings):
    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document",
            )
            embeddings.append(result["embedding"])
        return embeddings

    def embed_query(self, query):
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=query,
            task_type="retrieval_query",
        )
        return result["embedding"]


# Custom LLM class to get response from Google Gemini
class GoogleGeminiLLM(LLM):
    def _call(self, prompt, stop=None):
        models = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"]
        errors = []

        for model_name in models:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt).text
                return response
            except Exception as e:
                errors.append(f"{model_name}: {str(e)}")
                continue
        
        # If all models fail, raise an exception with all errors
        error_msg = "\n".join(errors)
        raise Exception(f"All Gemini models failed:\n{error_msg}")

    @property
    def _identifying_params(self):
        return {"model_name": "Gemini Multi-Model Fallback"}

    @property
    def _llm_type(self):
        return "gemini_fallback"
import json
import re
from typing import List

from backend.app.clients.gemini_client import gemini_client
from backend.app.clients.groq_client import groq_client
from backend.app.core.logging import logger


class QueryExpansionService:
    """Generates HyDE and multi-query search variations to enhance retrieval recall."""

    def expand_query(self, query: str) -> List[str]:
        """Expands user query into lexical variations and hypothetical answers."""
        try:
            if not query:
                return []

            clean_query = query.strip()
            queries = [clean_query]
            if len(clean_query.split()) < 3:
                return queries

            system_instruction = (
                "You are an expert AI search optimizer. Given a user search query, produce 2 distinct "
                "search variations (synonyms, technical terms, specific phrasing) and 1 brief hypothetical "
                "answer excerpt (HyDE). Output strictly JSON with a single key 'variations' containing a list of strings."
            )

            prompt = f"User Query: {clean_query}\n\nOutput JSON:"

            # Attempt Gemini Flash Lite first
            try:
                raw_response = gemini_client.generate_text(
                    model="gemini-3.1-flash-lite",
                    prompt=prompt,
                    system_instruction=system_instruction,
                    temperature=0.3,
                )
                parsed = self._parse_variations(raw_response)
                for v in parsed:
                    if v and v not in queries:
                        queries.append(v)
                logger.info(f"QueryExpansionService: Expanded query into {len(queries)} variations.")
                return queries[:4]
            except Exception as e:
                logger.warning(f"QueryExpansionService: Gemini expansion failed ({e}). Trying Groq fallback.")

            # Fallback to Groq
            try:
                groq_resp = groq_client.generate_text(
                    model="openai/gpt-oss-120b",
                    prompt=prompt,
                    system_instruction=system_instruction,
                    temperature=0.3,
                )
                parsed = self._parse_variations(groq_resp)
                for v in parsed:
                    if v and v not in queries:
                        queries.append(v)
                return queries[:4]
            except Exception as err:
                logger.warning(f"QueryExpansionService: Fallback expansion failed ({err}). Using original query.")
                return queries

        except Exception as outer_e:
            logger.error(f"QueryExpansionService: Error expanding query '{query}': {outer_e}", exc_info=True)
            return [query.strip()] if query else []

    @staticmethod
    def _parse_variations(raw_text: str) -> List[str]:
        """Extracts list of variations from JSON or formatted bullet list."""
        try:
            if not raw_text:
                return []

            clean = raw_text.strip()
            if "`json" in clean:
                clean = clean.split("`json")[1].split("`")[0].strip()
            elif "`" in clean:
                clean = clean.split("`")[1].split("`")[0].strip()

            try:
                data = json.loads(clean)
                if isinstance(data, dict) and "variations" in data:
                    return [str(item).strip() for item in data["variations"] if str(item).strip()]
                elif isinstance(data, list):
                    return [str(item).strip() for item in data if str(item).strip()]
            except Exception:
                pass

            lines = [re.sub(r"^[-*\d.)\s]+", "", line).strip() for line in clean.splitlines() if line.strip()]
            return [l for l in lines if len(l) > 3]
        except Exception as e:
            logger.warning(f"QueryExpansionService: Error parsing variations: {e}")
            return []


query_expansion_service = QueryExpansionService()

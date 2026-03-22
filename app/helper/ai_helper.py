from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from app.helper.general_helper import Helper
from app.helper.llm_helper import GoogleGeminiEmbeddings, GoogleGeminiLLM
from app.utils.exception_handler import handle_exception

class AIHelper:
    def __init__(self, api_key):
        # Initialize models once to save memory and processing time
        self.llm = GoogleGeminiLLM(api_key)
        self.embeddings = GoogleGeminiEmbeddings(api_key)

    def get_vectorstore_from_url(self, url: str):
        """Loads URL, splits text, and creates a FAISS vectorstore."""
        try:
            is_valid, error_message = Helper.validate_url(url)
            if not is_valid:
                return None, error_message

            loader = WebBaseLoader(url)
            docs = loader.load()
            
            # Optimized splitting for LLM context windows
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(docs)

            vector_store = FAISS.from_documents(chunks, self.embeddings)
            return vector_store, None
        except Exception as e:
            return None, handle_exception(e)

    def _get_context_retriever_chain(self, vector_store):
        """Reformulates user input based on history for better retrieval."""
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", "Based on the conversation above, generate a concise search query to find relevant info.")
        ])
        return create_history_aware_retriever(self.llm, retriever, prompt)

    def _get_conversational_rag_chain(self, retriever_chain):
        """Core RAG logic with system instructions."""
        system_instruction = (
            "Answer strictly using the provided context and history.\n"
            "If unknown, say 'Information not found.' Keep it brief.\n\n"
            "Context: {context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_instruction),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])
        
        doc_chain = create_stuff_documents_chain(self.llm, prompt)
        return create_retrieval_chain(retriever_chain, doc_chain)

    def get_response(self, user_input: str, chat_history: list, vector_store):
        """Orchestrates the retrieval and generation process."""
        try:
            if not vector_store:
                return "Please provide a valid data source first."

            # Build chains
            retriever_chain = self._get_context_retriever_chain(vector_store)
            rag_chain = self._get_conversational_rag_chain(retriever_chain)

            # Execute
            response = rag_chain.invoke({
                "chat_history": chat_history,
                "input": user_input
            })
            
            return response.get("answer", "I encountered an error processing that.")
        except Exception as e:
            return handle_exception(e)
import logging
logging.getLogger("langchain_community.utils.user_agent").setLevel(logging.ERROR)

import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

from app.helper.ai_helper import AIHelper
from app.helper.general_helper import Helper

def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=api_key)
    st.set_page_config(page_title="WebChat", page_icon="🌐")

    if "ai_helper" not in st.session_state:
        st.session_state.ai_helper = AIHelper(api_key)

    if "current_chat" not in st.session_state:
        st.session_state.current_chat = []
    if "chat_histories" not in st.session_state:
        st.session_state.chat_histories = []
    if "selected_chat_index" not in st.session_state:
        st.session_state.selected_chat_index = None
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    sidebar()
    chat_interface()

def sidebar():
    with st.sidebar:
        st.header("Chat with Websites")
        website_url = st.text_input("Enter Website URL")

        if st.button("Load Website"):
            with st.spinner("Processing website..."):
                st.session_state.selected_chat_index = None
                
                vector_store, error_message = st.session_state.ai_helper.get_vectorstore_from_url(website_url)

                if error_message:
                    st.error(error_message)
                elif vector_store:
                    st.session_state.vector_store = vector_store

                    current_chat = {
                        "website_url": website_url,
                        "vector_store": vector_store,
                        "messages": [
                            {
                                "role": "assistant",
                                "content": f"Hello! I'm ready to answer questions about {website_url}",
                            }
                        ],
                    }
                    st.session_state.chat_histories.insert(0, current_chat)
                    st.session_state.current_chat = current_chat["messages"]
                    st.success("Website loaded successfully!")

        if st.button("+ New Chat"):
            st.session_state.selected_chat_index = None
            st.session_state.vector_store = None
            st.session_state.current_chat = []
            st.rerun()

        display_chat_history()

def display_chat_history():
    st.sidebar.header("Chat History")
    for idx, history in enumerate(st.session_state.chat_histories):
        website_url = history.get("website_url", "Unknown Website")
        # Added a unique key to prevent duplicate widget errors
        if st.sidebar.button(f"{website_url}", key=f"hist_{idx}"):
            st.session_state.selected_chat_index = idx
            st.session_state.vector_store = history.get("vector_store")
            st.session_state.current_chat = history.get("messages", [])
            st.rerun()

def chat_interface():
    st.header("WebChat 🌐")

    # Core history display logic (Maintained)
    if st.session_state.selected_chat_index is not None:
        messages = st.session_state.chat_histories[st.session_state.selected_chat_index].get("messages", [])
    else:
        messages = st.session_state.current_chat

    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_query = st.chat_input("Type your message...")
    if user_query:
        with st.chat_message("user"):
            st.markdown(user_query)

        if st.session_state.vector_store is None:
            with st.chat_message("assistant"):
                st.warning("Please load a website first.")
        else:
            with st.chat_message("assistant"):
                placeholder = st.empty()
                placeholder.markdown("Generating response...")
                
                response = st.session_state.ai_helper.get_response(
                    user_query, 
                    st.session_state.current_chat, 
                    st.session_state.vector_store
                )
                
                placeholder.empty()
                Helper.typewriter_effect(response, speed=20)

            # Your core history maintenance logic (Maintained)
            if st.session_state.selected_chat_index is not None:
                current_chat = st.session_state.chat_histories[st.session_state.selected_chat_index]
                current_chat["messages"].append({"role": "user", "content": user_query})
                current_chat["messages"].append({"role": "assistant", "content": response})
                st.session_state.current_chat = current_chat["messages"]
            else:
                st.session_state.current_chat.append({"role": "user", "content": user_query})
                st.session_state.current_chat.append({"role": "assistant", "content": response})

            st.rerun()

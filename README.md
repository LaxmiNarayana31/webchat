# WebChat 🌐

WebChat is an intelligent Streamlit application that allows users to have interactive conversations with the content of any website. Powered by Google's Gemini AI, it scrapes website content, creates embeddings, and lets you query the information in a natural, conversational manner.

## 🚀 Features

-   **Chat with Any Website**: Input a URL, and WebChat will process its content for you to question.
-   **Multi-Model Intelligence**: integrated with multiple Gemini models for robustness.
    -   **Primary**: `gemini-2.5-flash`
    -   **Fallback 1**: `gemini-2.5-flash-lite`
    -   **Fallback 2**: `gemini-2.0-flash`
    -   *Automatic Fallback*: If a model is exhausted or unavailable, the system transparently switches to the next available model.
-   **Context-Aware**: Remembers your conversation history for follow-up questions.
-   **Source-Grounded**: Responses are generated strictly based on the website's content to minimize hallucinations.
-   **Interactive UI**: Clean Streamlit interface with a typewriter effect for responses.

## 🛠️ Prerequisites

-   Python 3.8+
-   [Pipenv](https://pipenv.pypa.io/en/latest/) (recommended) or `pip`
-   A Google Gemini API Key

## 📥 Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/LaxmiNarayana31/webchat.git
    cd webchat
    ```

2.  **Install Dependencies**
    Using Pipenv:
    ```bash
    pipenv install
    pipenv shell
    ```

3.  **Environment Setup**
    Create a `.env` file in the root directory:
    ```bash
    touch .env
    ```
    Add your Gemini API Key:
    ```env
    GEMINI_API_KEY=your_actual_api_key_here
    ```

## ▶️ Usage

Run the application:

```bash
streamlit run main.py
```

1.  **Load a Website**: In the sidebar, enter a URL (e.g., `https://example.com`) and click **Load Website**.
2.  **Start Chatting**: Once loaded, use the chat input to ask questions about the page content.
3.  **View History**: Previous chats are saved in the sidebar for easy switching.

## 📂 Project Structure

-   `app/`: Main application code.
    -   `helper/`: Helper classes for AI (`ai_helper.py`), LLM (`llm_helper.py`), and utilities.
    -   `streamlit/`: UI components (`streamlit_app.py`).
-   `main.py`: Entry point for the application.

## 🤖 Gemini Integration Details

The application uses a custom `GoogleGeminiLLM` class that implements a robust fallback mechanism. It iterates through the following models:
1.  `gemini-2.5-flash` (High performance)
2.  `gemini-2.5-flash-lite` (Lightweight fallback)
3.  `gemini-2.0-flash` (Legacy fallback)

This ensures high availability even if you hit the rate limits of the primary model.

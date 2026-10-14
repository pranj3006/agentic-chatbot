# Agentic ChatBot

An interactive chatbot built with Streamlit, LangChain, and LangGraph. It can
answer general questions, use external tools, search PDFs uploaded to a chat,
and pause stock-order simulations until a user approves or rejects them.

## Application entry point

Run the Streamlit interface from the project directory:

```bash
streamlit run streamlit_app_db_backend.py
```

The main application files are:

| File | Responsibility |
| --- | --- |
| `streamlit_app_db_backend.py` | Streamlit UI, chat navigation, PDF uploads, response streaming, and approval buttons |
| `agentic_chatbot_db_backend.py` | LangGraph workflow, LLM setup, order-intent routing, and SQLite checkpoints |
| `tools.py` | Calculator, web search, stock quote, weather, stock-order, and RAG tools |
| `rag.py` | PDF text extraction, per-chat FAISS indexes, and document retrieval |

## Requirements

- Python 3.12 (the dependencies in `requirements.txt` target this environment)
- API credentials for the services you want to use; see [Configuration](#configuration)

## Setup

Create and activate a virtual environment, then install the pinned dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Create a `.env` file in the project directory. The application loads it on
startup:

```dotenv
OPENAI_API_KEY=your-openai-api-key
TAVILY_API_KEY=your-tavily-api-key
ALPHA_VANTAGE_API_KEY=your-alpha-vantage-api-key
WEATHERSTACK_API_KEY=your-weatherstack-api-key
```

`OPENAI_API_KEY` is needed for chatbot responses and PDF embeddings. The other
keys are used by their corresponding search, stock quote, and weather tools.
Keep `.env` private; do not commit API keys to source control.

Start the app:

```bash
streamlit run streamlit_app_db_backend.py
```

Streamlit prints a local URL (usually `http://localhost:8501`) where the app can
be opened in a browser.

## Using the application

- **Chats:** Select a conversation in the sidebar or create a new one. Chat
  history is restored from the local SQLite checkpoint database.
- **General questions:** Ask in the message box; the model responds directly
  when no tool is needed.
- **Tools:** The model can use web search, calculator, stock quote, weather, and
  document retrieval tools.
- **PDF questions:** Attach a PDF in the chat where it should be available. The
  app extracts text and builds a separate FAISS index for that chat. Ask
  questions about the PDF in the same conversation.
- **Stock-order approval:** Submit a clear buy or sell request that includes a
  quantity (for example, “buy 2 units of Google”). The graph pauses and displays
  **Yes** and **No** buttons. The order tool simulates an order; it does not
  connect to a brokerage or place a real trade.

## Local data

- `agenticchatbot.db` stores LangGraph conversation checkpoints.
- `<thread-id>_faiss.db/` folders store FAISS indexes for uploaded PDFs.

These files are generated at runtime and should be backed up if the associated
chat history or indexes need to be retained. Do not load FAISS index folders
from untrusted sources: the index loader enables deserialization for locally
generated metadata.

## Troubleshooting

- **Missing API key:** Check that `.env` is in the project directory and that
  the variable name matches the configuration section above. Restart Streamlit
  after changing environment variables.
- **No PDF results:** Confirm the PDF was uploaded in the same chat and that
  indexing completed successfully. Each chat has its own index.
- **Approval remains pending:** Use one of the approval buttons. While the
  graph is paused for a decision, the message composer is disabled for that
  chat.

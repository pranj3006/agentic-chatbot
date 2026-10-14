"""Per-chat PDF indexing and retrieval for retrieval-augmented answers.

Each chat stores a separate FAISS index in a folder named from its thread ID.
The ``rag_tool`` uses the active graph configuration to select that index.
"""

from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(override=True)

# OpenAI embeddings map PDF text chunks and user queries into the same vector space.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
RAG_INDEX_DIR = Path(__file__).resolve().parent


def _index_path(thread_id: str) -> Path:
    """Build the index folder path after validating a single-component thread ID."""
    if not thread_id or Path(thread_id).name != thread_id:
        raise ValueError("A valid chat thread ID is required to access its PDF index.")
    return RAG_INDEX_DIR / f"{thread_id}_faiss.db"


def ingest_rag_document(
    filepath: str,
    thread_id: str,
    source_name: str | None = None,
) -> None:
    """Extract, split, embed, and save a PDF in the specified chat's index.

    If an index already exists, add the new PDF's chunks to it instead of
    replacing the documents previously uploaded to that chat.
    """
    loader = PyPDFLoader(filepath)
    docs = loader.load()
    if not docs:
        raise ValueError("The uploaded PDF does not contain any readable pages.")
    for document in docs:
        if source_name:
            document.metadata["source"] = source_name

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    if not chunks:
        raise ValueError("The uploaded PDF does not contain any readable text.")

    vector_store = FAISS.from_documents(chunks, embeddings)
    index_path = _index_path(thread_id)
    if (index_path / "index.faiss").is_file():
        existing_store = FAISS.load_local(
            folder_path=str(index_path),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        existing_store.merge_from(vector_store)
        vector_store = existing_store
    vector_store.save_local(str(index_path))


def has_rag_index(thread_id: str) -> bool:
    """Return whether the chat currently has a saved FAISS index."""
    return (_index_path(thread_id) / "index.faiss").is_file()


def get_retriever(thread_id: str):
    """Load a chat's FAISS index and return a similarity retriever.

    The saved index must be trusted because loading it enables Python
    deserialization for the FAISS metadata.
    """
    index_path = _index_path(thread_id)
    if not has_rag_index(thread_id):
        raise FileNotFoundError(
            "No PDF has been uploaded for this chat. Upload a PDF before asking document questions."
        )
    vector_store = FAISS.load_local(
        folder_path=str(index_path),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )


@tool
def rag_tool(
    query: str,
    config: RunnableConfig,
) -> str:
    """Retrieve and format relevant passages from the current chat's PDF index.

    Args:
        query: The natural-language question to search for.
        config: LangGraph tool context containing the current thread ID.

    Returns:
        Matching passages annotated with each source PDF and page number.
    """
    thread_id = config.get("configurable", {}).get("thread_id")
    if not isinstance(thread_id, str) or not thread_id:
        raise ValueError("The current chat thread ID is missing from the tool context.")

    documents = get_retriever(thread_id).invoke(query)
    if not documents:
        return "No relevant information was found in the PDF"

    formatted_documents = []
    # Include citations with each passage so the model can attribute its answer.
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "Unknown source")
        page = document.metadata.get("page", "Unknown page")
        formatted_documents.append(
            f"Document: {index}\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content: {document.page_content}\n"
        )
    return "\n\n".join(formatted_documents)

"""Streamlit user interface for the persistent, tool-enabled chatbot.

This module owns chat selection, message display, PDF uploads, and the approval
controls for graph interrupts. Conversation state and checkpoints are managed by
``agentic_chatbot_db_backend``.
"""

from datetime import datetime, timezone
import os
import tempfile
from uuid import uuid4
from rag import has_rag_index, ingest_rag_document
import streamlit as st
from agentic_chatbot_db_backend import chat_bot, get_all_threads
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command


st.set_page_config(page_title="Agentic ChatBot", page_icon=":material/chat:", layout="wide")


def load_chat_sessions() -> dict[str, dict[str, str]]:
    """Load existing checkpointed conversations and derive their sidebar titles."""
    sessions = {}
    for chat_id in get_all_threads():
        config = {
            "configurable": {"thread_id": chat_id},
            "metadata": {"thread_id": chat_id},
            "run_name": "chat_trace",
        }
        snapshot = chat_bot.get_state(config)
        messages = snapshot.values.get("messages", []) if snapshot.values else []
        first_user_message = next(
            (message for message in messages if isinstance(message, HumanMessage)),
            None,
        )
        title = "New chat"
        if first_user_message and isinstance(first_user_message.content, str):
            title = first_user_message.content.strip().replace("\n", " ")
            title = f"{title[:40]}..." if len(title) > 40 else title

        sessions[chat_id] = {
            "title": title or "New chat",
            "updated_at": snapshot.created_at or datetime.now(timezone.utc).isoformat(),
        }
    return sessions


def create_chat() -> str:
    """Create an empty chat session and make it the active conversation."""
    chat_id = str(uuid4())
    st.session_state.chat_sessions[chat_id] = {
        "title": "New chat",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    st.session_state.active_chat_id = chat_id
    return chat_id


def get_pending_interrupt(config: RunnableConfig) -> str | None:
    """Return the first approval prompt waiting in this conversation, if any."""
    snapshot = chat_bot.get_state(config)
    for task in snapshot.tasks:
        if task.interrupts:
            return str(task.interrupts[0].value)
    return None


# Keep the selected chat and sidebar metadata across Streamlit reruns.
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = load_chat_sessions()
if not st.session_state.chat_sessions:
    create_chat()
if st.session_state.get("active_chat_id") not in st.session_state.chat_sessions:
    st.session_state.active_chat_id = max(
        st.session_state.chat_sessions,
        key=lambda chat_id: st.session_state.chat_sessions[chat_id]["updated_at"],
    )

# The sidebar lets the user start a chat or switch between saved conversations.
with st.sidebar:
    st.title("Agentic ChatBot")
    if st.button("＋  New chat", use_container_width=True, type="primary"):
        create_chat()
        st.rerun()

    st.divider()
    st.caption("Your chats")
    sorted_chats = sorted(
        st.session_state.chat_sessions.items(),
        key=lambda item: item[1]["updated_at"],
        reverse=True,
    )
    for chat_id, chat in sorted_chats:
        if st.button(
            chat["title"],
            key=f"open_chat_{chat_id}",
            use_container_width=True,
            type="secondary" if chat_id == st.session_state.active_chat_id else "tertiary",
        ):
            st.session_state.active_chat_id = chat_id
            st.rerun()

active_chat_id = st.session_state.active_chat_id
# LangGraph uses this stable ID to load and update the correct SQLite checkpoint.
config = {
    "configurable": {"thread_id": active_chat_id},
    "metadata": {"thread_id": active_chat_id},
    "run_name": "chat_trace",
}
snapshot = chat_bot.get_state(config)
messages = snapshot.values.get("messages", []) if snapshot.values else []
pending_interrupt = get_pending_interrupt(config)

st.title(st.session_state.chat_sessions[active_chat_id]["title"])
pdf_is_indexed = has_rag_index(active_chat_id)
st.caption(
    f"Chat ID: `{active_chat_id}` · "
    f"PDF index: {'ready' if pdf_is_indexed else 'not found for this chat'}"
)
if not pdf_is_indexed:
    st.caption("Attach the PDF in this chat's message box to index it for this chat.")
if not messages:
    st.caption("Ask me anything")

# Tool events and empty tool-call messages are internal; only show chat text.
for chat_message in messages:
    if isinstance(chat_message, HumanMessage):
        role = "user"
    elif isinstance(chat_message, AIMessage):
        role = "assistant"
    else:
        continue

    if isinstance(chat_message.content, str) and chat_message.content.strip():
        with st.chat_message(role):
            st.markdown(chat_message.content)

submission = st.chat_input(
    "Message Agentic ChatBot or attach a PDF",
    accept_file=True,
    file_type=["pdf"],
    key=f"chat_input_{active_chat_id}",
    disabled=pending_interrupt is not None,
)
if submission:
    prompt = submission.text.strip()
    uploaded_pdf = submission.files[0] if submission.files else None
    indexing_succeeded = True

    if uploaded_pdf is not None:
        temporary_pdf_path = None
        try:
            # PyPDFLoader expects a filesystem path, so write the upload temporarily.
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temporary_pdf:
                temporary_pdf.write(uploaded_pdf.getvalue())
                temporary_pdf_path = temporary_pdf.name
            with st.status("Indexing PDF for this chat...", expanded=True) as upload_status:
                ingest_rag_document(
                    temporary_pdf_path,
                    active_chat_id,
                    source_name=uploaded_pdf.name,
                )
                upload_status.update(
                    label=f"Indexed {uploaded_pdf.name} for this chat",
                    state="complete",
                )
            st.success(f"{uploaded_pdf.name} is ready for questions in this chat.")
        except Exception as error:
            indexing_succeeded = False
            st.error(f"PDF indexing failed: {error}")
        finally:
            if temporary_pdf_path and os.path.exists(temporary_pdf_path):
                os.remove(temporary_pdf_path)

    if prompt and indexing_succeeded:
        with st.chat_message("user"):
            st.markdown(prompt)
    chat = st.session_state.chat_sessions[active_chat_id]
    if prompt and chat["title"] == "New chat":
        title = prompt.strip().replace("\n", " ")
        chat["title"] = f"{title[:40]}..." if len(title) > 40 else title
    if prompt and indexing_succeeded:
        try:
            with st.chat_message("assistant"):
                tool_names = {}
                response_placeholder = st.empty()
                with st.status("Thinking...", expanded=True) as tool_status:
                    def response_chunks():
                        """Stream model text while updating the visible tool status."""
                        for message_chunk, metadata in chat_bot.stream(
                            {"messages": [HumanMessage(content=prompt)]},
                            config=config,
                            stream_mode="messages",
                        ):
                            metadata = metadata or {}
                            node_name = metadata.get("langgraph_node")
                            call_chunks = getattr(message_chunk, "tool_call_chunks", None) or []
                            if call_chunks:
                                tool_status.update(
                                    label="Preparing tool call...",
                                    state="running",
                                )
                                for call_chunk in call_chunks:
                                    tool_index = call_chunk.get("index", 0)
                                    name_fragment = call_chunk.get("name")
                                    if name_fragment:
                                        tool_names[tool_index] = (
                                            tool_names.get(tool_index, "") + name_fragment
                                        )
                                        tool_status.update(
                                            label=f"Calling tool: {tool_names[tool_index]}",
                                            state="running",
                                        )
                            else:
                                for tool_call in getattr(message_chunk, "tool_calls", None) or []:
                                    tool_name = tool_call.get("name")
                                    if tool_name:
                                        tool_status.update(
                                            label=f"Calling tool: {tool_name}",
                                            state="running",
                                        )

                            tool_name = getattr(message_chunk, "name", None)
                            if tool_name and node_name == "tools":
                                tool_status.update(
                                    label=f"Finished tool: {tool_name}",
                                    state="running",
                                )
                            elif node_name == "chat_node" and not call_chunks:
                                tool_status.update(
                                    label="Generating response...",
                                    state="running",
                                )

                            if (
                                node_name == "chat_node"
                                and isinstance(message_chunk.content, str)
                                and message_chunk.content
                            ):
                                yield message_chunk.content

                    response_placeholder.write_stream(response_chunks())
                    status_label = (
                        "Waiting for approval"
                        if get_pending_interrupt(config) is not None
                        else "Response complete"
                    )
                    tool_status.update(label=status_label, state="complete")
            chat["updated_at"] = datetime.now(timezone.utc).isoformat()
        except Exception as error:
            st.error(f"Chat failed: {error}")
    elif uploaded_pdf is not None and indexing_succeeded:
        st.rerun()

pending_interrupt = get_pending_interrupt(config)
if pending_interrupt is not None:
    # Resuming the graph with either button value completes the interrupted tool.
    with st.chat_message("assistant"):
        st.warning(f"Approval required: {pending_interrupt}")
        yes_column, no_column = st.columns(2)
        if yes_column.button("Yes", key=f"hitl_yes_{active_chat_id}", type="primary"):
            decision = "yes"
        elif no_column.button("No", key=f"hitl_no_{active_chat_id}"):
            decision = "no"
        else:
            decision = None

        if decision is not None:
            try:
                chat_bot.invoke(Command(resume=decision), config=config)
            except Exception as error:
                st.error(f"Could not resume the chat: {error}")
            else:
                st.session_state.chat_sessions[active_chat_id]["updated_at"] = (
                    datetime.now(timezone.utc).isoformat()
                )
                st.rerun()
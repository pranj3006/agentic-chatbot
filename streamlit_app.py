from datetime import datetime, timezone
from uuid import uuid4

import streamlit as st
from agentic_chatbot_backend import chat_bot
from langchain_core.messages import HumanMessage


st.set_page_config(page_title="Agentic ChatBot", page_icon=":material/chat:", layout="wide")


def create_chat() -> str:
    chat_id = str(uuid4())
    st.session_state.chat_sessions[chat_id] = {
        "title": "New chat",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    st.session_state.active_chat_id = chat_id
    return chat_id


st.session_state.setdefault("chat_sessions", {})
if not st.session_state.chat_sessions:
    create_chat()
if st.session_state.get("active_chat_id") not in st.session_state.chat_sessions:
    st.session_state.active_chat_id = next(iter(st.session_state.chat_sessions))

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
config = {"configurable": {"thread_id": active_chat_id}}
snapshot = chat_bot.get_state(config)
messages = snapshot.values.get("messages", []) if snapshot.values else []

st.title(st.session_state.chat_sessions[active_chat_id]["title"])
if not messages:
    st.caption("Ask me anything")

for chat_message in messages:
    role = "user" if isinstance(chat_message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(chat_message.content)

prompt = st.chat_input("Message Agentic ChatBot")
if prompt and prompt.strip():
    with st.chat_message("user"):
        st.markdown(prompt)
    chat = st.session_state.chat_sessions[active_chat_id]
    if chat["title"] == "New chat":
        title = prompt.strip().replace("\n", " ")
        chat["title"] = f"{title[:40]}..." if len(title) > 40 else title
    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chat_bot.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
        chat["updated_at"] = datetime.now(timezone.utc).isoformat()
        st.rerun()
    except Exception as error:
        st.error(f"Chat failed: {error}")
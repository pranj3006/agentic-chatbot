from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal,Annotated
from pydantic import BaseModel,Field
import operator
import os
import requests
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage,HumanMessage,BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

from langgraph.graph.message import add_messages
import sqlite3
from tools import tools_list

DB_URI = str(Path(__file__).resolve().with_name("agenticchatbot.db"))

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

bot_llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0,
    streaming=True,
    openai_api_key=OPENAI_API_KEY)

llm_with_tools = bot_llm.bind_tools(tools_list)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state:ChatState)-> dict:
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages":response}

db_con = sqlite3.connect(database=DB_URI,check_same_thread=False)
db_checkpoint = SqliteSaver(db_con)
tool_node = ToolNode(tools_list)

# checkpoint = MemorySaver() # for in memory server

graph = StateGraph(ChatState)

graph.add_node('chat_node',chat_node)
graph.add_node('tools',tool_node)

graph.add_edge(START,'chat_node')
graph.add_conditional_edges('chat_node',tools_condition)
graph.add_edge('tools','chat_node')
graph.add_edge('chat_node',END)

chat_bot = graph.compile(checkpointer=db_checkpoint)


def get_all_threads() -> set[str]:
    all_threads = set()
    threads = db_checkpoint.list(None)

    for t in threads:
        all_threads.add(t.config["configurable"]["thread_id"])

    return all_threads

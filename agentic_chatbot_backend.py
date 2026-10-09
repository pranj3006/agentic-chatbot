from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal,Annotated
from pydantic import BaseModel,Field
import operator
import os
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage,HumanMessage,BaseMessage
from langgraph.checkpoint.memory import MemorySaver

from langgraph.graph.message import add_messages


load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

bot_llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0,
    openai_api_key=OPENAI_API_KEY)



class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state:ChatState)-> dict:
    messages = state["messages"]
    response = bot_llm.invoke(messages)
    return {"messages":response}

checkpoint = MemorySaver()

graph = StateGraph(ChatState)

graph.add_node('chat_node',chat_node)

graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

chat_bot = graph.compile(checkpointer=checkpoint)



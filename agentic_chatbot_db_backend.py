"""LangGraph chatbot workflow, model configuration, and SQLite checkpoints.

The Streamlit interface imports ``chat_bot`` to run conversations and resume
approval interrupts. Explicit, quantified buy/sell requests are routed directly
to the stock-order tool, which pauses until the user approves or rejects.
"""

from langgraph.graph import StateGraph, START
from typing import TypedDict, Literal,Annotated
from pydantic import BaseModel,Field
import operator
import os
import requests
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage,HumanMessage,BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt,Command
from langgraph.graph.message import add_messages
import sqlite3
import re
from tools import execute_stock_order, tools_list
from rag import has_rag_index

DB_URI = str(Path(__file__).resolve().with_name("agenticchatbot.db"))

# Load local environment variables for the model and tools.
load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

bot_llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0,
    streaming=True,
    openai_api_key=OPENAI_API_KEY)

# Keep a general-purpose model binding and a binding that must call the order tool.
llm_with_tools = bot_llm.bind_tools(tools_list)
llm_with_order_tool = bot_llm.bind_tools(
    [execute_stock_order],
    tool_choice="execute_stock_order",
)

class ChatState(TypedDict):
    """State passed between graph nodes; messages are appended by ``add_messages``."""

    messages: Annotated[list[BaseMessage], add_messages]


def is_stock_order_request(messages: list[BaseMessage]) -> bool:
    """Identify a new, explicit buy/sell request that includes a quantity.

    Only a latest human message can trigger this route. That prevents the
    previous order request from being repeated when the graph resumes after an
    approval interrupt and processes the tool result.
    """
    latest_message = messages[-1] if messages else None
    if not isinstance(latest_message, HumanMessage) or not isinstance(
        latest_message.content, str
    ):
        return False

    content = latest_message.content
    is_action = re.search(
        r"^\s*(?:please\s+)?(?:buy|sell)\b"
        r"|\b(?:can|could|would)\s+you\s+(?:please\s+)?(?:buy|sell)\b"
        r"|\bI\s+(?:want|need)\s+to\s+(?:buy|sell)\b"
        r"|\bI(?:'d| would) like to\s+(?:buy|sell)\b"
        r"|\bplace\s+(?:an?\s+)?(?:buy|sell)\s+order\b",
        content,
        re.IGNORECASE,
    )
    has_quantity = re.search(
        r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b",
        content,
        re.IGNORECASE,
    )
    return bool(is_action and has_quantity)


def chat_node(state: ChatState, config: RunnableConfig) -> dict:
    """Ask the model for a response, supplying chat-specific tool instructions."""
    thread_id = config.get("configurable", {}).get("thread_id")
    pdf_available = isinstance(thread_id, str) and has_rag_index(thread_id)
    pdf_instructions = (
        "A PDF index EXISTS for this chat. For any question about the uploaded PDF, "
        "you MUST call rag_tool before answering. Never claim that no PDF is attached "
        "unless rag_tool reports that the index is unavailable."
        if pdf_available
        else "No PDF index exists for this chat. If asked about a PDF, explain that "
        "the user must attach it in this chat's message composer."
    )
    system_message = SystemMessage(
        content=f"""
        You are a helpful agentic chatbot with access to several tools.

        Tool Usage Instructions:
        - {pdf_instructions}
        - The current chat's thread ID is supplied automatically to the RAG tool; never ask the user for a thread ID.
        - Use 'search_tool' for current events or information requiring an internet search.
        - Use 'calculator' for simple calculations.
        - Use 'get_stock_price' for current stock prices.
        - Use 'get_weather_data' for current weather.
        - When the user requests a stock order, call 'execute_stock_order' directly.
          Do not ask conversationally for confirmation; the tool pauses for explicit
          user approval before executing the order.

        Answer general questions directly when no tool is required.
        Do not invent information from uploaded documents.
        After receiving a tool result, provide a clear and helpful final answer.
        """
    )
    messages = [system_message,*state["messages"]]
    order_requested = is_stock_order_request(state["messages"])
    # Force the order tool for explicit order intents so the approval interrupt runs.
    model = llm_with_order_tool if order_requested else llm_with_tools
    response = model.invoke(messages)
    return {"messages":[response]}

# Persist graph state between app reruns and across process restarts.
db_con = sqlite3.connect(database=DB_URI,check_same_thread=False)
db_checkpoint = SqliteSaver(db_con)
tool_node = ToolNode(tools_list)

# The graph alternates between model responses and any requested tool calls.
graph = StateGraph(ChatState)

graph.add_node('chat_node',chat_node)
graph.add_node('tools',tool_node)

graph.add_edge(START,'chat_node')
graph.add_conditional_edges('chat_node',tools_condition)
graph.add_edge('tools','chat_node')

chat_bot = graph.compile(checkpointer=db_checkpoint)


def get_all_threads() -> set[str]:
    """Return the thread IDs stored in the SQLite checkpoint database."""
    all_threads = set()
    threads = db_checkpoint.list(None)

    for t in threads:
        all_threads.add(t.config["configurable"]["thread_id"])

    return all_threads

# if __name__=="__main__":
#     print("Agentic Chatbot CLI \n")
#     print("Type exit to quite \n")

#     thread_id = "demo_thread2"
#     while True:
#         user_message = input("User Message: ")
#         # print("User Message : ",user_message)
#         if user_message.strip().lower() in ["exit","quit","bye"]:
#             break
#         config ={"configurable": {"thread_id":thread_id}}
#         message = {"messages":[HumanMessage(content=user_message)]}
#         response = chat_bot.invoke(message,config=config)

#         interrupts = response.get("__interrupt__",[])
#         if interrupts:
#             prompt_to_human = interrupts[0].value
#             print(f"HITL: {prompt_to_human}")
#             decision = input("Your decision: ").strip().lower()

#             response = chat_bot.invoke(
#                 Command(resume=decision),
#                 config=config
#             )

#         print("AI : ",response["messages"][-1].content)
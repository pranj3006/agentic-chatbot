from agentic_chatbot_backend import chat_bot
from langchain_core.messages import SystemMessage,HumanMessage,BaseMessage

thread_id = "1"

while True:
    user_message = input("Your Input Here: ")
    print("User Message : ",user_message)
    if user_message.strip().lower() in ["exit","quit","bye"]:
        break
    config ={"configurable": {"thread_id":thread_id}}
    message = {"messages":[HumanMessage(content=user_message)]}
    response = chat_bot.invoke(message,config=config)
    print("AI : ",response["messages"][-1].content)
import streamlit as st
from contextlib import redirect_stdout
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

st.title("General Purpose Chatbot")
st.write("Ask any general purpose questions")

# st.chat_message("User").write("Hello Model !")
# st.chat_message("Assistant").write("Hello User !")

# Maintain all messag ehistory per session

# for session info 
if "state" not in st.session_state:
    st.session_state.store = {}


# for messages 
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display all previous messages 
for msg in st.session_state.messages:
    role, text = msg 
    if role == "user":
        st.chat_message("user").write(text)
    else :
        st.chat_message("assistant").write(text)


def get_history(session_id):
    if session_id not in  st.session_state.store:
         st.session_state.store[session_id] = InMemoryChatMessageHistory()
    return  st.session_state.store[session_id]


# Initialize model
model = ChatOllama(model="llama3.2")

# create prompt
prompt = ChatPromptTemplate.from_messages([
    ("system" , "You are a chatbot answer user query precisly "),
    MessagesPlaceholder(variable_name="place_where_history_added"),
    ("human", "{input}"),
])

# create a chain
chat_chain = prompt | model | StrOutputParser()

# create chain with memory 
memory_chain = RunnableWithMessageHistory(
    chat_chain,
    get_history,
    input_message_key = "input",
    history_messages_key = "place_where_history_added"
)

session_id = "s1"

# Take user input from input box 
user_input = st.chat_input("Enter your message here ...")

if user_input :
    # Display user input as user message
    st.chat_message("user").write(user_input)

    # append user question to streamlist messages list 
    st.session_state.messages.append(("user",user_input ))

    # run chain 
    result = memory_chain.invoke({"input":st.session_state.messages},
                                 config = {"configurable" : {"session_id" : session_id}})
    
    # append model response  to streamlist messages list 
    st.session_state.messages.append(("assistant",result ))

    # display as AI response
    st.chat_message("assistant").write(result)
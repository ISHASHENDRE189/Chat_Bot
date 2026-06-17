from langgraph.graph import StateGraph, END   # StateGraph builds our workflow; END marks the finish
from langchain_ollama import ChatOllama        # lets us talk to a local Ollama model
from typing import TypedDict                   # lets us define a typed dictionary for state

# TypedDict defines the "shape" of data that flows through our graph.
class ChatState(TypedDict):
    message  : str    # the user's original input text
    response : str    # the AI's final reply (filled in later)
    is_joke  : bool   # FIX: was "isjocke" — now consistent everywhere

model = ChatOllama(model="llama3.2")  # load the llama3.2 model running locally via Ollama

# Node 1: checks whether the user is asking for a joke
def joke_checking_node(state: ChatState):   # FIX: renamed from jock_checking_node
    print("[Node 1] Checking if the user is asking for a joke...")
    user_text = state["message"]
    ai_msg = model.invoke(
        f"""Answer True only if the user is asking for a joke, else answer False.
User question: {user_text}
Answer in only one word."""
    )
    answer = str(ai_msg.content).strip()
    print(f"[Node 1] Model replied: {answer}")
    if answer == "True":
        state["is_joke"] = True    # FIX: was state["isjock"]
    else:
        state["is_joke"] = False   # FIX: was "isjock" and "Flase"
        state["response"] = "Sorry, I can only answer joke-related questions!"
    return state

# Node 2: generates the actual joke reply
def generate_reply(state: ChatState):
    print("[Node 2] Generating a joke response...")
    user_text = state["message"]
    ai_msg    = model.invoke(user_text)
    state["response"] = ai_msg.content
    return state

# Decides which node to visit after joke_check
def decide_flow(state: ChatState):
    print("[Edge] Deciding flow based on is_joke...")
    print(f"[Edge] is_joke = {state['is_joke']}")   # FIX: was state["isjoke"]
    # FIX: logic was inverted — joke → generate reply; not a joke → end early
    if state["is_joke"]:
        return "lama_model"
    return "END"

graph = StateGraph(ChatState)
graph.add_node("joke_check",  joke_checking_node)  # FIX: function name now matches
graph.add_node("lama_model", generate_reply)
graph.add_conditional_edges(
    "joke_check",
    decide_flow,
    {"lama_model": "lama_model", "END": END}
)
graph.add_edge("lama_model", END)
graph.set_entry_point("joke_check")
compiled_graph = graph.compile()

while True:
    user_input = input("\n\tEnter Input : ")
    print("\n" + "*"*20 + "\n\tUser : ", user_input)
    if user_input == "q":
        break
    result: ChatState = compiled_graph.invoke({"message": user_input})
    print("\n" + "*"*20 + "\n\t AI : ", result["response"])
    print("Full State Info : ", result)


""" User input → joke_check → [is a joke?]
                              ├─ YES → lama_model → generate joke → END
                              └─ NO  → set "sorry" message → END  """

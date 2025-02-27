import os
import json
import uuid
import streamlit as st
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langgraph.graph import StateGraph, END
import redis
from macro import CHROMA_DB_PATH, MODEL_NAME

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = os.getenv("REDIS_PORT")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

# Redis client
redis_client = redis.StrictRedis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    decode_responses=True
)

# Define the state for the LangGraph workflow
class ChatbotState(TypedDict):
    query: str
    keywords: str
    domain: str
    response: str

# Initialize shared components
llm = ChatOpenAI(model="gpt-4", temperature=0.7, api_key=OPENAI_API_KEY)
search_tool = DuckDuckGoSearchRun()
embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
vector_store = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)

# Define nodes for the LangGraph workflow
def extract_keywords(state: ChatbotState) -> ChatbotState:
    keyword_prompt = PromptTemplate(
        input_variables=["query"],
        template="Extract the main keywords from this text (short list): '{query}'"
    )
    keyword_chain = keyword_prompt | llm
    keywords = keyword_chain.invoke({"query": state["query"]}).content.strip()
    return {"query": state["query"], "keywords": keywords, "domain": state.get("domain", ""), "response": state.get("response", "")}

def classify_domain(state: ChatbotState) -> ChatbotState:
    domain_prompt = PromptTemplate(
        input_variables=["query"],
        template="Classify the domain of this question: '{query}' (e.g., Economics, Mathematics, or Other)"
    )
    domain_chain = domain_prompt | llm
    domain = domain_chain.invoke({"query": state["query"]}).content.strip()
    return {"query": state["query"], "keywords": state["keywords"], "domain": domain, "response": state.get("response", "")}

def process_general_query(state: ChatbotState) -> ChatbotState:
    search_results = search_tool.run(state["query"])
    response_prompt = PromptTemplate(
        input_variables=["query", "results"],
        template="Answer the query based on the following search results and use following language: '{query}'\n\n{results}"
    )
    response_chain = response_prompt | llm
    response = response_chain.invoke({"query": state["query"], "results": search_results}).content.strip()
    return {"query": state["query"], "keywords": state["keywords"], "domain": state["domain"], "response": response}

def process_specialized_query(state: ChatbotState) -> ChatbotState:
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=False
    )
    response = qa_chain.run(state["query"])
    return {"query": state["query"], "keywords": state["keywords"], "domain": state["domain"], "response": response}

# Define the LangGraph workflow
workflow = StateGraph(ChatbotState)

# Add nodes
workflow.add_node("extract_keywords", extract_keywords)
workflow.add_node("classify_domain", classify_domain)
workflow.add_node("process_general_query", process_general_query)
workflow.add_node("process_specialized_query", process_specialized_query)

# Define edges
workflow.set_entry_point("extract_keywords")
workflow.add_edge("extract_keywords", "classify_domain")

# Conditional edge based on domain
def route_domain(state: ChatbotState) -> str:
    if state["domain"] in ["Economics", "Mathematics"]:
        return "process_specialized_query"
    return "process_general_query"

workflow.add_conditional_edges(
    "classify_domain",
    route_domain,
    {
        "process_general_query": "process_general_query",
        "process_specialized_query": "process_specialized_query"
    }
)

workflow.add_edge("process_general_query", END)
workflow.add_edge("process_specialized_query", END)

# Compile the graph
graph = workflow.compile()

# Function to process user input using the graph
def process_query(user_input: str) -> dict:
    initial_state = {"query": user_input, "keywords": "", "domain": "", "response": ""}
    result = graph.invoke(initial_state)
    return {"keywords": result["keywords"], "response": result["response"]}

# Save conversation to Redis
def save_conversation(session_id, messages):
    title = messages[0]['content'] if messages else "Untitled Session"
    redis_client.set(session_id, json.dumps({"title": title, "messages": messages}))
    redis_client.zadd("session_order", {session_id: int(uuid.uuid4().int % 1e9)})

# Load conversation from Redis
def load_conversation(session_id):
    data = redis_client.get(session_id)
    redis_client.zadd("session_order", {session_id: int(uuid.uuid4().int % 1e9)})
    return json.loads(data)["messages"] if data else []

# Delete conversation from Redis
def delete_conversation(session_id):
    redis_client.delete(session_id)
    redis_client.zrem("session_order", session_id)
    if "session_id" in st.session_state and st.session_state.session_id == session_id:
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
    st.rerun()

# Get all sessions sorted by last used
def get_all_sessions():
    session_ids = redis_client.zrevrange("session_order", 0, -1)
    return [{"id": session_id, "title": json.loads(redis_client.get(session_id)).get("title", "Untitled Session")}
            for session_id in session_ids]

# Streamlit configuration
st.set_page_config(page_title="Chatbot", page_icon="🤖", layout="centered")
st.title("Chatbot AI - BOOK RECOMMENDATION")

# Sidebar for sessions
with st.sidebar:
    st.header("📂 Previous Sessions")
    if st.button("➕ New Session"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())

    for session in get_all_sessions():
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            if st.button(session["title"]):
                st.session_state.messages = load_conversation(session["id"])
                st.session_state.session_id = session["id"]
        with col2:
            if st.button("❌", key=session["id"]):
                delete_conversation(session["id"])

# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Enter your question...")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Process query using the LangGraph workflow
    result = process_query(user_input)

    # Format and display the response
    response_text = f"**Keywords**: {result['keywords']}\n\n**Response**: {result['response']}"
    st.chat_message("assistant").markdown(response_text)
    st.session_state.messages.append({"role": "assistant", "content": response_text})

    # Save to Redis
    session_id = st.session_state.get("session_id", str(uuid.uuid4()))
    st.session_state.session_id = session_id
    save_conversation(session_id, st.session_state.messages)
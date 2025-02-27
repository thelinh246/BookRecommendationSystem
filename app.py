"""
2/27/2024
"""

import os
import json
import uuid
import streamlit as st
from dotenv import load_dotenv
import openai
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchResults
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

# Agent class with specified pipeline
class ChatbotAgent:
    """
        Agent to process user queries with keyword extraction and domain-specific responses.
    """
    def __init__(self):
        self.search_tool = DuckDuckGoSearchResults()
        self.client = openai.Client(api_key=OPENAI_API_KEY)

    def extract_keywords(self, prompt):
        """Extract main keywords from the user prompt using OpenAI.
        """
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": f"Trích xuất từ khóa chính từ văn bản sau (danh sách ngắn gọn): '{prompt}'"}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()

    def classify_domain(self, prompt):
        """"Classify the domain of the user prompt
        """
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": f"Xác định lĩnh vực của câu hỏi sau: '{prompt}' (ví dụ: Kinh tế, Toán học, hoặc Khác)"}],
            temperature=0
        )
        return response.choices[0].message.content.strip()

    def retrieve_from_db(self, query, db_path, model_name, top_k=5):
        """Retrieve relevant documents from the vector database
        """
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        db = Chroma(persist_directory=db_path, embedding_function=embeddings)
        return db.similarity_search(query, k=top_k)

    def generate_response(self, prompt):
        """Generate a response to the user query using OpenAI
        """
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()

    def process_query(self, user_input, db_path=CHROMA_DB_PATH, model_name=MODEL_NAME):
        """Process user query by extracting keywords and generating a domain-specific response
        """
    # Step 1: Extract keywords from user prompt
        keywords = self.extract_keywords(user_input)
        
        # Step 2: Analyze the domain/topic
        domain = self.classify_domain(user_input)
        
        # Step 3: Process based on domain
        if domain not in ["Kinh tế", "Toán học"]:
            search_results = self.search_tool.run(user_input)
            prompt = f"Answer the query based on the following search results: '{user_input}'\n\n{search_results}"
            response = self.generate_response(prompt)
        else:
            # Combine user_input and keywords for a more focused query
            enhanced_query = f"{user_input}"
            results = self.retrieve_from_db(enhanced_query, db_path, model_name)
            combined_content = "\n\n".join([result.page_content for result in results]) if results else "No relevant data found in vector database."
            prompt = f"Based on the following documents, answer the query: '{user_input}'\n\n{combined_content}"
            response = self.generate_response(prompt)
        
        return {"keywords": keywords, "response": response}

# Save conversation to Redis
def save_conversation(session_id, messages):
    """Save the conversation to Redis with a title and message history
    """
    title = messages[0]['content'] if messages else "Untitled Session"
    redis_client.set(session_id, json.dumps({"title": title, "messages": messages}))
    redis_client.zadd("session_order", {session_id: int(uuid.uuid4().int % 1e9)})

# Load conversation from Redis
def load_conversation(session_id):
    """Load a conversation from Redis by session ID
    """
    data = redis_client.get(session_id)
    redis_client.zadd("session_order", {session_id: int(uuid.uuid4().int % 1e9)})  # Update last used
    return json.loads(data)["messages"] if data else []

# Delete conversation from Redis
def delete_conversation(session_id):
    """Delete a conversation from Redis and update session state
    """
    redis_client.delete(session_id)
    redis_client.zrem("session_order", session_id)
    if "session_id" in st.session_state and st.session_state.session_id == session_id:
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
    st.rerun()

# Get all sessions sorted by last used
def get_all_sessions():
    """Retrieve all sessions from Redis sorted by last used
    """
    session_ids = redis_client.zrevrange("session_order", 0, -1)
    return [{"id": session_id, "title": json.loads(redis_client.get(session_id)).get("title", "Untitled Session")} for session_id in session_ids]

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

# Initialize the agent
agent = ChatbotAgent()

user_input = st.chat_input("Enter your question...")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Process query using the agent
    result = agent.process_query(user_input)

    # Format and display the response
    response_text = f"**Keywords**: {result['keywords']}\n\n**Response**: {result['response']}"
    st.chat_message("assistant").markdown(response_text)
    st.session_state.messages.append({"role": "assistant", "content": response_text})

    # Save to Redis
    session_id = st.session_state.get("session_id", str(uuid.uuid4()))
    st.session_state.session_id = session_id
    save_conversation(session_id, st.session_state.messages)
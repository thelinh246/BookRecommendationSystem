import streamlit as st
import google.generativeai as genai  
from dotenv import load_dotenv
import os
import openai
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from macro import CHROMA_DB_PATH, MODEL_NAME

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Retrieve from VectorDB
def retrieve_from_db(query, db_path, model_name, top_k=1):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    return db.similarity_search(query, k=top_k)

# Get response from OpenAI
def get_openai_response(prompt):
    client = openai.Client(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

# Streamlit configuration
st.set_page_config(page_title="Chatbot", page_icon="🤖", layout="centered")
st.title("Chatbot AI - BOOK RECOMMENDATION")

# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Nhập câu hỏi của bạn...")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    results = retrieve_from_db(user_input, CHROMA_DB_PATH, MODEL_NAME)
    combined_content = "\n\n".join([result.page_content for result in results])
    prompt = f"Based on the following documents, answer the query: '{user_input}'\n\n{combined_content}"
    response = get_openai_response(prompt)
    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
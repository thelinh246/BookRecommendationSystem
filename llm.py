"""
2/21/2025
"""

import openai
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
from macro import CHROMA_DB_PATH, MODEL_NAME
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ========== RETRIEVE FROM VECTORDB ==========
def retrieve_from_db(query, db_path, model_name, top_k=1):
    """Retrieves relevant documents from a Chroma vector database using similarity search.

    Args:
        query (str): User's query.
        db_path (str): Path to the Chroma vector database.
        model_name (str): Name of the Hugging Face embedding model used for vectorization.
        top_k (int, optional): Number of top results to return. Defaults to 1.

    Returns:
        List[Document]: List of Document objects containing retrieved information.
    """
    # Load embedding model
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    # Load VectorDB from saved folder
    db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    # Retrieve nearest results
    results = db.similarity_search(query, k=top_k)
    return results

# ========== SEND TO OPENAI ==========
def get_openai_response(prompt):
    """Sends prompt to OpenAI API and returns the response."""
    client = openai.Client(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

# ========== MAIN ==========
if __name__ == "__main__":
    # Receive prompt from users
    query = "I want to find information about Pipeline Configuration for chatbot"
    # Retrieve
    print("⚡ Retrieving data from VectorDB...")
    results = retrieve_from_db(query, CHROMA_DB_PATH, MODEL_NAME)
    for i, result in enumerate(results, start=1):
        file_name = result.metadata.get('source', 'Unknown source')
        print(f"\n🔸 Results {i} (Book: {file_name},\n Content:\n {result.page_content})")
    # Combine content of all documents
    combined_content = "\n\n".join([result.page_content for result in results])
    prompt = f"Based on the following documents, answer the query: '{query}'\n\n{combined_content}"

    # Get OpenAI response
    print("💡 Sending data to OpenAI LLM...")
    response = get_openai_response(prompt)

    # Print result
    print("\n💡 Final response:")
    print(response)
    print(f"Source: {file_name[4:]}" )
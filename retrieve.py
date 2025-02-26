from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from macro import CHROMA_DB_PATH, MODEL_NAME


# ========== RETRIEVE FROM VECTORDB ==========


def retrieve_from_db(query, db_path, model_name, top_k=5):
    # Load embedding model
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # Load VectorDB from saved folder
    db = Chroma(persist_directory=db_path, embedding_function=embeddings)

    # Retrieve nearest results
    results = db.similarity_search(query, k=top_k)
    return results


# ========== MAIN ==========
if __name__ == "__main__":
    # Receive prompt from users
    query = input()

    # Retrieve
    print("⚡ Retrieving data from VectorDB...")
    results = retrieve_from_db(query, CHROMA_DB_PATH, MODEL_NAME)

    # Print result
    print("\n💡 Retrieval result:")
    for i, result in enumerate(results, start=1):
        file_name = result.metadata.get("source", "Unknown source")
        print(
            f"\n🔸 Results {i} (Book: {file_name},\n Content:\n {result.page_content})"
        )

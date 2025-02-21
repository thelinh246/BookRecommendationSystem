"""
2/21/2025
"""

import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from macro import PDF_FOLDER, CHROMA_DB_PATH, MODEL_NAME


# ========== STEP 1: LOAD PDF FILES ==========
def load_pdf_files(folder_path):
    """Loads all PDF files from a specified folder and extracts their content.

    Args:
        folder_path (str): Path to the folder containing PDF files.

    Returns:
        List[Document]: A list of Document objects containing the content of each PDF file.
    """
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    documents = []
    for file in pdf_files:
        loader = PyPDFLoader(file)
        documents.extend(loader.load())
    return documents

# ========== STEP 2: TEXT SPLITTING ==========
def split_documents(documents):
    """Splits a list of documents into smaller chunks using RecursiveCharacterTextSplitter.

    Args:
        documents (List[Document]): A list of Document objects to be split.

    Returns:
        List[Document]: A list of Document objects after splitting into smaller chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=384,
        chunk_overlap=64,
        separators=["\n", " ", ""]
    )
    return text_splitter.split_documents(documents)

# ========== STEP 3: EMBEDDING & VECTORIZATION ==========
def embed_and_store(chunks, db_path, model_name):
    """Embeds document chunks and stores them in a Chroma vector database.

    Args:
        chunks (List[Document]): List of document chunks to be embedded.
        db_path (str): Path where the Chroma vector database will be saved.
        model_name (str): Name of the Hugging Face embedding model used for vectorization.
    """
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    db = Chroma.from_documents(chunks, embeddings, persist_directory=db_path)
    db.persist()
    print("✅ VectorDB successfully saved at:", db_path)

# ========== MAIN PIPELINE ==========
def main():
    """Main function to load, chunk, and vectorize PDF files into ChromaDB."""
    print("📂 Loading PDF files...")
    documents = load_pdf_files(PDF_FOLDER)
    print("✂️ Chunking text...")
    chunks = split_documents(documents)
    print("🧠 Vectorizing and storing into ChromaDB...")
    embed_and_store(chunks, CHROMA_DB_PATH, MODEL_NAME)

if __name__ == "__main__":
    main()

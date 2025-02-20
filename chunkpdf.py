import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from macro import PDF_FOLDER, CHROMA_DB_PATH, MODEL_NAME


# ========== STEP 1: LOAD PDF FILES ==========
def load_pdf_files(folder_path):
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    documents = []
    for file in pdf_files:
        loader = PyPDFLoader(file)
        documents.extend(loader.load())
    return documents

# ========== STEP 2: TEXT SPLITTING ==========
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=384,
        chunk_overlap=64,
        separators=["\n", " ", ""]
    )
    return text_splitter.split_documents(documents)

# ========== STEP 3: EMBEDDING & VECTORIZATION ==========
def embed_and_store(chunks, db_path, model_name):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    db = Chroma.from_documents(chunks, embeddings, persist_directory=db_path)
    db.persist()
    print("✅ VectorDB successfully saved at:", db_path)

# ========== MAIN PIPELINE ==========
def main():
    print("📂 Loading PDF files...")
    documents = load_pdf_files(PDF_FOLDER)
    
    print("✂️ Chunking text...")
    chunks = split_documents(documents)
    
    print("🧠 Vectorizing and storing into ChromaDB...")
    embed_and_store(chunks, CHROMA_DB_PATH, MODEL_NAME)

if __name__ == "__main__":
    main()

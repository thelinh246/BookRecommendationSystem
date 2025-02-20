from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ========== CONFIGURATION ==========
CHROMA_DB_PATH = "vectordb"
MODEL_NAME = "intfloat/multilingual-e5-large"  # Model embedding

# ========== RETRIEVE FROM VECTORDB ==========
def retrieve_from_db(query, db_path, model_name, top_k=1):
    # Tải embedding model
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    # Tải VectorDB từ thư mục đã lưu
    db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    
    # Truy xuất các kết quả gần nhất
    results = db.similarity_search(query, k=top_k)
    return results

# ========== MAIN ==========
if __name__ == "__main__":
    # Nhập prompt từ người dùng
    query = "Pipeline Configuration"
    
    # Thực hiện truy xuất
    print("⚡ Đang truy xuất dữ liệu từ VectorDB...")
    results = retrieve_from_db(query, CHROMA_DB_PATH, MODEL_NAME)
    
    # In kết quả
    print("\n💡 Kết quả truy xuất:")
    for i, result in enumerate(results, start=1):
        # Lấy tên file từ metadata
        file_name = result.metadata.get('source', 'Không rõ nguồn')
        print(f"\n🔸 Kết quả {i} (Sách: {file_name}, Nội dung: {result.page_content})")

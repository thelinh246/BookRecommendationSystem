📚 PDF Information Retrieval using ChromaDB and LangChain

📝 Overview
This project is designed to extract, process, and store information from PDF documents using advanced embedding models and vector databases. Users can query the system to retrieve relevant information based on semantic similarity.

Key Features:
    Load and extract content from PDF files
    Split large documents into manageable chunks
    mbed text chunks using Hugging Face models
    Store embeddings in ChromaDB
    Retrieve top-matching documents using similarity search

🧩 Technologies Used
    Programming Language: Python 3.12.5
    PDF Loader: LangChain Community (PyPDFLoader)
    Text Splitter: RecursiveCharacterTextSplitter
    Embedding Model: Hugging Face (intfloat/multilingual-e5-large)
    Vector Database: ChromaDB

💾 Environment Setup

1. Prerequisites:
    Python 3.12.5 or higher
    Pip installed

2. Clone the Repository:
    git clone https://github.com/thelinh246/BookRecommendationSystem.git
    cd BookRecommendationSystem

3. Create a Virtual Environment:
    python -m venv venv
    venv\Scripts\activate (Window)

4. Install Dependencies:
    pip install -r requirements.txt
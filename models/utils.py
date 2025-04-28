import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv

# Đọc biến môi trường từ file .env
load_dotenv()

def get_env_variable(variable_name):
    """Lấy biến môi trường từ file .env"""
    value = os.getenv(variable_name)
    if not value:
        raise ValueError(f"Missing environment variable: {variable_name}")
    return value

def preprocess_text(text):
    """Tiền xử lý chuỗi văn bản: loại bỏ ký tự đặc biệt và chuyển về chữ thường."""
    text = text.lower()
    # Loại bỏ ký tự đặc biệt, dấu câu
    text = ''.join(e for e in text if e.isalnum() or e.isspace())
    return text

def load_data_from_csv(file_path):
    """Đọc dữ liệu từ tệp CSV."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File {file_path} không tồn tại!")
        raise

def normalize_interaction(row):
    """Chuẩn hóa dữ liệu tương tác người dùng."""
    view_score = 1 if row['views'] > 0 else 0
    download_score = 3 if row['downloads'] > 0 else 0
    time_score = min(row['time_spent'] // 60, 5)
    total_score = view_score + download_score + time_score
    return min(total_score, 5)

def create_faiss_index(embeddings):
    """Tạo chỉ mục FAISS từ vector embeddings."""
    index = faiss.IndexHNSWFlat(embeddings.shape[1], 32)
    index.add(np.array(embeddings))
    return index

def generate_embeddings(documents):
    """Tạo vector embeddings từ tài liệu bằng Sentence-BERT."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(documents)

def save_embeddings_to_faiss(embeddings, index_file_path):
    """Lưu embeddings vào file FAISS."""
    faiss.write_index(embeddings, index_file_path)

def load_embeddings_from_faiss(index_file_path):
    """Tải embeddings từ file FAISS."""
    return faiss.read_index(index_file_path)

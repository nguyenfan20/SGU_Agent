
<p align="center">
  <img src="static/img/logo.png" alt="Logo" width="120"/>
</p>

# SGU_Agent: Hệ thống Gợi ý Tài liệu Học tập Thông minh SGU

## 🎉 Giới thiệu

📅 Đây là dự án đồ án chuyên ngành xây dựng một hệ thống gợi ý tài liệu học tập cá nhân hóa, chính xác và minh bạch cho sinh viên Đại học Sài Gòn. Hệ thống sử dụng kết hợp các phương pháp Collaborative Filtering (CF), Content-Based Filtering (CBF), tìm kiếm ngữ nghĩa với **Sentence-BERT**, và sinh giải thích gợi ý bằng **mô hình ngôn ngữ T5 tiếng Việt**. Hệ thống được phát triển trên nền tảng Flask.

**🚀 Mục tiêu chính:**
- 🎯 Cung cấp gợi ý tài liệu học tập phù hợp dựa trên hồ sơ sinh viên, lịch sử tương tác và truy vấn tìm kiếm.
- ❄️ Khắc phục vấn đề Cold-start cho người dùng mới.
- 🖥️ Tích hợp giao diện web đơn giản cho phép sinh viên tương tác và nhận gợi ý kèm giải thích rõ ràng.
- 🚀 Sử dụng công nghệ tiên tiến (Embedding, FAISS, LLM) để nâng cao chất lượng gợi ý và trải nghiệm người dùng.

## 🏛️ Kiến trúc Hệ thống

Hệ thống được xây dựng theo kiến trúc microservice đơn giản, với các module chính:

1. 📝 **Nhập liệu & Tiền xử lý:** Đọc dữ liệu từ các file CSV (`documents.csv`, `student_profiles.csv`, `interactions.csv`), chuẩn hóa và làm giàu metadata tài liệu.
2. 🔍 **Retrieval (Truy xuất):**
    *   Sử dụng mô hình **Sentence-BERT** (ví dụ: `sentence-transformers/all-MiniLM-L6-v2`) để tạo vector embedding cho tài liệu (dựa trên tiêu đề, tóm tắt, từ khóa, môn học, ngành liên quan) và truy vấn.
    *   Xây dựng chỉ mục **FAISS (Index HNSWFlat)** để lưu trữ và tìm kiếm nhanh các vector tài liệu gần nhất (semantic search).
3. 🧩 **Candidate Generation & Combination:**
    *   Tạo các ứng viên tài liệu từ các nguồn:
        *   CBF (Query): Tài liệu có vector gần với vector truy vấn (sử dụng FAISS search).
        *   CBF (Profile): Tài liệu có vector gần với vector được tạo từ thông tin hồ sơ sinh viên (ngành, năm học) (sử dụng FAISS search với query từ profile).
        *   CF (Collaborative Filtering): Tài liệu được quan tâm bởi các sinh viên có lịch sử tương tác tương tự (sử dụng Jaccard Similarity trên tập tài liệu đã tương tác).
    *   Kết hợp các ứng viên từ các nguồn khác nhau, loại bỏ trùng lặp và xếp hạng dựa trên điểm số heuristic ban đầu.
4. 🧠 **Explanation Generation (Sinh Giải thích):**
    *   Sử dụng **mô hình ngôn ngữ T5 tiếng Việt (VietAI/vit5-base)** (hoặc template động/tĩnh nếu T5 gặp lỗi phông).
    *   Nhận thông tin của từng tài liệu gợi ý hàng đầu, truy vấn ban đầu, và thông tin người dùng làm ngữ cảnh.
    *   Sinh ra một đoạn giải thích cụ thể lý do tại sao tài liệu đó phù hợp với truy vấn và/hoặc hồ sơ sinh viên.
5. 📚 **Quản lý Lịch sử:** Lưu trữ lịch sử truy vấn của từng sinh viên vào database **SQLite**.
6. 🌐 **Giao diện Web (Flask):**
    *   Ứng dụng Flask phục vụ các route (đăng nhập/xuất, gợi ý, API lịch sử, API danh sách tài liệu).
    *   Giao diện người dùng dạng chat (HTML/CSS/JS) cho phép nhập truy vấn, hiển thị tin nhắn chat (truy vấn của sinh viên, phản hồi gợi ý từ AI), hiển thị lịch sử truy vấn trong sidebar trái, và danh sách tài liệu có thể tìm kiếm trong sidebar phải.
    *   Sử dụng Fetch API (JavaScript) để giao tiếp không đồng bộ với backend.

## 🗂️ Cấu trúc Thư mục
```
SGU_Agent/
├── app.py
├── history_manager.py
├── models/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── explanation.py
│   ├── recommendation.py
│   └── retrieval.py
├── data/
│   ├── documents.csv            
│   ├── student_profiles.csv
│   ├── interactions.csv
│   └── query_history.db        
├── templates/
│   ├── index.html              
│   └── login.html              
├── static/
│   ├── style.css               
│   └── img/
│       └── logo.png            
├── .env                        
└── requirements.txt            
└── README.models
```

## ⚙️ Yêu cầu Hệ thống

- Python 3.7+
- Kết nối Internet

## 🛠️ Cài đặt

1. **Clone Repository** hoặc tải xuống mã nguồn.
2. **Cài đặt Thư viện Python:**
    ```bash
    python -m venv venv
    venv\Scripts\activate   # Windows
    source venv/bin/activate  # macOS/Linux
    pip install -r requirements.txt
    ```
3. **Chuẩn bị Dữ liệu:** Đặt các file CSV vào thư mục `data/`.
4.  **Cấu hình Secret Key:**
    *   Tạo file `.env` ở thư mục gốc của dự án (nếu chưa có).
    *   Thêm dòng sau vào file `.env`:
        ```dotenv
        SECRET_KEY=A_RANDOM_LONG_AND_COMPLEX_STRING_FOR_SESSION_SECURITY
        ```
    *   Thay thế placeholder bằng một chuỗi ngẫu nhiên, dài, phức tạp của riêng bạn.
    *   (Quan trọng) **KHÔNG** chia sẻ file `.env`. Nếu dùng Git, thêm `.env` vào `.gitignore`.

## 🚀 Chạy Ứng dụng

```bash
python app.py
```

Mở trình duyệt và truy cập `http://127.0.0.1:5000/`.

## 🎯 Sử dụng Hệ thống

- Đăng nhập bằng User ID.
- Nhập truy vấn và nhận tài liệu gợi ý cùng giải thích.
- Xem lịch sử truy vấn và tìm kiếm tài liệu.

## 🧪 Thử nghiệm

- Thử nhiều truy vấn khác nhau.
- Dùng User ID khác nhau để kiểm thử.

## 🤝 Đóng góp

Nếu bạn có ý tưởng hoặc muốn đóng góp vào dự án này, vui lòng liên hệ [nguyenphan201203@gmail.com].

## 📜 Giấy phép

[MIT License](LICENSE.txt)

---

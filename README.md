
<p align="center">
  <img src="static/img/logo.png" alt="Logo" width="120"/>
</p>

# SGU_Agent: Hệ thống Gợi ý Tài liệu Học tập Thông minh SGU

## 🎉 Giới thiệu

📅 Đây là dự án đồ án chuyên ngành xây dựng một hệ thống gợi ý tài liệu học tập cá nhân hóa, chính xác và minh bạch cho sinh viên Đại học Sài Gòn. Hệ thống sử dụng kết hợp các phương pháp hiện đại bao gồm Collaborative Filtering (CF), Content-Based Filtering (CBF), và Retrieval-Augmented Generation (RAG) trên nền tảng Flask.

**🚀 Mục tiêu chính:**
- 🎯 Cung cấp gợi ý tài liệu học tập phù hợp dựa trên hồ sơ sinh viên, lịch sử tương tác và truy vấn tìm kiếm.
- ❄️ Khắc phục vấn đề Cold-start cho người dùng mới.
- 🖥️ Tích hợp giao diện web đơn giản cho phép sinh viên tương tác và nhận gợi ý kèm giải thích rõ ràng.
- 🚀 Sử dụng công nghệ tiên tiến (Embedding, FAISS, LLM) để nâng cao chất lượng gợi ý và trải nghiệm người dùng.

## 🏛️ Kiến trúc Hệ thống

Hệ thống được xây dựng theo kiến trúc microservice đơn giản, với các module chính:

1. 📝 **Nhập liệu & Tiền xử lý:** Đọc dữ liệu từ các file CSV (tài liệu, hồ sơ sinh viên, tương tác), chuẩn hóa và làm giàu metadata.
2. 🔍 **Retrieval (Truy xuất):** Sử dụng OpenAI `text-embedding-ada-002` để tạo vector embedding cho tài liệu.
3. 🧩 **Candidate Generation & Combination:** Tạo ứng viên tài liệu từ nhiều nguồn (CBF, CF) và loại bỏ trùng lặp.
4. 📈 **Reranking & RAG Context Selection:** Áp dụng heuristic để xếp hạng lại và chọn tài liệu tham chiếu.
5. 🧠 **LLM (Large Language Model):** GPT-3.5 nhận truy vấn và thông tin tài liệu ứng viên để sinh giải thích.
6. 📚 **Quản lý Lịch sử:** Lưu lịch sử truy vấn của sinh viên vào database SQLite.
7. 🌐 **Giao diện Web (Flask):** Xử lý đăng nhập, chat, hiển thị tài liệu gợi ý và lịch sử.

## 🗂️ Cấu trúc Thư mục
```
SGU_Agent/
├── app.py
├── history_manager.py
├── openai_utils.py
├── models/
│   ├── __init__.py
│   ├── data_processing.py
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
- Tài khoản OpenAI API và API Key

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
4. **Cấu hình API Key:** Tạo file `.env` và thêm:
    ```dotenv
    OPENAI_API_KEY=YOUR_OPENAI_API_KEY_HERE
    SECRET_KEY=A_RANDOM_LONG_AND_COMPLEX_STRING_FOR_SESSION_SECURITY
    ```

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

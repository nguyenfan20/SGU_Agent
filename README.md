# SGU_Agent: Hệ thống Gợi ý Tài liệu Học tập Thông minh SGU

## Giới thiệu

Đây là dự án đồ án chuyên ngành xây dựng một hệ thống gợi ý tài liệu học tập cá nhân hóa, chính xác và minh bạch cho sinh viên Đại học Sài Gòn. Hệ thống sử dụng kết hợp các phương pháp hiện đại bao gồm Collaborative Filtering (CF), Content-Based Filtering (CBF), và Retrieval-Augmented Generation (RAG) trên nền tảng Flask.

Mục tiêu chính:
- Cung cấp gợi ý tài liệu học tập phù hợp dựa trên hồ sơ sinh viên, lịch sử tương tác và truy vấn tìm kiếm.
- Khắc phục vấn đề Cold-start cho người dùng mới.
- Tích hợp giao diện web đơn giản cho phép sinh viên tương tác và nhận gợi ý kèm giải thích rõ ràng.
- Sử dụng công nghệ tiên tiến (Embedding, FAISS, LLM) để nâng cao chất lượng gợi ý và trải nghiệm người dùng.

## Kiến trúc Hệ thống

Hệ thống được xây dựng theo kiến trúc microservice đơn giản, với các module chính:

1.  **Nhập liệu & Tiền xử lý:** Đọc dữ liệu từ các file CSV (tài liệu, hồ sơ sinh viên, tương tác), chuẩn hóa và làm giàu metadata (thêm `subject`, `relevant_majors` vào dữ liệu tài liệu).
2.  **Retrieval (Truy xuất):**
    *   Sử dụng OpenAI `text-embedding-ada-002` để tạo vector embedding cho tài liệu.
    *   Xây dựng chỉ mục FAISS (Index HNSWFlat) để lưu trữ và tìm kiếm nhanh các vector tài liệu.
    *   Tìm kiếm ngữ nghĩa các tài liệu ứng viên dựa trên vector embedding của truy vấn hoặc profile.
3.  **Candidate Generation & Combination:**
    *   Tạo các ứng viên tài liệu từ nhiều nguồn:
        *   CBF (Query): Tài liệu có vector gần với vector truy vấn (sử dụng FAISS).
        *   CBF (Profile): Tài liệu có vector gần với vector được tạo từ thông tin hồ sơ sinh viên (ngành, năm học).
        *   CF (Collaborative Filtering): Tài liệu được quan tâm bởi các sinh viên có lịch sử tương tác tương tự.
    *   Kết hợp các ứng viên từ các nguồn khác nhau, loại bỏ trùng lặp.
4.  **Reranking & RAG Context Selection:**
    *   Áp dụng heuristic đơn giản (ví dụ: ưu tiên tài liệu có `relevant_majors` khớp với ngành của sinh viên) để xếp hạng lại các ứng viên.
    *   Chọn một tập nhỏ các tài liệu hàng đầu (ví dụ: top 10) làm ngữ cảnh tham chiếu cho LLM.
5.  **LLM (Large Language Model):**
    *   Sử dụng OpenAI GPT-3.5 (`gpt-3.5-turbo`).
    *   Nhận truy vấn ban đầu, thông tin hồ sơ sinh viên, và chi tiết các tài liệu ứng viên đã chọn (tên, tóm tắt, từ khóa, môn học, ngành liên quan) dưới dạng prompt.
    *   GPT-3.5 thực hiện:
        *   Phân tích sự phù hợp của các tài liệu với truy vấn và ngành của sinh viên.
        *   Tổng hợp và tóm tắt nội dung liên quan.
        *   Sinh ra một đoạn giải thích tự nhiên lý do các tài liệu được gợi ý.
6.  **Quản lý Lịch sử:** Lưu trữ lịch sử truy vấn của từng sinh viên vào database SQLite.
7.  **Giao diện Web (Flask):**
    *   Xử lý luồng đăng nhập/đăng xuất sử dụng Flask Session.
    *   Cung cấp giao diện chat đơn giản cho phép nhập truy vấn và hiển thị kết quả gợi ý (tên tài liệu + giải thích RAG).
    *   Hiển thị lịch sử truy vấn gần đây trong sidebar trái.
    *   Hiển thị danh sách tất cả tài liệu với chức năng tìm kiếm ID/tiêu đề trong sidebar phải.
    *   Sử dụng AJAX (Fetch API) để giao tiếp không đồng bộ giữa frontend và backend.

## Cấu trúc Thư mục
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


## Yêu cầu Hệ thống

*   Python 3.7+
*   Kết nối Internet (để tải mô hình Sentence-BERT lần đầu và gọi OpenAI API).
*   Tài khoản OpenAI API và API Key.

## Cài đặt

1.  **Clone Repository** (nếu có sử dụng Git) hoặc tải xuống mã nguồn.
2.  **Cài đặt Thư viện Python:**
    *   Mở Terminal hoặc Command Prompt.
    *   Di chuyển đến thư mục gốc của dự án (`SGU_Agent`).
    *   (Khuyến khích) Tạo và kích hoạt môi trường ảo:
        ```bash
        python -m venv venv
        # On Windows:
        venv\Scripts\activate
        # On macOS/Linux:
        source venv/bin/activate
        ```
    *   Cài đặt các thư viện từ `requirements.txt`:
        ```bash
        pip install -r requirements.txt
        ```
3.  **Chuẩn bị Dữ liệu:**
    *   Đặt các file CSV (`documents.csv`, `student_profiles.csv`, `interactions.csv`) vào thư mục `data/`. Đảm bảo `documents.csv` có các cột `subject` và `relevant_majors`.
    *   (Tùy chọn) Nếu không có sẵn file CSV, ứng dụng sẽ tạo dummy data cho lần chạy đầu tiên.
    *   File `query_history.db` sẽ được tạo tự động khi chạy ứng dụng lần đầu.
4.  **Cấu hình API Key và Secret Key:**
    *   Tạo file `.env` ở thư mục gốc của dự án (nếu chưa có).
    *   Thêm dòng sau vào file `.env`, thay thế placeholder bằng API Key và Secret Key của bạn:
        ```dotenv
        OPENAI_API_KEY=YOUR_OPENAI_API_KEY_HERE
        SECRET_KEY=A_RANDOM_LONG_AND_COMPLEX_STRING_FOR_SESSION_SECURITY
        ```
    *   (Quan trọng) **KHÔNG** chia sẻ file `.env` hoặc API Key của bạn. Nếu dùng Git, thêm `.env` vào `.gitignore`.

## Chạy Ứng dụng

1.  Mở Terminal hoặc Command Prompt và di chuyển đến thư mục gốc của dự án.
2.  Kích hoạt môi trường ảo (nếu đã tạo):
    ```bash
    # On Windows: venv\Scripts\activate
    # On macOS/Linux: source venv/bin/activate
    ```
3.  Chạy ứng dụng Flask:
    ```bash
    python app.py
    ```
4.  Mở trình duyệt web và truy cập địa chỉ hiển thị trong console (thường là `http://127.0.0.1:5000/`).

## Sử dụng Hệ thống

1.  Bạn sẽ được chuyển hướng đến trang đăng nhập. Nhập User ID và nhấn "Đăng nhập". (Trong demo này, bất kỳ User ID nào có trong `student_profiles.csv` sẽ được "đăng nhập" thành công).
2.  Giao diện chat sẽ hiện ra.
3.  Nhập truy vấn của bạn vào ô nhập liệu ở dưới cùng và nhấn Enter hoặc nút gửi.
4.  Hệ thống sẽ xử lý, tìm kiếm tài liệu liên quan, sử dụng GPT-3.5 để sinh giải thích, và hiển thị kết quả trong luồng chat.
5.  Lịch sử truy vấn của bạn sẽ được lưu và hiển thị trong sidebar bên trái.
6.  Nhấn nút 3 gạch ở góc trên bên phải để bật/tắt sidebar danh sách tài liệu, nơi bạn có thể tìm kiếm tài liệu theo ID hoặc tiêu đề.

## Thử nghiệm

*   Thử các truy vấn khác nhau (ví dụ: tên môn học, khái niệm, dạng bài tập).
*   Thử với các User ID khác nhau (có lịch sử, không có lịch sử).
*   Kiểm tra lịch sử truy vấn trong sidebar trái.
*   Sử dụng chức năng tìm kiếm trong sidebar phải để tìm tài liệu cụ thể.

## Đóng góp

Nếu bạn có ý tưởng hoặc muốn đóng góp vào dự án này, vui lòng liên hệ [nguyenphan201203@gmail.com].

## Giấy phép

[MIT License](LICENSE.txt)

---
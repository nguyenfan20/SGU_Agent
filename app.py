# app.py (Updated with /api/documents route)
from flask import Flask, request, render_template, redirect, url_for, session, jsonify
import os
import time
import datetime
import pandas as pd # Needed for dummy data generation if files missing

# Import modules
from models.recommendation import RecommendationSystem
import history_manager # SQLite

# Removed imports for gemini_chat and openai_utils

from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'supersecretdevkey')

DATA_DIRECTORY = 'data'

# --- Initialize Database and Recommendation System ---
history_manager.init_db()

print(f"Initializing Recommendation System from data in '{DATA_DIRECTORY}'... This might take a moment.")
recommender = None
try:
    recommender = RecommendationSystem(
        data_dir=DATA_DIRECTORY
    )
    print("Recommendation System initialization attempt complete.")
except Exception as e:
    print(f"FATAL ERROR during Recommendation System initialization: {e}")
    recommender = None

# Check system readiness after initialization
if recommender is None or not recommender.is_ready:
     print("FATAL: Recommendation System is not ready. Check logs above for details (likely OpenAI config, data, or retrieval init).")


# --- Flask Routes ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    sample_user_ids = []
    if recommender and recommender.profiles_df is not None and not recommender.profiles_df.empty:
         sample_user_ids = recommender.profiles_df['user_id'].tolist()[:10]

    if request.method == 'POST':
        user_id = request.form.get('user_id', '').strip()
        if user_id:
            if recommender and recommender.profiles_df is not None and not recommender.profiles_df.empty:
                 if user_id not in recommender.profiles_df['user_id'].tolist():
                      return render_template('login.html', error=f"Mã sinh viên '{user_id}' không tồn tại trong dữ liệu hồ sơ sinh viên.", sample_user_ids=sample_user_ids, user_id=user_id)
            elif recommender is None:
                 print(f"Warning: Recommender system failed to initialize, cannot check user ID '{user_id}' against profiles.")

            session['user_id'] = user_id
            print(f"User '{user_id}' logged in.")
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="Vui lòng nhập Mã Sinh viên.", sample_user_ids=sample_user_ids)

    return render_template('login.html', sample_user_ids=sample_user_ids)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    print("User logged out.")
    return redirect(url_for('login'))


@app.route('/', methods=['GET'])
def index():
    user_id = session.get('user_id')
    if not user_id:
        return redirect(url_for('login'))
    return render_template('index.html', current_user_id=user_id)


@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Handle recommendation request using RAG.
    Returns JSON with 'recommendations' (including titles, explanation) or 'error'.
    """
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "Bạn chưa đăng nhập."}), 401

    request_data = request.get_json()
    query = request_data.get('query', '').strip()

    if not query:
        return jsonify({"error": "Vui lòng nhập nội dung truy vấn."})

    history_manager.add_query_to_history(user_id, query)

    start_time = time.time()
    response_data = {"logged_query": {"user_id": user_id, "query": query, "timestamp": datetime.datetime.now().isoformat()}}

    try:
        if recommender is None or not recommender.is_ready:
             response_data["error"] = "Hệ thống gợi ý tài liệu chưa sẵn sàng. Vui lòng kiểm tra cấu hình OpenAI API, dữ liệu tài liệu, và quá trình khởi tạo Retrieval."
             response_data["recommendations"] = []
        else:
            recommendations, doc_rec_error = recommender.get_recommendations(user_id, query, top_n=5, k_retrieval=20)

            if doc_rec_error:
                response_data["error"] = doc_rec_error
                response_data["recommendations"] = []
            elif recommendations:
                response_data["recommendations"] = recommendations
                response_data["overall_source"] = "RAG (OpenAI + Document System)"

            else:
                 response_data["error"] = f"Không tìm thấy gợi ý tài liệu phù hợp với truy vấn '{query}'. Vui lòng thử lại."
                 response_data["recommendations"] = []


    except Exception as e:
        response_data["error"] = f"Đã xảy ra lỗi không mong muốn trong quá trình xử lý yêu cầu: {e}"
        print(f"Unexpected error during request processing: {e}")

    end_time = time.time()
    print(f"Total request time for user {user_id}, query '{query}': {end_time - start_time:.2f} seconds.")

    final_response_json = {}
    if "recommendations" in response_data and response_data["recommendations"]:
        final_response_json["recommendations"] = response_data["recommendations"]
        final_response_json["source"] = response_data.get("overall_source", "RAG")

    elif "error" in response_data:
        final_response_json["error"] = response_data["error"]
        final_response_json["source"] = "System Error"
    else:
         final_response_json["error"] = "Hệ thống không phản hồi."
         final_response_json["source"] = "Unknown"

    final_response_json["logged_query"] = response_data.get("logged_query")


    return jsonify(final_response_json)


@app.route('/api/history', methods=['GET'])
def api_get_history():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"history": [], "error": "Bạn chưa đăng nhập."}), 401

    try:
        history = history_manager.get_query_history(user_id, limit=20)
        formatted_history = []
        for entry in history:
             timestamp_str = entry['timestamp'].isoformat() if isinstance(entry['timestamp'], datetime.datetime) else str(entry['timestamp'])
             formatted_history.append({
                 'user_id': entry['user_id'],
                 'query': entry['query'],
                 'timestamp': timestamp_str
             })

        return jsonify({"history": formatted_history, "error": None})
    except Exception as e:
        error_message = f"Đã xảy ra lỗi khi đọc lịch sử truy vấn: {e}"
        print(f"Error retrieving history via API for user {user_id}: {e}")
        return jsonify({"history": [], "error": error_message}), 500

# --- NEW API ROUTE TO GET DOCUMENT LIST ---
@app.route('/api/documents', methods=['GET'])
def api_get_documents():
    """
    API endpoint to fetch the list of all documents (ID and Title).
    Requires login.
    Returns JSON: { "documents": [...], "error": "..." }
    """
    # Check if user is logged in
    if not session.get('user_id'):
        return jsonify({"documents": [], "error": "Bạn chưa đăng nhập."}), 401

    # Check if recommender and document data are available
    if recommender is None or recommender.documents_df is None or recommender.documents_df.empty:
         return jsonify({"documents": [], "error": "Hệ thống gợi ý hoặc dữ liệu tài liệu chưa sẵn sàng."}), 500

    try:
        # Select only the necessary columns (doc_id, title)
        documents_list = recommender.documents_df[['doc_id', 'title']].to_dict('records')
        return jsonify({"documents": documents_list, "error": None})
    except Exception as e:
        error_message = f"Đã xảy ra lỗi khi lấy danh sách tài liệu: {e}"
        print(f"Error retrieving document list via API: {e}")
        return jsonify({"documents": [], "error": error_message}), 500

# --- Run the Flask App ---
if __name__ == '__main__':
    history_manager.init_db()

    data_dir_path = 'data'
    if not os.path.exists(os.path.join(data_dir_path, 'documents.csv')):
        print("Data files not found. Creating dummy data...")
        os.makedirs(data_dir_path, exist_ok=True)
        dummy_docs_data = {
            'item_id': [f'DOC{i:03d}' for i in range(100)],
            'title': [f'Bài giảng môn học {i+1}' for i in range(100)],
            'abstract': [f'Tóm tắt nội dung bài giảng {i+1} về chủ đề X, Y, Z.' for i in range(100)],
            'keywords': [f'môn học {i+1}, chủ đề X, chủ đề Y' for i in range(100)],
            'subject': [f'Môn {i % 15 + 1}' for i in range(100)],
            'relevant_majors': [f'Ngành {i % 5 + 1}, Ngành {(i+1) % 5 + 1}' for i in range(100)]
        }
        dummy_docs = pd.DataFrame(dummy_docs_data)

        dummy_profiles_data = {
            'user_id': [f'SGU{i:03d}' for i in range(50)],
            'major': [f'Ngành {i % 5 + 1}' for i in range(50)],
            'year': [i % 4 + 1 for i in range(50)]
        }
        dummy_profiles = pd.DataFrame(dummy_profiles_data)

        dummy_interactions_data = {
            'user_id': [f'SGU{i % 50:03d}' for i in range(200)],
            'content_id': [f'DOC{i % 100:03d}' for i in range(200)],
            'action_type': ['view', 'download'][i % 2],
            'timestamp': [datetime.datetime.now() - datetime.timedelta(days=i) for i in range(200)]
        }
        dummy_interactions = pd.DataFrame(dummy_interactions_data)

        dummy_docs.to_csv(os.path.join(data_dir_path, 'documents.csv'), index=False)
        dummy_profiles.to_csv(os.path.join(data_dir_path, 'student_profiles.csv'), index=False)
        dummy_interactions.to_csv(os.path.join(data_dir_path, 'interactions.csv'), index=False)
        print("Dummy data created.")
    else:
        print("Data files found.")


    app.run(debug=True, port=5000)
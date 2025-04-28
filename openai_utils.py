# openai_utils.py (Final version with numpy import and RAG)
import openai
import os
from dotenv import load_dotenv
import time
import math
import numpy as np # <-- ENSURE THIS IS HERE

# Load environment variables
load_dotenv()

# Get API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure the API client
openai_client = None
is_openai_ready = False
if OPENAI_API_KEY:
    try:
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        is_openai_ready = True
        print("OpenAI API client configured successfully.")

    except Exception as e:
        print(f"Error configuring OpenAI API client: {e}")
        is_openai_ready = False
        openai_client = None
else:
    print("Warning: OPENAI_API_KEY not found. OpenAI features will not be available.")
    is_openai_ready = False
    openai_client = None

# Define Embedding Model
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"
OPENAI_EMBEDDING_DIMENSION = 1536

# Define Chat Model for RAG/Generation
OPENAI_CHAT_MODEL = "gpt-3.5-turbo"


def get_embedding(text):
    """
    Gets the embedding for a given text using OpenAI's embedding model.
    Returns: numpy array or None if error/not ready
    """
    if not is_openai_ready or openai_client is None:
        print("OpenAI API is not ready for embedding.")
        return None

    if not text or not text.strip():
         print("Warning: Attempted to get embedding for empty text.")
         return np.full(OPENAI_EMBEDDING_DIMENSION, np.nan, dtype=np.float32)


    try:
        response = openai_client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=[text.strip()]
        )

        if response and response.data and len(response.data) > 0:
             embedding = response.data[0].embedding
             embedding_np = np.array(embedding, dtype=np.float32)

             if embedding_np.shape[0] != OPENAI_EMBEDDING_DIMENSION:
                 print(f"Warning: Unexpected embedding dimension from OpenAI. Expected {OPENAI_EMBEDDING_DIMENSION}, got {embedding_np.shape[0]}.")
                 return np.full(OPENAI_EMBEDDING_DIMENSION, np.nan, dtype=np.float32)

             if np.isnan(embedding_np).any() or np.isinf(embedding_np).any():
                  print("Warning: Received NaN or Inf in embedding from OpenAI.")
                  return np.full(OPENAI_EMBEDDING_DIMENSION, np.nan, dtype=np.float32)

             return embedding_np
        else:
            print(f"Warning: OpenAI embedding response data is empty for text: '{text[:50]}...'. Response: {response}")
            return np.full(OPENAI_EMBEDDING_DIMENSION, np.nan, dtype=np.float32)

    except Exception as e:
        print(f"Error getting embedding for text: '{text[:50]}...'. Error: {e}")
        return np.full(OPENAI_EMBEDDING_DIMENSION, np.nan, dtype=np.float32)


def get_rag_completion(query, documents_context, user_profile=None):
    """
    Gets a natural language response and explanation from GPT-3.5,
    using the provided query, document context, and user profile.
    Documents context should be a list of dicts/strings representing relevant documents details.
    """
    if not is_openai_ready or openai_client is None:
        return None, "OpenAI API chưa sẵn sàng cho sinh nội dung. Vui lòng kiểm tra cấu hình API Key."

    if not query or not query.strip():
         return None, "Truy vấn rỗng."

    # --- Prepare Context for GPT ---
    context_text = ""
    if documents_context:
         context_text += "Thông tin tài liệu tham khảo:\n"
         for i, doc in enumerate(documents_context):
              # Ensure 'details' dict contains expected keys, provide fallbacks
              title = doc.get('title', 'Không có tiêu đề')
              abstract = doc.get('abstract', 'Không có tóm tắt')
              keywords = doc.get('keywords', '')
              subject = doc.get('subject', 'Không rõ môn học') # Use new fields
              relevant_majors = doc.get('relevant_majors', 'Không rõ ngành liên quan') # Use new fields


              context_text += f"### Tài liệu {i+1}:\n"
              context_text += f"Tiêu đề: {title}\n"
              context_text += f"Môn học: {subject}\n" # Include subject
              context_text += f"Ngành liên quan: {relevant_majors}\n" # Include relevant majors
              if abstract.strip(): context_text += f"Tóm tắt: {abstract.strip()}\n"
              if keywords.strip(): context_text += f"Từ khóa: {keywords.strip()}\n"
              context_text += "\n"

    if user_profile:
        # Assume user_profile is a dict like {'user_id': '...', 'major': '...', 'year': '...'}
        user_id_str = user_profile.get('user_id', 'N/A')
        major = user_profile.get('major', 'một ngành học')
        year = user_profile.get('year', 'không rõ năm học')

        profile_text = f"Thông tin người dùng: Sinh viên ID {user_id_str}, Ngành {major}"
        if year and year != '0':
             profile_text += f", Năm {year}"
        profile_text += ".\n\n"
        context_text = profile_text + context_text # Add profile context first

    # --- Craft the Prompt for GPT ---
    # Refine the system message and user message to leverage the new context fields
    system_message = """Bạn là một trợ lý AI cho sinh viên Đại học Sài Gòn, chuyên về gợi ý tài liệu học tập.
    Nhiệm vụ của bạn là:
    1. Hiểu câu hỏi/truy vấn của sinh viên.
    2. Dựa vào thông tin người dùng (ngành, năm học) và các tài liệu tham khảo được cung cấp (bao gồm tiêu đề, tóm tắt, từ khóa, môn học, ngành liên quan), giải thích **cụ thể** lý do tại sao **từng tài liệu liên quan** phù hợp với truy vấn **và đặc biệt là ngành học của sinh viên**.
    3. Tóm tắt nội dung chính của các tài liệu liên quan đến truy vấn.
    4. Trả lời một cách tự nhiên, thân thiện, và hoàn toàn bằng tiếng Việt.
    5. **CHỈ** sử dụng thông tin từ các tài liệu tham khảo được cung cấp và thông tin người dùng (nếu có). KHÔNG sử dụng kiến thức chung của bạn để tóm tắt nội dung tài liệu hoặc xác định sự phù hợp ngành nếu thông tin đó không có trong dữ liệu tài liệu hoặc profile.
    6. Nếu các tài liệu tham khảo không đủ để trả lời truy vấn hoặc không liên quan đến ngành của sinh viên, hãy nói rõ điều đó và khuyến khích tìm kiếm thêm hoặc cung cấp truy vấn chi tiết hơn.
    7. Trình bày câu trả lời súc tích.
    """

    user_message = f"""Truy vấn của sinh viên: {query}

    {context_text if context_text else 'Không có tài liệu tham khảo nào được cung cấp.'}

    Dựa trên thông tin trên, hãy giải thích lý do gợi ý, tóm tắt nội dung các tài liệu liên quan, và đánh giá sự phù hợp với truy vấn và ngành học của sinh viên. Trả lời bằng tiếng Việt."""


    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    print(f"Sending RAG prompt to OpenAI (model: {OPENAI_CHAT_MODEL}) for query: '{query}'")
    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=messages,
            temperature=0.5,
            max_tokens=600 # Increased max tokens slightly for more detailed explanation
        )

        if response and response.choices and response.choices[0].message and response.choices[0].message.content:
             rag_response_text = response.choices[0].message.content.strip()
             print("Received RAG response from OpenAI.")
             return rag_response_text, None
        else:
            error_detail = "OpenAI chat response data is empty or problematic."
            print(f"OpenAI chat response empty: {response}")
            return None, error_detail

    except Exception as e:
        print(f"Error calling OpenAI chat API for RAG: {e}")
        return None, f"Lỗi khi gọi OpenAI API: {e}"
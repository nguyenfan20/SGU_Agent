from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch # Needed for T5/Transformers
import pandas as pd
import os
import time
import re # Useful for cleaning T5 output

class ExplanationGenerator:
    # Using VietAI/vit5-base as the explanation model
    def __init__(self, profiles_df, model_name='VietAI/vit5-base'):
        self.profiles_df = profiles_df
        self.model_name = model_name
        self.explanation_model = None
        self.explanation_tokenizer = None
        self.is_ready = False # Readiness depends on T5 model loading

        self._load_explanation_model()

        if self.explanation_model is not None and self.explanation_tokenizer is not None:
             self.is_ready = True
             print("Explanation Generator (T5-based) ready.")
        else:
             print(f"Explanation Generator not ready: T5 Model '{self.model_name}' load failed.")

    def _load_explanation_model(self):
        print(f"Loading explanation model: {self.model_name}")
        try:
            # Load T5 tokenizer and model
            self.explanation_tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self.explanation_model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            # Optional: Move model to GPU if available and you have torch-gpu installed
            # if torch.cuda.is_available():
            #      self.explanation_model.to('cuda')
            #      print("T5 model moved to GPU.")
            print("Explanation model loaded successfully.")
        except Exception as e:
             print(f"Error loading explanation model {self.model_name}: {e}")
             self.explanation_model = None
             self.explanation_tokenizer = None


    def generate(self, doc_info, query, user_id, source):
        """Generates a text explanation for a recommendation using T5."""
        if not self.is_ready:
            # Fallback template if T5 not ready
             if not self.profiles_df.empty:
                  user_profile = self.profiles_df[self.profiles_df['user_id'] == user_id]
                  user_major = user_profile.iloc[0].get('major', 'ngành học của bạn') if not user_profile.empty else 'ngành học của bạn'
                  doc_title = doc_info.get('title', 'tài liệu này')
                  doc_subject = doc_info.get('subject', 'môn học không rõ')
                  return f"Hệ thống sinh giải thích chưa sẵn sàng. Tài liệu '{doc_title}' (Môn: {doc_subject}) được gợi ý dựa trên nguồn '{source}' (ví dụ: liên quan đến ngành {user_major} hoặc truy vấn của bạn)."
             else:
                 doc_title = doc_info.get('title', 'tài liệu này')
                 doc_subject = doc_info.get('subject', 'môn học không rõ')
                 return f"Hệ thống sinh giải thích chưa sẵn sàng. Tài liệu '{doc_title}' (Môn: {doc_subject}) được gợi ý dựa trên nguồn '{source}'."


        if not doc_info:
             return f"Không thể tạo giải thích chi tiết cho tài liệu (Nguồn: {source})."

        # Get document information
        title = doc_info.get('title', 'Tài liệu này')
        abstract = doc_info.get('abstract', 'có nội dung học tập')
        keywords = doc_info.get('keywords', '')
        subject = doc_info.get('subject', 'môn học không rõ') # Use subject
        relevant_majors = doc_info.get('relevant_majors', '') # Use relevant_majors

        keywords_list = [k.strip() for k in keywords.split(',') if k.strip()]
        keywords_str = ", ".join(keywords_list) if keywords_list else "không có"

        relevant_majors_list = [m.strip() for m in relevant_majors.split(',') if m.strip()]
        relevant_majors_str = ", ".join(relevant_majors_list) if relevant_majors_list else "không rõ"


        # Get user profile information
        user_major = "một ngành học"
        user_year = "không rõ năm học"
        user_profile = self.profiles_df[self.profiles_df['user_id'] == user_id]
        if not user_profile.empty:
             user_major = user_profile.iloc[0].get('major', user_major)
             user_year = user_profile.iloc[0].get('year', user_year)
             user_year_str = f"năm {user_year}" if user_year and str(user_year).strip() and str(user_year) != '0' else ""
        else:
             user_major = "ngành học của bạn"
             user_year_str = ""


        # --- Craft prompt for T5 (refined to include new fields) ---
        # Make the prompt as clear as possible about what context is available
        input_text = f"""Giải thích chi tiết:
        Tài liệu: {title}
        Môn học: {subject}
        Ngành liên quan của tài liệu: {relevant_majors_str}
        Tóm tắt tài liệu: {abstract}
        Từ khóa tài liệu: {keywords_str}

        Thông tin người dùng: Sinh viên ngành {user_major} {user_year_str} (ID: {user_id})
        Nguồn gợi ý ban đầu: {source}
        Truy vấn tìm kiếm: {query}

        Giải thích tại sao tài liệu này được gợi ý cho sinh viên dựa trên thông tin trên, tập trung vào sự liên quan đến truy vấn và ngành học của sinh viên:
        """
        # Example: A shorter prompt
        # input_text = f"explain: Tài liệu '{title}' (Môn {subject}, Ngành liên quan: {relevant_majors_str}) được gợi ý cho sinh viên ngành {user_major} {user_year_str} vì liên quan đến truy vấn '{query}'. Lý do:"


        input_ids = self.explanation_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

        print(f"Generating explanation for '{title}' (Source: {source}, T5 Model: {self.model_name})...")
        raw_explanation = ""
        try:
            # Use generation parameters suitable for Vietnamese and getting reasonable length
            output_ids = self.explanation_model.generate(
                input_ids,
                max_length=250, # Increased max length
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=3,
                temperature=0.8, # Experiment with temperature (higher for more variation, lower for more focused)
                top_k=50,
                top_p=0.95
                # Add specific decoding strategies if available and needed for diacritics
                # E.g., constraints, custom logits processing if the model supports it
            )
            raw_explanation = self.explanation_tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # Simple post-processing to remove prompt prefixes or unwanted tokens
            # This might need careful tuning based on actual model output
            raw_explanation = re.sub(r'^.*?Giải thích tại sao tài liệu này được gợi ý.*?[:\.]?\s*', '', raw_explanation, flags=re.DOTALL).strip() # Remove prompt leading part


            # Add a concluding sentence if the generated text is too short or seems incomplete
            if not raw_explanation.strip() or len(raw_explanation.split()) < 15: # Check for slightly longer minimum length
                 fallback_suffix = f"Tài liệu '{title}' (Môn: {subject}, Ngành liên quan: {relevant_majors_str}). "
                 if source == 'CBF_Query_Retrieval' and query:
                     fallback_suffix += f"Nó rất phù hợp với truy vấn '{query}' của bạn. "
                 elif source == 'CBF_Profile' and user_major != "ngành học của bạn":
                      fallback_suffix += f"Nó phù hợp với ngành {user_major} {user_year_str} của bạn. "
                 elif source == 'CF':
                      fallback_suffix += "Đây là tài liệu được sinh viên khác có sở thích tương tự quan tâm. "

                 # Combine generated part with fallback if necessary
                 explanation = (raw_explanation.strip() + " " + fallback_suffix.strip()).strip() if raw_explanation.strip() else fallback_suffix.strip()

                 # Final check if explanation is still empty after fallback
                 if not explanation:
                     explanation = f"Tài liệu '{title}' (Môn: {subject}) được gợi ý dựa trên nguồn '{source}'."

            else:
                 explanation = raw_explanation # Use the generated explanation if it looks reasonable


        except Exception as e:
            print(f"Error generating raw explanation for '{title}' with T5: {e}")
            # If T5 generation fails, provide a simple template-based explanation
            source_reason = ""
            if source == 'CBF_Query_Retrieval' and query: source_reason = f" liên quan đến truy vấn '{query}'"
            elif source == 'CBF_Profile' and user_major != "ngành học của bạn": source_reason = f" do bạn quan tâm đến ngành {user_major}"
            elif source == 'CF': source_reason = " được quan tâm bởi sinh viên khác có sở thích tương tự"
            explanation = f"Tài liệu '{title}' (Môn: {subject}) được gợi ý vì{source_reason}. (Lỗi sinh giải thích T5: {e})"
            # Return the fallback explanation directly
            return explanation.strip()


        # Return the generated/fallback explanation
        return explanation.strip()

# Example Usage (optional) - Ensure you have dummy data with correct columns
# And T5 model can be downloaded/loaded
if __name__ == '__main__':
    print("Testing models/explanation.py (T5-based)")
    # Need dummy data with new columns (user_id, major, year | doc_id, title, abstract, keywords, subject, relevant_majors)
    dummy_profiles_df = pd.DataFrame({
        'user_id': ['test_user_001', 'test_user_002', 'test_user_no_profile'],
        'major': ['Công nghệ thông tin', 'Quản trị Kinh doanh', None],
        'year': [3, 2, None]
    })
    dummy_doc_info_it = {'doc_id': 'DOC_IT', 'title': 'Bài giảng Cơ sở dữ liệu', 'abstract': 'Giới thiệu các mô hình cơ sở dữ liệu quan hệ, phi quan hệ, và ngôn ngữ truy vấn SQL cơ bản và nâng cao.', 'keywords': 'cơ sở dữ liệu, SQL, mô hình dữ liệu, NoSQL', 'subject': 'Cơ sở dữ liệu', 'relevant_majors': 'Công nghệ thông tin, Khoa học dữ liệu'}
    dummy_doc_info_business = {'doc_id': 'DOC_BUSINESS', 'title': 'Nguyên lý Marketing', 'abstract': 'Các khái niệm cơ bản về marketing, phân tích thị trường, định vị sản phẩm, và chiến lược 4Ps.', 'keywords': 'marketing, 4Ps, phân tích thị trường, định vị, kinh doanh', 'subject': 'Marketing', 'relevant_majors': 'Marketing, Quản trị Kinh doanh, Kinh tế'}
    dummy_doc_info_generic = {'doc_id': 'DOC_GENERIC', 'title': 'Kỹ năng học tập hiệu quả', 'abstract': 'Phương pháp ghi chép, ôn tập, quản lý thời gian.', 'keywords': 'kỹ năng học tập, phương pháp, quản lý thời gian', 'subject': 'Kỹ năng mềm', 'relevant_majors': 'Tất cả ngành'}


    # Initialize explanation generator (T5-based)
    exp_generator = ExplanationGenerator(dummy_profiles_df, model_name='VietAI/vit5-base')

    if exp_generator.is_ready:
        print("\nExplanation Generator initialized (T5-based).")


        # Test CBF Query + IT user
        explanation_query_it = exp_generator.generate(dummy_doc_info_it, "truy vấn sql nâng cao", "test_user_001", "CBF_Query_Retrieval")
        print(f"\nExplanation (Query, IT): {explanation_query_it}")

        # Test CBF Profile + IT user
        explanation_profile_it = exp_generator.generate(dummy_doc_info_it, "", "test_user_001", "CBF_Profile")
        print(f"\nExplanation (Profile, IT): {explanation_profile_it}")

        # Test CF + Business user
        explanation_cf_business = exp_generator.generate(dummy_doc_info_it, "", "test_user_002", "CF")
        print(f"\nExplanation (CF, Business on IT doc): {explanation_cf_business}")

        # Test Cold Start user + Query (profile not found)
        explanation_cold_query = exp_generator.generate(dummy_doc_info_generic, "quản lý thời gian", "COLD_START_USER", "CBF_Query_Retrieval")
        print(f"\nExplanation (Cold Start User, Query): {explanation_cold_query}")

         # Test Cold Start user + Profile CBF (should use fallback if profile not found)
        explanation_cold_profile = exp_generator.generate(dummy_doc_info_business, "", "COLD_START_USER", "CBF_Profile")
        print(f"\nExplanation (Cold Start User, Profile): {explanation_cold_profile}")

        # Test user with no major/year in profile + CBF Profile
        explanation_no_major_profile = exp_generator.generate(dummy_doc_info_generic, "", "test_user_no_profile", "CBF_Profile")
        print(f"\nExplanation (User No Profile Data, Profile): {explanation_no_major_profile}")

    else:
         print("\nExplanation generator could not be initialized.")

    print("\nRecommendation System testing complete.")
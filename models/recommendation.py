# models/recommendation.py (Final version for RAG with new data)
import pandas as pd
import numpy as np
import time
import os
import faiss
# Removed SentenceTransformer import

# Import modules from the same package
from . import data_processing
from .retrieval import DocumentRetrieval # Uses OpenAI Embedding
# Removed import ExplanationGenerator

# Import OpenAI utility module (provides embedding and chat completion)
import openai_utils

class RecommendationSystem:
    def __init__(self, data_dir='data'): # Removed embedding_model_name param
        self.data_dir = data_dir
        self.embedding_model_name = openai_utils.OPENAI_EMBEDDING_MODEL # Use the name from openai_utils

        # 1. Load Data
        print("Loading data...")
        self.documents_df, self.profiles_df, self.interactions_df = data_processing.load_all_data(self.data_dir)
        print("Data loading complete.")

        # Check if crucial data is loaded and has necessary columns for RAG/Retrieval
        required_doc_cols_for_retrieval = ['doc_id', 'title', 'abstract', 'keywords', 'subject', 'relevant_majors']
        if self.documents_df.empty or not all(col in self.documents_df.columns for col in required_doc_cols_for_retrieval):
            missing_cols = [col for col in required_doc_cols_for_retrieval if col not in self.documents_df.columns]
            print(f"FATAL: documents.csv is empty or missing required columns for Retrieval/RAG context: {missing_cols}. Recommendation system cannot function.")
            self.is_ready = False
            self.retrieval_system = None
            # Ensure dataframes are initialized even if empty for safety
            self.documents_df = pd.DataFrame(columns=required_doc_cols_for_retrieval)
            self.profiles_df = pd.DataFrame(columns=['user_id', 'major', 'year'])
            self.interactions_df = pd.DataFrame(columns=['user_id', 'doc_id', 'action_type', 'timestamp'])
            return


        # Check if OpenAI is ready for both embedding (for retrieval) and chat (for RAG)
        if not openai_utils.is_openai_ready:
             print("FATAL: OpenAI API is not ready. Cannot initialize Recommendation System (embedding/RAG unavailable).")
             self.is_ready = False
             self.retrieval_system = None
             return

        # 2. Initialize components (Retrieval)
        self.retrieval_system = DocumentRetrieval(
            documents_df=self.documents_df.copy() # DocumentRetrieval uses openai_utils internally
        )

        # Recalculate capabilities based on data availability and retrieval system readiness
        self._can_do_query_cbf = self.retrieval_system is not None and self.retrieval_system.is_ready
        # Profile CBF needs profiles data AND retrieval system
        self._can_do_profile_cbf = self._can_do_query_cbf and not self.profiles_df.empty and 'major' in self.profiles_df.columns
        # CF needs interactions data AND documents data (for details)
        self._can_do_cf = not self.interactions_df.empty and not self.documents_df.empty


        # System is "ready" if it can do *at least* Query CBF (the base for RAG) AND OpenAI is ready
        # And documents have required columns for RAG context
        self.is_ready = self._can_do_query_cbf and openai_utils.is_openai_ready and all(col in self.documents_df.columns for col in required_doc_cols_for_retrieval)


        if not self.is_ready:
             print("Warning: Recommendation system may not function fully due to initialization issues or missing data.")
             if not self._can_do_query_cbf: print("- Query CBF is unavailable (Check documents and Retrieval system init).")
             if not openai_utils.is_openai_ready: print("- OpenAI API not ready (Embedding/RAG unavailable).")
             if not self._can_do_profile_cbf: print("- CBF Profile is unavailable (Check profiles data and retrieval readiness).")
             if not self._can_do_cf: print("- CF is unavailable (Check interactions and documents data).")


    # --- Core Recommendation Methods ---

    def _get_cbf_profile_recommendations(self, user_id, k=10):
        """Generates recommendations based on user's major and year from profile (using combined string as query)."""
        if not self._can_do_profile_cbf:
             print(f"Cannot perform CBF (Profile) for user {user_id}: profiles data missing or retrieval system not ready.")
             return []

        user_profile = self.profiles_df[self.profiles_df['user_id'] == user_id]
        if user_profile.empty:
            print(f"User {user_id} profile not found for CBF (Profile).")
            return []

        user_major = user_profile.iloc[0].get('major')
        user_year = user_profile.iloc[0].get('year')

        # --- UPDATED PROFILE QUERY STRING ---
        # Combine major and year into a more descriptive query string
        profile_query_string = ""
        if user_major and user_major.strip():
             profile_query_string += f"Tài liệu học tập cho sinh viên ngành {user_major}"
             if user_year and str(user_year).strip() and str(user_year) != '0':
                  profile_query_string += f" năm {user_year}"
             profile_query_string += "."
        elif user_year and str(user_year).strip() and str(user_year) != '0':
             profile_query_string += f"Tài liệu học tập cho sinh viên năm {user_year}."
        else: # No major or year data
            print(f"User {user_id} has insufficient profile data (major/year) for CBF (Profile).")
            return []


        if not profile_query_string.strip():
             return []

        print(f"Performing CBF (Profile) search for user {user_id} with query: '{profile_query_string}'")

        # Use the combined profile string as a query for the retrieval system
        # Retrieval results now include 'details' and 'score'
        profile_query_results = self.retrieval_system.search(profile_query_string, k=k)

        # Add source info to each result item
        for rec in profile_query_results:
            rec['source_candidate'] = 'CBF_Profile' # Indicate this came from Profile CBF
            # Score is already included by retrieval_system

        print(f"Found {len(profile_query_results)} CBF (Profile) recommendations based on profile query.")
        return profile_query_results # Return the list of dicts


    def _get_cf_recommendations(self, user_id, k=10):
        """Generates recommendations based on simple user-based collaborative filtering."""
        if not self._can_do_cf:
             print(f"Cannot perform CF for user {user_id}: interactions or documents data missing.")
             return []

        user_interactions = self.interactions_df[self.interactions_df['user_id'] == user_id]
        if user_interactions.empty or len(user_interactions['doc_id'].unique()) < 2:
            print(f"User {user_id} has insufficient interaction history ({len(user_interactions['doc_id'].unique())} unique docs) for CF.")
            return []

        print(f"Performing CF for user {user_id}")

        user_interacted_docs = set(user_interactions['doc_id'].unique())

        all_users_with_interactions = self.interactions_df['user_id'].unique()
        other_users = [u for u in all_users_with_interactions if u != user_id]

        similarity_scores = {}
        candidate_docs_scores = {} # doc_id -> accumulated similarity score

        for other_user in other_users:
            other_user_interactions = self.interactions_df[self.interactions_df['user_id'] == other_user]
            other_interacted_docs = set(other_user_interactions['doc_id'].unique())

            intersection = user_interacted_docs.intersection(other_interacted_docs)
            union = user_interacted_docs.union(other_interacted_docs)

            if union:
                similarity = len(intersection) / len(union)
                if similarity > 0:
                    similarity_scores[other_user] = similarity

        sorted_similar_users = sorted(similarity_scores.items(), key=lambda item: item[1], reverse=True)

        for similar_user, similarity in sorted_similar_users:
             similar_user_interactions = self.interactions_df[self.interactions_df['user_id'] == similar_user]
             similar_user_interacted_docs = set(similar_user_interactions['doc_id'].unique())

             new_docs = similar_user_interacted_docs - user_interacted_docs

             for doc_id in new_docs:
                 if doc_id not in candidate_docs_scores:
                     candidate_docs_scores[doc_id] = 0
                 candidate_docs_scores[doc_id] += similarity # Accumulate score

        recommendations = []
        doc_info_map = self.documents_df.set_index('doc_id').to_dict('index')

        sorted_candidate_ids = sorted(candidate_docs_scores, key=candidate_docs_scores.get, reverse=True)

        for doc_id in sorted_candidate_ids[:k]: # Take top k based on accumulated score
             doc_details = doc_info_map.get(doc_id)
             if doc_details:
                 recommendations.append({
                     'doc_id': doc_id,
                     'score': candidate_docs_scores[doc_id], # Use accumulated score
                     'source_candidate': 'CF', # Indicate this came from CF
                     'details': doc_details
                 })
             else:
                 print(f"Warning: CF recommended doc ID {doc_id} not found in documents_df.")
                 # Still include basic entry if details missing
                 recommendations.append({
                      'doc_id': doc_id,
                      'score': candidate_docs_scores[doc_id],
                      'source_candidate': 'CF',
                      'details': {'doc_id': doc_id, 'title': f'Tài liệu ID: {doc_id}', 'abstract': '', 'keywords': '', 'subject': '', 'relevant_majors': ''}
                 })


        print(f"Found {len(recommendations)} CF recommendations.")
        return recommendations


    def get_recommendations(self, user_id, query, top_n=5, k_retrieval=20):
        """
        Orchestrates the document recommendation process using RAG.
        1. Retrieve candidates from various sources (Query CBF, Profile CBF, CF).
        2. Combine unique candidates.
        3. Select top candidates for RAG context (can include reranking/filtering).
        4. Prepare context for GPT using selected candidates' details and user profile.
        5. Call OpenAI GPT-3.5 for RAG completion (ranking + explanation + summary).
        6. Format results, linking GPT explanation to the top N displayed documents.
        Returns: tuple (list of recommended_documents_with_explanation, error_message)
        """
        # Check core system readiness (OpenAI + Retrieval + Documents with necessary columns)
        if not self.is_ready:
             return [], "Hệ thống gợi ý tài liệu chưa sẵn sàng. Vui lòng kiểm tra cấu hình OpenAI API, dữ liệu tài liệu (cần subject, relevant_majors), và quá trình khởi tạo Retrieval."

        # Determine which recommendation methods are possible based on data availability
        can_do_query_cbf = query and self._can_do_query_cbf
        can_do_profile_cbf = self._can_do_profile_cbf
        can_do_cf = self._can_do_cf


        # Handle cold start user without query
        is_cold_start_user_no_query = self.profiles_df[self.profiles_df['user_id'] == user_id].empty \
                                     and self.interactions_df[self.interactions_df['user_id'] == user_id].empty \
                                     and not query.strip()

        if is_cold_start_user_no_query:
            return [], "Bạn là người dùng mới và chưa cung cấp truy vấn. Vui lòng nhập truy vấn để nhận gợi ý."

        # Handle case where no recommendation method can even attempt to retrieve candidates
        if not can_do_query_cbf and not can_do_profile_cbf and not can_do_cf:
             # This implies data for all potential methods is missing for this user/query
             return [], f"Không thể tìm kiếm ứng viên tài liệu. Vui lòng nhập truy vấn hoặc đảm bảo hồ sơ và lịch sử học tập của bạn đầy đủ."


        start_time = time.time()

        # --- 1. Retrieve candidates from available sources ---
        query_recs = self.retrieval_system.search(query, k=k_retrieval) if can_do_query_cbf else []
        profile_recs = self._get_cbf_profile_recommendations(user_id, k=k_retrieval) if can_do_profile_cbf else []
        cf_recs = self._get_cf_recommendations(user_id, k=k_retrieval) if can_do_cf else []

        # --- 2. Combine unique candidates ---
        unique_candidates = {}
        def add_candidates(recs_list):
            for rec in recs_list:
                doc_id = rec['doc_id']
                if doc_id not in unique_candidates:
                    unique_candidates[doc_id] = rec
                else:
                     # Keep the candidate with the highest score
                     if rec['score'] > unique_candidates[doc_id]['score']:
                          unique_candidates[doc_id] = rec

        add_candidates(query_recs)
        add_candidates(profile_recs)
        add_candidates(cf_recs)

        all_candidates_list = list(unique_candidates.values())

        if not all_candidates_list:
             return [], f"Không tìm thấy ứng viên tài liệu nào từ các nguồn gợi ý ({'Query' if can_do_query_cbf else ''} {'Profile' if can_do_profile_cbf else ''} {'CF' if can_do_cf else ''}). Vui lòng thử lại với truy vấn khác."


        # --- 3. Select and Filter/Rerank top candidates for RAG context ---
        MAX_DOCS_FOR_RAG_CONTEXT = 10

        # Apply heuristic reranking and sort
        user_profile_details = {}
        user_major = None
        user_profile_row = self.profiles_df[self.profiles_df['user_id'] == user_id]
        if not user_profile_row.empty:
             user_profile_details = user_profile_row.iloc[0].to_dict()
             user_major = user_profile_details.get('major')

        def apply_reranking_heuristic(candidate):
             boost = 0.0
             doc_details = candidate.get('details', {})
             doc_relevant_majors_str = doc_details.get('relevant_majors', '')

             if user_major and user_major.strip() and doc_relevant_majors_str.strip():
                  if user_major.strip().lower() in [m.strip().lower() for m in doc_relevant_majors_str.split(',')]:
                       boost = 0.1 # Add a small boost if major matches

             return candidate['score'] + boost

        candidates_with_rerank_score = []
        for candidate in all_candidates_list:
             rerank_score = apply_reranking_heuristic(candidate)
             candidates_with_rerank_score.append((rerank_score, candidate))

        sorted_candidates_for_context = [cand for score, cand in sorted(candidates_with_rerank_score, key=lambda item: item[0], reverse=True)][:MAX_DOCS_FOR_RAG_CONTEXT]


        if not sorted_candidates_for_context:
             return [], f"Không tìm thấy ứng viên tài liệu nào sau khi xếp hạng."


        # Prepare the document context list to pass to GPT
        docs_context_for_gpt = [candidate['details'] for candidate in sorted_candidates_for_context]


        # --- 4. Call OpenAI GPT-3.5 for RAG Completion ---
        # Pass user profile details to GPT for better personalization in explanation
        rag_response_text, rag_error = openai_utils.get_rag_completion(
            query=query,
            documents_context=docs_context_for_gpt,
            user_profile=user_profile_details # Pass profile details here
        )

        end_time = time.time()
        print(f"Total recommendation process time: {end_time - start_time:.2f} seconds.")

        # --- 5. Format the final response ---
        recommended_documents_with_explanation = []

        if rag_error:
             return [], f"Lỗi sinh giải thích gợi ý: {rag_error}. Vui lòng thử lại hoặc kiểm tra cấu hình OpenAI API."
        elif rag_response_text:
             # GPT generated a response. Format it.
             # We need to list the documents that were used as context (top_n from reranked list).
             docs_to_display = sorted_candidates_for_context[:top_n]

             if not docs_to_display:
                  # If GPT gave a response but no docs to link it to
                  return [], f"Hệ thống đã tìm được thông tin liên quan nhưng không thể liên kết trực tiếp với tài liệu cụ thể: {rag_response_text}"


             for doc in docs_to_display:
                  recommended_documents_with_explanation.append({
                      'doc_id': doc['doc_id'],
                      'title': doc['details'].get('title', f"Tài liệu ID: {doc['doc_id']}"),
                      'explanation': rag_response_text, # The main explanation is the combined GPT response
                      'source': doc.get('source_candidate', 'Retrieval') # Original source of this candidate
                  })

             return recommended_documents_with_explanation, None

        else:
             # If GPT didn't provide a response and no error
             return [], "Hệ thống không thể sinh giải thích gợi ý dựa trên các tài liệu tìm được."


# Example Usage (optional) - Ensure you have data files and OPENAI_API_KEY set
if __name__ == '__main__':
     print("Testing models/recommendation.py (RAG with OpenAI)")
     data_dir_path = '../data'
     # Check for data files and required columns for testing
     required_doc_cols = ['doc_id', 'title', 'abstract', 'keywords', 'subject', 'relevant_majors']
     required_profile_cols = ['user_id', 'major', 'year']
     required_interaction_cols = ['user_id', 'doc_id', 'action_type', 'timestamp']

     # Load data for testing the system
     documents_df = pd.DataFrame(columns=required_doc_cols)
     profiles_df = pd.DataFrame(columns=required_profile_cols)
     interactions_df = pd.DataFrame(columns=required_interaction_cols)

     try:
         # Attempt to load real data first
         if os.path.exists(os.path.join(data_dir_path, 'documents.csv')):
              temp_docs_df = pd.read_csv(os.path.join(data_dir_path, 'documents.csv'))
              temp_docs_df.rename(columns={'item_id': 'doc_id'}, inplace=True)
              # Ensure new columns exist even if not in file
              for col in ['subject', 'relevant_majors']:
                   if col not in temp_docs_df.columns:
                        temp_docs_df[col] = ''
              documents_df = temp_docs_df[['doc_id', 'title', 'abstract', 'keywords', 'subject', 'relevant_majors']]


         if os.path.exists(os.path.join(data_dir_path, 'student_profiles.csv')):
              profiles_df = pd.read_csv(os.path.join(data_dir_path, 'student_profiles.csv'))
              for col in ['major', 'year']:
                   if col not in profiles_df.columns:
                        profiles_df[col] = ''
              profiles_df = profiles_df[['user_id', 'major', 'year']]


         if os.path.exists(os.path.join(data_dir_path, 'interactions.csv')):
              interactions_df = pd.read_csv(os.path.join(data_dir_path, 'interactions.csv'))
              interactions_df.rename(columns={'content_id': 'doc_id'}, inplace=True)
              interactions_df['timestamp'] = pd.to_datetime(interactions_df['timestamp'], errors='coerce')
              interactions_df.dropna(subset=['timestamp'], inplace=True)
              interactions_df = interactions_df[['user_id', 'doc_id', 'action_type', 'timestamp']]


     except FileNotFoundError:
          print(f"Data files not found in {data_dir_path}. Using empty dataframes.")
     except Exception as e:
          print(f"Error loading data files for testing: {e}. Using potentially incomplete dataframes.")


     if not openai_utils.is_openai_ready:
          print("\nOpenAI API is not ready. Cannot initialize Recommendation System for testing.")
     elif documents_df.empty or not all(col in documents_df.columns for col in required_doc_cols):
          print("\nDocument data is empty or missing required columns. Cannot initialize Recommendation System for testing.")
     else:
          # Initialize the main RecommendationSystem (RAG based)
          # Pass the loaded dataframes for testing convenience
          # Set the system's dataframes after loading/checking in RecommendationSystem init
          system = RecommendationSystem(data_dir=data_dir_path)
          # Override the internal dataframes with the loaded ones for the test script
          system.documents_df = documents_df
          system.profiles_df = profiles_df
          system.interactions_df = interactions_df

          # Need to re-initialize retrieval system with the loaded documents_df if overriding
          if system.documents_df is not None and not system.documents_df.empty and openai_utils.is_openai_ready:
               print("Re-initializing Retrieval system with loaded test data...")
               system.retrieval_system = DocumentRetrieval(system.documents_df.copy())
               # Recheck _can_do flags after setting dataframes and retrieval system
               system._can_do_query_cbf = system.retrieval_system is not None and system.retrieval_system.is_ready
               system._can_do_profile_cbf = system._can_do_query_cbf and not system.profiles_df.empty and 'major' in system.profiles_df.columns
               system._can_do_cf = not system.interactions_df.empty and not system.documents_df.empty
               # Recheck overall readiness
               system.is_ready = system._can_do_query_cbf and openai_utils.is_openai_ready and all(col in system.documents_df.columns for col in required_doc_cols)

               if system.is_ready:
                    print("\nRecommendation System initialized successfully for testing (RAG).")

                    # Test users: Existing user (with profile/interactions if available), user with profile but no history, cold start user
                    test_user_full = system.profiles_df['user_id'].iloc[0] if not system.profiles_df.empty else 'SGU001_TEST'
                    user_with_interactions = system.interactions_df['user_id'].iloc[0] if not system.interactions_df.empty else None
                    if user_with_interactions and user_with_interactions != test_user_full:
                         test_user_full = user_with_interactions # Prioritize user with interactions if different

                    test_user_no_history = 'SGU002_TEST' # User with profile but no interactions
                    if 'SGU002_TEST' not in system.profiles_df['user_id'].tolist() and not system.profiles_df.empty:
                         # If SGU002_TEST doesn't exist, use an existing profile user who has no interactions
                         existing_profile_users = system.profiles_df['user_id'].tolist()
                         existing_interaction_users = system.interactions_df['user_id'].tolist()
                         users_with_profile_no_interaction = [u for u in existing_profile_users if u not in existing_interaction_users]
                         if users_with_profile_no_interaction:
                              test_user_no_history = users_with_profile_no_interaction[0]
                              print(f"Using existing user with profile but no interactions for test: {test_user_no_history}")
                         elif not system.profiles_df.empty:
                              # If no such user, use an arbitrary profile user and warn
                              test_user_no_history = system.profiles_df['user_id'].iloc[0]
                              print(f"Warning: Cannot find/create user with profile but no interactions. Using user {test_user_no_history} for this test.")
                         else:
                              test_user_no_history = 'N/A_NO_PROFILE' # Cannot test


                    test_user_cold_start = 'COLD_START_USER_TEST' # User with no profile or interactions

                    # Define test users to iterate through
                    test_users_to_run = []
                    if test_user_full != 'SGU001_TEST': test_users_to_run.append(test_user_full)
                    if test_user_no_history != 'N/A_NO_PROFILE': test_users_to_run.append(test_user_no_history)
                    test_users_to_run.append(test_user_cold_start)

                    # Ensure there's at least one user if all tests fall through
                    if not test_users_to_run:
                        test_users_to_run = ['SGU001_TEST'] # Add a minimal test user if data is very sparse


                    test_queries = ['mạng máy tính cơ bản', 'giải thuật sắp xếp', 'cơ sở dữ liệu', 'đại số tuyến tính', 'giới thiệu về marketing', 'Lập trình Web', 'Trí tuệ nhân tạo', 'Kinh tế Vi mô', ''] # Added diverse queries including empty


                    print(f"Testing with users: {test_users_to_run}")

                    for user_id in test_users_to_run:
                         # Get user profile for printing (handle cases where profile might not exist in original data)
                         current_user_profile = system.profiles_df[system.profiles_df['user_id'] == user_id].iloc[0].to_dict() if user_id in system.profiles_df['user_id'].tolist() else {'user_id': user_id, 'major': 'N/A', 'year': 'N/A'}
                         print(f"\n--- Testing RAG for User: {current_user_profile}, Query: ---")

                         for query in test_queries:
                             print(f"\n  Query: '{query}' ---")
                             recommendations, error = system.get_recommendations(user_id, query, top_n=3) # Reduce top_n for quicker test

                             if error:
                                 print(f"  Error: {error}")
                             elif recommendations:
                                 print("  Recommendations:")
                                 # The explanation is now the *same* GPT response for all docs returned
                                 combined_explanation = recommendations[0]['explanation']
                                 print(f"  --- Combined Explanation from AI ---\n  {combined_explanation}\n  ----------------------------------")
                                 print("  Relevant Documents Found (Top 3):")
                                 for i, rec in enumerate(recommendations):
                                     print(f"  {i+1}. {rec['title']} [ID: {rec['doc_id']}, Nguồn ứng viên: {rec['source']}]")

                             else:
                                 print("  No recommendations found.")

                    print("\nRecommendation System testing complete.")
               else:
                    print("\nRecommendation System failed to re-initialize with loaded data. Check logs above.")

          else:
              print("\nRecommendation System failed to initialize or is not ready. Check logs above.")
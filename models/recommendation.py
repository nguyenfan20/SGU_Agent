# models/recommendation.py (Using Sentence-BERT Retrieval and T5 Explanation)
import pandas as pd
import numpy as np
import time
import os
import faiss
from sentence_transformers import SentenceTransformer # Needed for Retrieval

# Import modules from the same package
from . import data_processing
from .retrieval import DocumentRetrieval # Now uses SBERT
from .explanation import ExplanationGenerator # Now uses T5 for per-document explanation

# Removed import openai_utils

class RecommendationSystem:
    # Re-added embedding_model_name and explanation_model_name parameters
    def __init__(self, data_dir='data',
                 embedding_model_name='sentence-transformers/all-MiniLM-L6-v2', # Default SBERT
                 explanation_model_name='VietAI/vit5-base'): # Default T5 Vietnamese

        self.data_dir = data_dir
        self.embedding_model_name = embedding_model_name
        self.explanation_model_name = explanation_model_name

        # 1. Load Data
        print("Loading data...")
        self.documents_df, self.profiles_df, self.interactions_df = data_processing.load_all_data(self.data_dir)
        print("Data loading complete.")

        # Check if crucial documents data is loaded and has necessary columns
        required_doc_cols_for_embedding = ['doc_id', 'title', 'abstract', 'keywords', 'subject', 'relevant_majors']
        if self.documents_df.empty or not all(col in self.documents_df.columns for col in required_doc_cols_for_embedding):
            missing_cols = [col for col in required_doc_cols_for_embedding if col not in self.documents_df.columns]
            print(f"FATAL: documents.csv is empty or missing required columns for Retrieval: {missing_cols}. Recommendation system cannot function.")
            self.is_ready = False
            self.retrieval_system = None
            self.explanation_system = None # Explanation also needs doc details
            # Ensure dataframes are initialized even if empty for safety
            self.documents_df = pd.DataFrame(columns=required_doc_cols_for_embedding) # Use embedding cols as base
            self.profiles_df = pd.DataFrame(columns=['user_id', 'major', 'year'])
            self.interactions_df = pd.DataFrame(columns=['user_id', 'doc_id', 'action_type', 'timestamp'])
            return


        # 2. Initialize components (Retrieval, Explanation)
        self.retrieval_system = DocumentRetrieval(
            documents_df=self.documents_df.copy(),
            embedding_model_name=self.embedding_model_name # Pass SBERT model name
        )
        self.explanation_system = ExplanationGenerator(
             profiles_df=self.profiles_df.copy(),
             model_name=self.explanation_model_name # Pass T5 model name
        )

        # Recalculate capabilities based on data availability and component readiness
        self._can_do_query_cbf = self.retrieval_system is not None and self.retrieval_system.is_ready
        # Profile CBF needs profiles data AND retrieval system
        self._can_do_profile_cbf = self._can_do_query_cbf and not self.profiles_df.empty and 'major' in self.profiles_df.columns
        # CF needs interactions data AND documents data (for details)
        self._can_do_cf = not self.interactions_df.empty and not self.documents_df.empty


        # System is "ready" if Retrieval and Explanation systems initialized AND documents data is loaded with necessary columns
        self.is_ready = (self._can_do_query_cbf and
                         self.explanation_system is not None and self.explanation_system.is_ready and
                         not self.documents_df.empty and all(col in self.documents_df.columns for col in required_doc_cols_for_embedding))


        if not self.is_ready:
             print("Warning: Recommendation system may not function fully due to initialization issues or missing data.")
             if not self._can_do_query_cbf: print("- Query CBF is unavailable (Check documents and Retrieval system init).")
             if not self.explanation_system or not self.explanation_system.is_ready: print("- Explanation Generator is unavailable (Check T5 model load).")
             if not self._can_do_profile_cbf: print("- CBF Profile is unavailable (Check profiles data and retrieval readiness).")
             if not self._can_do_cf: print("- CF is unavailable (Check interactions and documents data).")


    # --- Core Recommendation Methods (Keep these for candidate generation) ---

    def _get_cbf_profile_recommendations(self, user_id, k=10):
        # ... (This method remains the same, uses retrieval_system.search with profile query) ...
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

        profile_query_string = ""
        if user_major and user_major.strip():
             profile_query_string += f"Tài liệu học tập cho sinh viên ngành {user_major}"
             if user_year and str(user_year).strip() and str(user_year) != '0':
                  profile_query_string += f" năm {user_year}"
             profile_query_string += "."
        elif user_year and str(user_year).strip() and str(user_year) != '0':
             profile_query_string += f"Tài liệu học tập cho sinh viên năm {user_year}."
        else:
            print(f"User {user_id} has insufficient profile data (major/year) for CBF (Profile).")
            return []

        if not profile_query_string.strip(): return []

        profile_query_results = self.retrieval_system.search(profile_query_string, k=k)

        for rec in profile_query_results:
            rec['source_candidate'] = 'CBF_Profile'
        return profile_query_results


    def _get_cf_recommendations(self, user_id, k=10):
        # ... (This method remains the same, returns list of dicts with doc_id, score, source_candidate, details) ...
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
        candidate_docs_scores = {}

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
                 candidate_docs_scores[doc_id] += similarity

        recommendations = []
        doc_info_map = self.documents_df.set_index('doc_id').to_dict('index')

        sorted_candidate_ids = sorted(candidate_docs_scores, key=candidate_docs_scores.get, reverse=True)

        for doc_id in sorted_candidate_ids[:k]:
             doc_details = doc_info_map.get(doc_id)
             if doc_details:
                 recommendations.append({
                     'doc_id': doc_id,
                     'score': candidate_docs_scores[doc_id],
                     'source_candidate': 'CF',
                     'details': doc_details
                 })
             else:
                 print(f"Warning: CF recommended doc ID {doc_id} not found in documents_df.")
                 recommendations.append({
                      'doc_id': doc_id,
                      'score': candidate_docs_scores[doc_id],
                      'source_candidate': 'CF',
                      'details': {'doc_id': doc_id, 'title': f'Tài liệu ID: {doc_id}', 'abstract': '', 'keywords': '', 'subject': '', 'relevant_majors': ''}
                 })


        print(f"Found {len(recommendations)} CF recommendations.")
        return recommendations


    # --- Helper for combining and ranking candidates (Simplified from RAG version) ---
    # This is needed to get a single ranked list before generating explanations
    def _combine_and_rank_candidates(self, query_recs, profile_recs, cf_recs, top_k=20):
        """Combines unique candidates and ranks them based on score and source."""
        unique_candidates = {}
        # Add candidates from each source. Prioritize Query, then Profile, then CF if score is similar
        def add_candidates(recs_list):
            for rec in recs_list:
                doc_id = rec['doc_id']
                if doc_id not in unique_candidates:
                    unique_candidates[doc_id] = rec # Store the whole rec dict
                else:
                     # Keep the candidate with the highest score
                     if rec['score'] > unique_candidates[doc_id]['score']:
                          unique_candidates[doc_id] = rec

        add_candidates(query_recs)
        add_candidates(profile_recs)
        add_candidates(cf_recs)

        all_candidates_list = list(unique_candidates.values())

        if not all_candidates_list:
             return []

        # Sort candidates by their score (higher is better)
        # You could add reranking heuristic here if desired, before taking top_k
        sorted_candidates = sorted(all_candidates_list, key=lambda x: x['score'], reverse=True)[:top_k]

        print(f"Combined and ranked down to {len(sorted_candidates)} unique candidates.")
        return sorted_candidates


    def get_recommendations(self, user_id, query, top_n=5, k_retrieval=20):
        """
        Orchestrates the document recommendation process.
        1. Retrieve candidates from various sources (Query CBF, Profile CBF, CF).
        2. Combine unique candidates and rank them.
        3. Generate explanation for each of the top N ranked documents.
        Returns: tuple (list of recommended_documents_with_explanation, error_message)
        """
        # Check core system readiness (Retrieval + Explanation + Documents)
        if not self.is_ready:
             return [], "Hệ thống gợi ý tài liệu chưa sẵn sàng. Vui lòng kiểm tra quá trình khởi tạo (mô hình Embedding, FAISS, mô hình Giải thích, dữ liệu)."

        # Determine which recommendation methods are possible based on data availability
        can_do_query_cbf = query and self._can_do_query_cbf
        can_do_profile_cbf = self._can_do_profile_cbf
        can_do_cf = self._can_do_cf


        # Handle cold start user without query or cases where no method is possible
        if not can_do_query_cbf and not can_do_profile_cbf and not can_do_cf:
             is_cold_start_user_no_query = self.profiles_df[self.profiles_df['user_id'] == user_id].empty \
                                          and self.interactions_df[self.interactions_df['user_id'] == user_id].empty \
                                          and not query.strip()

             if is_cold_start_user_no_query:
                 return [], "Bạn là người dùng mới và chưa cung cấp truy vấn. Vui lòng nhập truy vấn để nhận gợi ý."

             # If system is ready but no data sources apply for this user/query combo
             if not query.strip() and not can_do_profile_cbf and not can_do_cf:
                 return [], "Vui lòng nhập truy vấn hoặc đảm bảo hồ sơ và lịch sử học tập của bạn đầy đủ để nhận gợi ý."

             # Fallback if methods could potentially run but returned no candidates (should be caught later too)
             # This message might be redundant if _combine_and_rank_candidates also returns empty.
             pass # Let the combination check handle the empty case


        start_time = time.time()

        # --- 1. Retrieve candidates from available sources ---
        # Retrieval results now include 'details' and 'score'
        query_recs = self.retrieval_system.search(query, k=k_retrieval) if can_do_query_cbf else []
        profile_recs = self._get_cbf_profile_recommendations(user_id, k=k_retrieval) if can_do_profile_cbf else []
        cf_recs = self._get_cf_recommendations(user_id, k=k_retrieval) if can_do_cf else []

        # --- 2. Combine unique candidates and rank them ---
        # Using a helper method for clarity
        ranked_candidates = self._combine_and_rank_candidates(query_recs, profile_recs, cf_recs, top_k=k_retrieval * 2) # Get more candidates before final selection


        if not ranked_candidates:
             return [], f"Không tìm thấy ứng viên tài liệu nào từ các nguồn gợi ý ({'Query' if can_do_query_cbf else ''} {'Profile' if can_do_profile_cbf else ''} {'CF' if can_do_cf else ''}). Vui lòng thử lại với truy vấn khác."

        # Select the top N documents to generate explanations for
        final_recommendations_to_explain = ranked_candidates[:top_n]

        if not final_recommendations_to_explain:
             # This case means ranked_candidates was less than top_n, but not empty.
             # It's still a valid result, just fewer than requested.
             # So, don't return an error here, just proceed with explanation generation.
             pass # continue


        # --- 3. Generate Explanations for each of the top N documents ---
        recommended_documents_with_explanation = []
        # ExplanationGenerator needs the full document details for context
        doc_info_map = self.documents_df.set_index('doc_id').to_dict('index')

        for rec in final_recommendations_to_explain:
            doc_id = rec['doc_id']
            doc_info = doc_info_map.get(doc_id, {}) # Get full document details

            # Call ExplanationGenerator for each document
            # ExplanationGenerator handles its own readiness check
            explanation = self.explanation_system.generate(doc_info, query, user_id, rec['source_candidate'])

            recommended_documents_with_explanation.append({
                'doc_id': doc_id,
                'title': rec['details'].get('title', f'Tài liệu ID: {doc_id}'), # Use title from details
                'explanation': explanation, # Explanation generated by T5/Template
                'source': rec['source_candidate'] # Original source of this candidate
            })

        end_time = time.time()
        print(f"Total recommendation process time: {end_time - start_time:.2f} seconds.")

        # Final check if any recommendations were produced with explanations
        if not recommended_documents_with_explanation:
             # This could happen if explanation generation failed for all candidates
             return [], f"Hệ thống đã tìm được tài liệu liên quan nhưng không thể sinh giải thích."


        return recommended_documents_with_explanation, None


# Example Usage (optional) - Ensure you have data files and T5 model is loadable
if __name__ == '__main__':
     print("Testing models/recommendation.py (SBERT Retrieval, T5 Explanation)")
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
                        temp_docs_df[col] = '' # Add missing columns
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


     # Need to ensure Sentence-BERT model can be loaded and documents have required cols
     if documents_df.empty or not all(col in documents_df.columns for col in required_doc_cols):
          print("\nDocument data is empty or missing required columns. Cannot initialize Recommendation System for testing.")
     else:
          # Initialize the main RecommendationSystem
          system = RecommendationSystem(data_dir=data_dir_path) # init will load internally

          # Override the internal dataframes with the loaded ones for the test script
          # Note: RecommendationSystem.__init__ already loads data internally,
          # this override is mainly for testing with manually loaded/modified dataframes.
          # If you trust RecommendationSystem's internal loading, you don't need this.
          system.documents_df = documents_df
          system.profiles_df = profiles_df
          system.interactions_df = interactions_df

          # Need to re-initialize retrieval and explanation systems with the loaded dataframes if overriding
          # Re-initializing Retrieval system with the loaded documents_df
          print("Re-initializing Retrieval system with loaded test data...")
          system.retrieval_system = DocumentRetrieval(system.documents_df.copy()) # Pass copy

          # Re-initializing Explanation system with loaded profiles_df
          print("Re-initializing Explanation system with loaded test data...")
          system.explanation_system = ExplanationGenerator(system.profiles_df.copy()) # Pass copy


          # Recheck _can_do flags after setting dataframes and components
          system._can_do_query_cbf = system.retrieval_system is not None and system.retrieval_system.is_ready
          system._can_do_profile_cbf = system._can_do_query_cbf and not system.profiles_df.empty and 'major' in system.profiles_df.columns
          system._can_do_cf = not system.interactions_df.empty and not system.documents_df.empty
          # Recheck overall readiness
          system.is_ready = (system._can_do_query_cbf and
                              system.explanation_system is not None and system.explanation_system.is_ready and
                              not system.documents_df.empty and all(col in system.documents_df.columns for col in required_doc_cols))


          if system.is_ready:
              print("\nRecommendation System initialized successfully for testing.")

              # Test users: Existing user, user with profile but no history, cold start user
              # Use user IDs from the loaded data for realistic testing
              test_user_full = system.profiles_df['user_id'].iloc[0] if not system.profiles_df.empty else 'SGU001_TEST'
              user_with_interactions = system.interactions_df['user_id'].iloc[0] if not system.interactions_df.empty else None
              if user_with_interactions and user_with_interactions != test_user_full:
                  test_user_full = user_with_interactions

              test_user_no_history = 'SGU002_TEST'
              if 'SGU002_TEST' not in system.profiles_df['user_id'].tolist() and not system.profiles_df.empty:
                  existing_profile_users = system.profiles_df['user_id'].tolist()
                  existing_interaction_users = system.interactions_df['user_id'].tolist()
                  users_with_profile_no_interaction = [u for u in existing_profile_users if u not in existing_interaction_users]
                  if users_with_profile_no_interaction:
                       test_user_no_history = users_with_profile_no_interaction[0]
                       print(f"Using existing user with profile but no interactions for test: {test_user_no_history}")
                  elif not system.profiles_df.empty:
                       test_user_no_history = system.profiles_df['user_id'].iloc[0]
                       print(f"Warning: Cannot find/create user with profile but no interactions. Using user {test_user_no_history} for this test.")
                  else:
                       test_user_no_history = 'N/A_NO_PROFILE'

              test_user_cold_start = 'COLD_START_USER_TEST'

              test_users_to_run = []
              if test_user_full != 'SGU001_TEST': test_users_to_run.append(test_user_full)
              if test_user_no_history != 'N/A_NO_PROFILE': test_users_to_run.append(test_user_no_history)
              test_users_to_run.append(test_user_cold_start)

              if not test_users_to_run:
                  test_users_to_run = ['SGU001_TEST'] # Add a minimal test user if data is very sparse


              test_queries = ['mạng máy tính cơ bản', 'giải thuật sắp xếp', 'cơ sở dữ liệu', 'đại số tuyến tính', 'giới thiệu về marketing', 'Lập trình Web', 'Trí tuệ nhân tạo', 'Kinh tế Vi mô', '']


              print(f"Testing with users: {test_users_to_run}")

              for user_id in test_users_to_run:
                   current_user_profile = system.profiles_df[system.profiles_df['user_id'] == user_id].iloc[0].to_dict() if user_id in system.profiles_df['user_id'].tolist() else {'user_id': user_id, 'major': 'N/A', 'year': 'N/A'}
                   print(f"\n--- Testing for User: {current_user_profile}, Query: ---")

                   for query in test_queries:
                       print(f"\n  Query: '{query}' ---")
                       recommendations, error = system.get_recommendations(user_id, query, top_n=3)

                       if error:
                           print(f"  Error: {error}")
                       elif recommendations:
                           print("  Recommendations:")
                           # Each recommendation now has its own explanation from T5
                           for i, rec in enumerate(recommendations):
                               print(f"  {i+1}. {rec['title']} [ID: {rec['doc_id']}, Nguồn ứng viên: {rec['source']}]")
                               print(f"     Giải thích: {rec['explanation']}")

                       else:
                           print("  No recommendations found.")

              print("\nRecommendation System testing complete.")
          else:
              print("\nRecommendation System failed to initialize or is not ready. Check logs above.")
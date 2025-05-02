# models/retrieval.py (Using Sentence-BERT Vietnamese/Multilingual)
import faiss
import numpy as np
import pandas as pd
import os
from sentence_transformers import SentenceTransformer # Added back

# Removed import openai_utils

# Choose a Vietnamese or Multilingual Sentence-BERT model
# Example: keepitreal/vietnamese-sbert (might need testing)
# Example: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (multilingual)
# Example: sentence-transformers/all-MiniLM-L6-v2 (multilingual, smaller)
# Let's use all-MiniLM-L6-v2 as it's relatively small and good balance
SBERT_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2' # Or keepitreal/vietnamese-sbert

class DocumentRetrieval:
    # Removed dependency on openai_utils being ready in __init__
    def __init__(self, documents_df, embedding_model_name=SBERT_MODEL_NAME):
        self.documents_df = documents_df
        self.embedding_model_name = embedding_model_name
        self.embedding_model = None # Added back
        self.faiss_index = None
        self.doc_embeddings = None
        self._valid_doc_ids = []
        self.is_ready = False # Flag to indicate if index is ready

        # Check if document data exists, is not empty, and has required columns
        # Required columns for embedding string (subject, relevant_majors)
        required_doc_cols_for_embedding = ['doc_id', 'title', 'abstract', 'keywords', 'subject', 'relevant_majors']
        if self.documents_df is None or self.documents_df.empty or not all(col in self.documents_df.columns for col in required_doc_cols_for_embedding):
             missing_cols = [col for col in required_doc_cols_for_embedding if col not in self.documents_df.columns]
             print(f"No documents loaded, or missing required columns {missing_cols}. Skipping FAISS index build.")
             # Ensure dataframes are initialized even if empty for safety
             self.documents_df = pd.DataFrame(columns=required_doc_cols_for_embedding) # Use embedding cols as base
             return # Cannot build index without documents + required columns

        self._load_embedding_model() # Load SBERT model

        if self.embedding_model is not None:
            self._build_faiss_index() # Build index if SBERT loaded

        if self.faiss_index is not None:
            self.is_ready = True
            print("Document Retrieval system ready.")
        else:
            print("Document Retrieval system not ready: FAISS index build failed.")


    def _load_embedding_model(self):
        print(f"Loading embedding model: {self.embedding_model_name}")
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            print("Embedding model loaded successfully.")
        except Exception as e:
             print(f"Error loading embedding model {self.embedding_model_name}: {e}")
             self.embedding_model = None


    def _build_faiss_index(self):
        # Redundant checks
        required_doc_cols_for_embedding = ['doc_id', 'title', 'abstract', 'keywords', 'subject', 'relevant_majors']
        if self.documents_df is None or self.documents_df.empty or self.embedding_model is None or not all(col in self.documents_df.columns for col in required_doc_cols_for_embedding):
             print("Cannot build FAISS index: Document data missing, incomplete, or embedding model not loaded.")
             self.faiss_index = None
             self.doc_embeddings = None
             self._valid_doc_ids = []
             return


        print("Creating document embeddings using Sentence-BERT...")
        texts_to_embed = self.documents_df.apply(
            # Combine fields into a single string for embedding
            lambda row: f"Môn học: {row['subject']}. Ngành liên quan: {row['relevant_majors']}. Tiêu đề: {row['title']}. Tóm tắt: {row['abstract']}. Từ khóa: {row['keywords']}".strip(), axis=1
        ).tolist()

        # Use SBERT model to get embeddings
        # Process in batches if needed
        embeddings_list = []
        self._valid_doc_ids = [] # Store doc_ids corresponding to embeddings in embeddings_list

        # Process embeddings and filter invalid ones (SBERT usually doesn't return NaN unless input is weird)
        embeddings = self.embedding_model.encode(texts_to_embed, convert_to_numpy=True)

        for index, embedding in enumerate(embeddings):
             doc_id = self.documents_df.iloc[index]['doc_id']
             # Check for NaN/Inf if SBERT could produce them (less likely than OpenAI errors)
             if embedding is not None and not np.isnan(embedding).any() and not np.isinf(embedding).any():
                 embeddings_list.append(embedding)
                 self._valid_doc_ids.append(doc_id)
             else:
                 print(f"Warning: Failed to get valid embedding for document ID {doc_id}. Skipping.")


        if not embeddings_list:
            print("No valid document embeddings were created.")
            self.faiss_index = None
            self.doc_embeddings = None
            self._valid_doc_ids = []
            return

        self.doc_embeddings = np.vstack(embeddings_list) # Stack embeddings into a numpy array
        print(f"Successfully created {len(self.doc_embeddings)} embeddings. Shape: {self.doc_embeddings.shape}")

        dimension = self.doc_embeddings.shape[1]
        # No fixed dimension check like OpenAI, use the one from the model

        print(f"Building FAISS index with dimension {dimension} (using HNSW)...")
        try:
            hnsw_m = 32
            # --- CORRECTED IndexHNSWFlat Initialization ---
            self.faiss_index = faiss.IndexHNSWFlat(dimension, hnsw_m)
            # --- END CORRECTED ---

            self.faiss_index.add(self.doc_embeddings)

            hnsw_ef_search = 64
            self.faiss_index.hnsw.efSearch = hnsw_ef_search

            print(f"FAISS index built with {self.faiss_index.ntotal} vectors.")

        except Exception as e:
            print(f"Error building FAISS index: {e}")
            self.faiss_index = None
            self.doc_embeddings = None
            self._valid_doc_ids = []


    def search(self, query, k=10):
        """Performs semantic search on document embeddings using FAISS."""
        # Check if Retrieval system is ready and embedding model loaded
        if not self.is_ready or self.embedding_model is None:
            print("Document Retrieval system is not ready.")
            return []

        if not query or not query.strip():
             print("Empty query provided for search in retrieval.")
             return []

        print(f"Performing FAISS search for query: '{query}'")
        try:
            # Get embedding for the query using Sentence-BERT
            query_embedding = self.embedding_model.encode([query.strip()], convert_to_numpy=True)
            query_embedding = query_embedding[0] # Get the single embedding

            if query_embedding is None or np.isnan(query_embedding).any():
                 print("Failed to get embedding for the query.")
                 return []

            query_embedding = np.array([query_embedding]) # Reshape for faiss.search

            distances, indices = self.faiss_index.search(query_embedding, k)

            recommendations = []
            if indices is not None and indices.size > 0 and indices.shape[1] > 0:
                 for i in range(indices.shape[1]):
                     doc_embedding_index = indices[0][i]
                     distance = distances[0][i]

                     if doc_embedding_index != -1 and doc_embedding_index < len(self._valid_doc_ids):
                           doc_id = self._valid_doc_ids[doc_embedding_index]

                           score = 1.0 / (1.0 + distance) if distance >= 0 else 0.0

                           # Fetch the full document details from the original dataframe using doc_id
                           doc_details_row = self.documents_df[self.documents_df['doc_id'] == doc_id]
                           if not doc_details_row.empty:
                                doc_details = doc_details_row.iloc[0].to_dict()
                                recommendations.append({
                                    'doc_id': doc_id,
                                    'score': score,
                                    'source_candidate': 'CBF_Query_Retrieval',
                                    'details': doc_details
                                })
                           else:
                               print(f"Warning: Doc ID {doc_id} found in FAISS but not in original documents_df.")
                               recommendations.append({
                                   'doc_id': doc_id,
                                   'score': score,
                                   'source_candidate': 'CBF_Query_Retrieval',
                                   'details': {'doc_id': doc_id, 'title': f'Tài liệu ID: {doc_id} (Chi tiết không đủ)', 'abstract': '', 'keywords': '', 'subject': '', 'relevant_majors': ''}
                               })

            print(f"Retrieved {len(recommendations)} documents from FAISS for query '{query}'.")
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            return recommendations

        except Exception as e:
            print(f"Error during FAISS search for query '{query}': {e}")
            return []
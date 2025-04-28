# models/retrieval.py (Final version using OpenAI Embedding and FAISS)
import faiss
import numpy as np
import pandas as pd
import os

# Import OpenAI utility module
import openai_utils # Provides get_embedding and is_openai_ready

class DocumentRetrieval:
    def __init__(self, documents_df):
        self.documents_df = documents_df
        self.faiss_index = None
        self.doc_embeddings = None
        self._valid_doc_ids = [] # Store doc_ids corresponding to embeddings
        self.is_ready = False # Flag to indicate if index is ready

        if not openai_utils.is_openai_ready:
             print("OpenAI API is not ready for embedding. Document Retrieval will not be available.")
             return # Cannot proceed without embedding

        # Check if document data exists, is not empty, and has required columns
        required_doc_cols = ['doc_id', 'title', 'abstract', 'keywords', 'subject', 'relevant_majors']
        if self.documents_df is None or self.documents_df.empty or not all(col in self.documents_df.columns for col in required_doc_cols):
             missing_cols = [col for col in required_doc_cols if col not in self.documents_df.columns]
             print(f"No documents loaded, or missing required columns {missing_cols}. Skipping FAISS index build.")
             return # Cannot build index without documents


        self._build_faiss_index()

        if self.faiss_index is not None:
            self.is_ready = True
            print("Document Retrieval system ready.")
        else:
            print("Document Retrieval system not ready: FAISS index build failed.")


    def _build_faiss_index(self):
        required_doc_cols = ['doc_id', 'title', 'abstract', 'keywords', 'subject', 'relevant_majors']
        if self.documents_df is None or self.documents_df.empty or not all(col in self.documents_df.columns for col in required_doc_cols):
             print("Cannot build FAISS index: Document data missing or incomplete.")
             self.faiss_index = None
             self.doc_embeddings = None
             self._valid_doc_ids = []
             return

        print("Creating document embeddings using OpenAI...")
        texts_to_embed = self.documents_df.apply(
            # Combine subject, relevant_majors, title, abstract, keywords
            lambda row: f"Môn học: {row['subject']}. Ngành liên quan: {row['relevant_majors']}. Tiêu đề: {row['title']}. Tóm tắt: {row['abstract']}. Từ khóa: {row['keywords']}".strip(), axis=1
        ).tolist()

        embeddings_list = []
        self._valid_doc_ids = [] # Store doc_ids corresponding to embeddings in embeddings_list

        # Process embeddings and filter invalid ones
        # Batching could be implemented here for larger datasets
        for index, text in enumerate(texts_to_embed):
             doc_id = self.documents_df.iloc[index]['doc_id']
             embedding = openai_utils.get_embedding(text)
             # Check if embedding is not None and doesn't contain NaN/Inf
             if embedding is not None and not np.isnan(embedding).any() and not np.isinf(embedding).any():
                 embeddings_list.append(embedding)
                 self._valid_doc_ids.append(doc_id)
             else:
                 print(f"Warning: Failed to get valid embedding for document ID {doc_id} ('{text[:50]}...'). Skipping.")


        if not embeddings_list:
            print("No valid document embeddings were created.")
            self.faiss_index = None
            self.doc_embeddings = None
            self._valid_doc_ids = []
            return

        # Convert list of embeddings to numpy array
        self.doc_embeddings = np.vstack(embeddings_list)
        print(f"Successfully created {len(self.doc_embeddings)} embeddings. Shape: {self.doc_embeddings.shape}")

        dimension = self.doc_embeddings.shape[1]
        # Ensure dimension matches the expected OpenAI embedding dimension
        if dimension != openai_utils.OPENAI_EMBEDDING_DIMENSION:
            print(f"Error: Embedding dimension mismatch. Expected {openai_utils.OPENAI_EMBEDDING_DIMENSION}, got {dimension}.")
            self.faiss_index = None
            self.doc_embeddings = None
            self._valid_doc_ids = []
            return


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
        if not self.is_ready or not openai_utils.is_openai_ready:
            print("Document Retrieval system is not ready or OpenAI API is not available.")
            return []

        if not query or not query.strip():
             print("Empty query provided for search in retrieval.")
             return []

        print(f"Performing FAISS search for query: '{query}'")
        try:
            query_embedding = openai_utils.get_embedding(query)

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

                     # Ensure a valid document was found and the index is within bounds
                     if doc_embedding_index != -1 and doc_embedding_index < len(self._valid_doc_ids):
                        doc_id = self._valid_doc_ids[doc_embedding_index]

                          # Score: higher is better. Use 1 / (1 + distance) for L2 distance.
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
                               # Include basic entry with empty fields for subject/relevant_majors
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
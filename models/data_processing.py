# models/data_processing.py (Final version for new data format)
import pandas as pd
import os

def load_documents(path='data/documents.csv'):
    """Loads and preprocesses the documents data (item_id, title, abstract, keywords, subject, relevant_majors)."""
    try:
        df = pd.read_csv(path)

        # Validate required columns (added subject, relevant_majors)
        # relevant_majors might be missing in some datasets, subject is essential for the new logic
        required_cols_essential = ['item_id', 'title', 'abstract', 'keywords', 'subject']
        required_cols_helpful = ['relevant_majors'] # This one is used in embedding and reranking heuristic

        if not all(col in df.columns for col in required_cols_essential):
             missing = [col for col in required_cols_essential if col not in df.columns]
             print(f"Error: Document file at {path} is missing essential columns: {missing}.")
             return pd.DataFrame(columns=required_cols_essential + required_cols_helpful) # Return empty DataFrame on critical error

        # Rename item_id to doc_id
        df.rename(columns={'item_id': 'doc_id'}, inplace=True)

        # Ensure all expected columns exist, even if missing in file (fill with empty string)
        all_expected_cols = ['doc_id'] + [col for col in required_cols_essential if col != 'item_id'] + required_cols_helpful
        for col in all_expected_cols:
             if col not in df.columns:
                 df[col] = '' # Add missing column with empty strings

        # Handle potential NaN in text fields and new columns
        text_cols = ['title', 'abstract', 'keywords', 'subject', 'relevant_majors']
        for col in text_cols:
             df[col] = df[col].fillna('').astype(str)


        print(f"Loaded {len(df)} documents from {path}. Renamed 'item_id' to 'doc_id'.")
        # Return all relevant columns
        return df[all_expected_cols]

    except FileNotFoundError:
        print(f"Error: Document file not found at {path}")
        return pd.DataFrame(columns=['doc_id', 'title', 'abstract', 'keywords', 'subject', 'relevant_majors'])
    except Exception as e:
        print(f"Error loading documents from {path}: {e}")
        return pd.DataFrame(columns=['doc_id', 'title', 'abstract', 'keywords', 'subject', 'relevant_majors'])


def load_student_profiles(path='data/student_profiles.csv'):
    """Loads and preprocesses the student profiles data (user_id, major, year)."""
    try:
        df = pd.read_csv(path)
        required_cols = ['user_id', 'major', 'year']
        if not all(col in df.columns for col in required_cols):
             missing = [col for col in required_cols if col not in df.columns]
             print(f"Warning: Student profile file at {path} is missing required columns: {missing}.")
             # Ensure expected columns exist
             for col in required_cols:
                  if col not in df.columns:
                       df[col] = ''
        df['major'] = df['major'].fillna('').astype(str)
        # Convert year to string, fill NaN with empty string for consistency
        df['year'] = df['year'].fillna('').astype(str)


        print(f"Loaded {len(df)} student profiles from {path}.")
        return df[['user_id', 'major', 'year']]
    except FileNotFoundError:
        print(f"Error: Student profile file not found at {path}")
        return pd.DataFrame(columns=['user_id', 'major', 'year'])
    except Exception as e:
        print(f"Error loading student profiles from {path}: {e}")
        return pd.DataFrame(columns=['user_id', 'major', 'year'])


def load_interactions(path='data/interactions.csv'):
    """Loads and preprocesses the interactions data (user_id, content_id, action_type, timestamp)."""
    try:
        df = pd.read_csv(path)
        required_cols = ['user_id', 'content_id', 'action_type', 'timestamp']
        if not all(col in df.columns for col in required_cols):
             missing = [col for col in required_cols if col not in df.columns]
             print(f"Warning: Interaction file at {path} is missing required columns: {missing}.")
              # Ensure expected columns exist
             for col in required_cols:
                  if col not in df.columns:
                       df[col] = ''

        df.rename(columns={'content_id': 'doc_id'}, inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)
        df['action_type'] = df['action_type'].fillna('').astype(str)
        print(f"Loaded {len(df)} interactions from {path}. Renamed 'content_id' to 'doc_id'.")
        return df[['user_id', 'doc_id', 'action_type', 'timestamp']]

    except FileNotFoundError:
        print(f"Error: Interaction file not found at {path}")
        return pd.DataFrame(columns=['user_id', 'doc_id', 'action_type', 'timestamp'])
    except Exception as e:
        print(f"Error loading interactions from {path}: {e}")
        return pd.DataFrame(columns=['user_id', 'doc_id', 'action_type', 'timestamp'])


def load_all_data(data_dir='data'):
    """Loads all data files using the updated column names and renames content_id/item_id to doc_id."""
    doc_path = os.path.join(data_dir, 'documents.csv')
    profile_path = os.path.join(data_dir, 'student_profiles.csv')
    interactions_path = os.path.join(data_dir, 'interactions.csv')

    documents_df = load_documents(doc_path)
    profiles_df = load_student_profiles(profile_path)
    interactions_df = load_interactions(interactions_path)

    # Basic check if data is loaded and contains required columns after processing
    # More detailed checks are done within the load_* functions and RecommendationSystem init
    if documents_df.empty:
         print("Warning: Documents data is empty.")
    if profiles_df.empty:
         print("Warning: Student profiles data is empty.")
    if interactions_df.empty:
         print("Warning: Interactions data is empty.")

    return documents_df, profiles_df, interactions_df
# history_manager.py (Updated for SQLite)
import sqlite3
import os
import datetime

# Define the path for the SQLite database file
DB_PATH = 'data/query_history.db'

def _connect_db():
    """Helper to create a database connection."""
    data_dir = os.path.dirname(DB_PATH)
    if data_dir and not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created data directory: {data_dir}")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row # Allows accessing columns by name
    return conn

def init_db():
    """Initializes the database, creating the table if it doesn't exist."""
    conn = _connect_db()
    with conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS query_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                query TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
    conn.close()
    print(f"Database initialized at {DB_PATH}")


def add_query_to_history(user_id, query):
    """Appends a user's query and timestamp to the history database."""
    if not user_id or not query.strip():
        # Don't log empty queries or empty user_ids
        return

    conn = _connect_db()
    try:
        with conn:
            conn.execute('''
                INSERT INTO query_history (user_id, query, timestamp)
                VALUES (?, ?, ?)
            ''', (user_id, query.strip(), datetime.datetime.now()))
        # print(f"Logged query for user {user_id}: '{query.strip()}'") # Optional: log to console
    except Exception as e:
        print(f"Error adding query to history database: {e}")
    finally:
        conn.close()

def get_query_history(user_id, limit=None):
    """Reads query history for a specific user from the database, newest first."""
    if not user_id:
        return []

    conn = _connect_db()
    history = []
    try:
        cursor = conn.cursor()
        query_sql = '''
            SELECT user_id, query, timestamp FROM query_history
            WHERE user_id = ?
            ORDER BY timestamp DESC
        '''
        if limit is not None:
            query_sql += f' LIMIT {int(limit)}' # Add limit clause

        cursor.execute(query_sql, (user_id,))
        history_rows = cursor.fetchall()

        # Convert rows to list of dictionaries
        for row in history_rows:
            history.append({
                'user_id': row['user_id'],
                'query': row['query'],
                'timestamp': row['timestamp'] # Timestamp is already in a useful format from DB
            })

        # print(f"Retrieved {len(history)} history entries for user {user_id}.") # Optional: log

    except Exception as e:
        print(f"Error reading query history from database for user {user_id}: {e}")
        history = [] # Return empty list on error
    finally:
        conn.close()

    return history

# Example Usage (optional)
if __name__ == '__main__':
    print("Testing history_manager.py with SQLite")
    # Initialize the database
    init_db()

    # Add some dummy history
    add_query_to_history('test_user_001', 'lập trình python')
    add_query_to_history('test_user_002', 'cơ sở dữ liệu nâng cao')
    add_query_to_history('test_user_001', 'flask web development')
    add_query_to_history('test_user_003', 'linear algebra')
    add_query_to_history('test_user_001', 'machine learning basics')

    # Get history for test_user_001
    user1_history = get_query_history('test_user_001')
    print("\nHistory for test_user_001:")
    for entry in user1_history:
        print(f"- Query: '{entry['query']}' at {entry['timestamp']}")

    # Get history for test_user_002 (latest 2)
    user2_history = get_query_history('test_user_002', limit=2)
    print("\nHistory for test_user_002 (latest 2):")
    for entry in user2_history:
        print(f"- Query: '{entry['query']}' at {entry['timestamp']}")

    # Get history for a non-existent user
    non_existent_history = get_query_history('non_existent_user')
    print(f"\nHistory for non_existent_user: {non_existent_history}")
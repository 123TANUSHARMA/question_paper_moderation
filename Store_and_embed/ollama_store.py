import json
import sqlite3
from pathlib import Path
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Use latest available Ollama embeddings
try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:
    from langchain_community.embeddings import OllamaEmbeddings

# -------- CONFIG --------
QUESTIONS_FILE = "/Users/tanusharma/Downloads/coe-project/output_of_stages/generated_questions.json"
STORAGE_DIR = Path("question_storage")
DB_PATH = STORAGE_DIR / "questions.db"
FAISS_PATH = STORAGE_DIR / "faiss_index"
# ------------------------

def load_json():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def create_database():
    STORAGE_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            marks REAL,
            question_type TEXT,
            difficulty_level TEXT,
            cognitive_level TEXT,
            topic TEXT,
            subtopic TEXT,
            source_file TEXT,
            added_date TEXT,
            unique_hash TEXT UNIQUE,
            estimation_time REAL,
            category TEXT
        )
    ''')
    conn.commit()
    conn.close()

def store_data():
    data = load_json()
    create_database()
    conn = sqlite3.connect(DB_PATH, timeout=10)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")

    added_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    source_name = "Generated_Questions"
    db_data = []
    faiss_docs = []

    for category, questions in data.items():
        for q in questions:
            question_text = q.get("question", "").strip()
            if not question_text:
                continue

            unique_hash = str(hash(question_text))

            # Extract all fields
            marks = q.get("predicted_marks")
            estimation_time = q.get("predicted_estimation_time")
            question_type = q.get("question_type", "unknown")
            difficulty = q.get("difficulty_level", "unknown")
            cognitive = q.get("cognitive_level", "unknown")
            topic = q.get("topic", "general")
            subtopic = q.get("subtopic", "misc")

            db_data.append((
                question_text, marks, question_type, difficulty, cognitive,
                topic, subtopic, source_name, added_date, unique_hash,
                estimation_time, category
            ))

            faiss_docs.append(Document(
                page_content=question_text,
                metadata={
                    "marks": marks,
                    "estimation_time": estimation_time,
                    "question_type": question_type,
                    "difficulty_level": difficulty,
                    "cognitive_level": cognitive,
                    "topic": topic,
                    "subtopic": subtopic,
                    "source_file": source_name,
                    "added_date": added_date,
                    "category": category
                }
            ))

    cursor.executemany('''
        INSERT OR IGNORE INTO questions (
            question, marks, question_type, difficulty_level, cognitive_level,
            topic, subtopic, source_file, added_date, unique_hash,
            estimation_time, category
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', db_data)

    conn.commit()
    conn.close()

    print(f"✅ Stored {len(db_data)} questions in SQLite database")

    try:
        print("🔄 Creating FAISS index...")
        embeddings = OllamaEmbeddings(model="llama3")
        vector_store = FAISS.from_documents(faiss_docs, embeddings)
        vector_store.save_local(str(FAISS_PATH))
        print(f"✅ FAISS index created at {FAISS_PATH}")
    except Exception as e:
        print(f"⚠️ FAISS creation failed: {e}")

    print("🎉 Done! Database and FAISS index are ready.")

if __name__ == "__main__":
    store_data()

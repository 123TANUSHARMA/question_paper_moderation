# JSON Diagnostic and Auto-Fix Script
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path

# Fix for OllamaEmbeddings import
try:
    from langchain_ollama import OllamaEmbeddings
    print("✅ Using updated langchain_ollama")
except ImportError:
    try:
        from langchain_community.embeddings import OllamaEmbeddings
        print("⚠️ Using deprecated langchain_community")
    except ImportError:
        print("❌ Please install: pip install langchain-ollama")
        exit(1)

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Your file path
QUESTIONS_FILE = "/Users/tanusharma/Downloads/coe-project/output_of_stages/generated_questions.json"

def diagnose_json_structure():
    """Diagnose the JSON structure to understand the format"""
    
    print("🔍 Diagnosing JSON Structure...")
    print("=" * 50)
    
    if not os.path.exists(QUESTIONS_FILE):
        print(f"❌ File not found: {QUESTIONS_FILE}")
        return None
    
    with open(QUESTIONS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📊 JSON Analysis:")
    print(f"   Type: {type(data)}")
    print(f"   Length/Keys: {len(data) if isinstance(data, (list, dict)) else 'N/A'}")
    
    if isinstance(data, list):
        print(f"   Structure: List with {len(data)} items")
        if len(data) > 0:
            print(f"   First item type: {type(data[0])}")
            if isinstance(data[0], dict):
                print(f"   First item keys: {list(data[0].keys())}")
                print(f"   Sample item: {data[0]}")
            else:
                print(f"   First item content: {data[0]}")
    
    elif isinstance(data, dict):
        print(f"   Structure: Dictionary")
        print(f"   Keys: {list(data.keys())}")
        # Check if it's a nested structure
        for key, value in data.items():
            print(f"   '{key}': {type(value)} ({len(value) if isinstance(value, (list, dict)) else 'scalar'})")
            if isinstance(value, list) and len(value) > 0:
                print(f"      First item in '{key}': {type(value[0])}")
                if isinstance(value[0], dict):
                    print(f"      Sample keys: {list(value[0].keys())}")
            break  # Just show first key for diagnosis
    
    else:
        print(f"   Unexpected structure: {type(data)}")
    
    return data

def extract_questions_from_data(data):
    """Extract questions from whatever structure we have"""
    
    questions = []
    
    if isinstance(data, list):
        # Direct list of questions
        for item in data:
            if isinstance(item, dict):
                questions.append(item)
            elif isinstance(item, str):
                # Just strings - create question objects
                questions.append({
                    'question': item,
                    'marks': None,
                    'question_type': 'unknown',
                    'difficulty_level': 'unknown',
                    'cognitive_level': 'unknown'
                })
    
    elif isinstance(data, dict):
        # Check common nested structures
        possible_keys = ['questions', 'data', 'items', 'content', 'results']
        
        for key in possible_keys:
            if key in data and isinstance(data[key], list):
                print(f"✅ Found questions in '{key}' field")
                return extract_questions_from_data(data[key])
        
        # Maybe each key is a question category
        for key, value in data.items():
            if isinstance(value, list):
                print(f"✅ Treating '{key}' as question category")
                category_questions = extract_questions_from_data(value)
                # Add category info
                for q in category_questions:
                    q['category'] = key
                questions.extend(category_questions)
            elif isinstance(value, str):
                # Single question per key
                questions.append({
                    'question': value,
                    'marks': None,
                    'question_type': key,
                    'difficulty_level': 'unknown',
                    'cognitive_level': 'unknown'
                })
    
    return questions

def smart_setup():
    """Smart setup that handles different JSON structures"""
    
    print("🧠 Smart Setup for Your Question Store")
    print("=" * 50)
    
    # Diagnose JSON structure
    data = diagnose_json_structure()
    if data is None:
        return
    
    # Extract questions
    questions = extract_questions_from_data(data)
    
    if not questions:
        print("❌ Could not extract questions from the JSON structure")
        print("Please share a sample of your JSON structure for manual fix")
        return
    
    print(f"✅ Extracted {len(questions)} questions")
    
    # Show sample
    print(f"\n📝 Sample Question:")
    sample = questions[0]
    for key, value in sample.items():
        print(f"   {key}: {value}")
    
    # Create storage directory
    storage_dir = Path("question_storage")
    storage_dir.mkdir(exist_ok=True)
    
    db_path = storage_dir / "questions.db"
    faiss_path = storage_dir / "faiss_index"
    
    print(f"\n📂 Storage: {storage_dir}")
    
    # Setup database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS questions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT NOT NULL,
        marks INTEGER,
        question_type TEXT,
        difficulty_level TEXT,
        cognitive_level TEXT,
        topic TEXT,
        subtopic TEXT,
        source_file TEXT,
        added_date TEXT,
        unique_hash TEXT UNIQUE,
        category TEXT
    )
''')
    conn.commit()
    conn.close()
    
    print(f"✅ Database created: {db_path}")
    
    # Process questions
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    source_name = "Initial_Questions"
    added_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    db_data = []
    vector_documents = []
    added_count = 0
    
    for question in questions:
        # Ensure we have a question text
        question_text = question.get('question', str(question))
        if not question_text or question_text.strip() == '':
            continue
        
        unique_hash = str(hash(question_text))
        
        # Check for duplicates
        cursor.execute('SELECT id FROM questions WHERE unique_hash = ?', (unique_hash,))
        if cursor.fetchone():
            continue
        
        db_entry = (
    question_text,
    question.get('marks'),
    question.get('question_type', 'unknown'),
    question.get('difficulty_level', 'unknown'),
    question.get('cognitive_level', 'unknown'),
    question.get('topic', 'general'),
    question.get('subtopic', 'misc'),
    source_name,
    added_date,
    unique_hash,
    question.get('category', 'general')
)

        db_data.append(db_entry)
        
        # Create document for FAISS
        doc = Document(
    page_content=question_text,
    metadata={
        'marks': question.get('marks'),
        'question_type': question.get('question_type', 'unknown'),
        'difficulty_level': question.get('difficulty_level', 'unknown'),
        'cognitive_level': question.get('cognitive_level', 'unknown'),
        'topic': question.get('topic', 'general'),
        'subtopic': question.get('subtopic', 'misc'),
        'source_file': source_name,
        'added_date': added_date,
        'category': question.get('category', 'general')
    }
)

        vector_documents.append(doc)
        added_count += 1
    
    # Insert into database
    if db_data:
        cursor.executemany('''
    INSERT INTO questions 
    (question, marks, question_type, difficulty_level, cognitive_level, 
     topic, subtopic, source_file, added_date, unique_hash, category)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
''', db_data)

        conn.commit()
    
    conn.close()
    
    print(f"✅ Added {added_count} questions to database")
    
    # Create FAISS index
    if vector_documents:
        try:
            print("🔄 Creating FAISS index... (this may take a moment)")
            embeddings = OllamaEmbeddings(model="llama3")
            vector_store = FAISS.from_documents(vector_documents, embeddings)
            vector_store.save_local(str(faiss_path))
            print(f"✅ Created FAISS index: {faiss_path}")
        except Exception as e:
            print(f"⚠️ FAISS creation failed: {e}")
            print("(Database still created successfully - you can create FAISS later)")
    
    # Show results
    print(f"\n🎉 Setup Complete!")
    print(f"   📊 Database: {db_path}")
    print(f"   🔍 FAISS Index: {faiss_path}")
    print(f"   📝 Questions Added: {added_count}")
    
    return storage_dir

def show_database_content():
    """Show what's actually in the database"""
    db_path = Path("question_storage") / "questions.db"
    
    if not db_path.exists():
        print("❌ Database not found")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all questions
    cursor.execute('SELECT question, question_type, category FROM questions')
    questions = cursor.fetchall()
    
    print(f"\n📋 Database Contents ({len(questions)} questions):")
    print("=" * 60)
    
    for i, (question, q_type, category) in enumerate(questions, 1):
        print(f"{i}. {question}")
        print(f"   Type: {q_type}, Category: {category}")
        print()
    
    conn.close()

def show_raw_json():
    """Show raw JSON content for debugging"""
    print(f"\n📄 Raw JSON Content (first 500 chars):")
    print("=" * 50)
    
    with open(QUESTIONS_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
        print(content[:500])
        if len(content) > 500:
            print("... (truncated)")

if __name__ == "__main__":
    print("🎯 Choose an option:")
    print("1. Diagnose JSON structure")
    print("2. Smart setup (auto-detect format)")
    print("3. Show raw JSON content")
    print("4. Show database content")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        diagnose_json_structure()
    
    elif choice == "2":
        storage_dir = smart_setup()
        if storage_dir:
            show_database_content()
    
    elif choice == "3":
        show_raw_json()
    
    elif choice == "4":
        show_database_content()
    
    else:
        print("❌ Invalid choice")

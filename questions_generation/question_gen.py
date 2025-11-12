import json
import random
import re
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from question_inference import get_predicted_values

# Load Ollama LLM with increased temperature for better generation
llm = OllamaLLM(model="gemma3:1b", temperature=0.7)
output_parser = StrOutputParser()

# SIMPLIFIED GENERATION PROMPT - More direct and easier for small model
question_prompt = PromptTemplate.from_template("""
Create a university exam question from this content:

{text}

Create a {marks}-mark question following these rules:

2-MARK QUESTION FORMAT:
- Ask to explain/compare 2 concepts
- Or trace an algorithm with small example
- Keep it 2-3 sentences max
Example: "Explain the difference between Stack and Queue data structures. Give one real-world application of each."

3-MARK QUESTION FORMAT:
- Ask to implement/design something with explanation
- Or solve a problem showing steps
- Keep it 3-4 sentences
Example: "Write a function to insert a node at the beginning of a singly linked list. Explain the steps involved."

5-MARK QUESTION FORMAT:
- Multi-part question with (1), (2), (3)
- Combine implementation + trace + analysis
- Keep it 4-5 sentences
Example: "Consider an array A = {{5, 2, 8, 1, 9}}. (1) Sort this array using bubble sort. (2) Show all passes and swaps. (3) What is the time complexity in worst case?"

Now create a {marks}-mark question. Write ONLY the question, nothing else.
""")

question_chain = question_prompt | llm | output_parser

# TOPIC DETECTION
topic_detection_prompt = PromptTemplate.from_template("""
What is the main topic in this text? Answer in 2-3 words only.

Text: {text}

Topic:""")

topic_detection_chain = topic_detection_prompt | llm | output_parser

subtopic_detection_prompt = PromptTemplate.from_template("""
What specific subtopic does this question cover? Answer in 2-3 words only.

Question: {question}
Main Topic: {topic}

Subtopic:""")

subtopic_detection_chain = subtopic_detection_prompt | llm | output_parser

MARKS_META = {
    2: {"question_type": "short", "difficulty_level": "medium", "time": "2-3 min", "cognitive_level": "applying"},
    3: {"question_type": "descriptive", "difficulty_level": "medium", "time": "4-5 min", "cognitive_level": "analyzing"},
    5: {"question_type": "long", "difficulty_level": "hard", "time": "8-12 min", "cognitive_level": "evaluating"}
}

splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150)

def extract_existing_questions(text):
    """Extract questions with improved regex patterns"""
    extracted = []
    
    print("\n🔍 Analyzing text for existing questions...")
    print(f"   Text sample (first 500 chars):\n{text[:500]}\n")
    
    # Multiple patterns to catch different formats
    patterns = [
        # Pattern 1: Q1. [CO1] Question text [4 Marks]
        r'(Q\d+\..*?\[\d+\s*Marks?\])',
        # Pattern 2: Q1. Question text [4 Marks] (without CO)
        r'(Q\d+\.\s*(?!\[CO).*?\[\d+\s*Marks?\])',
        # Pattern 3: 1. Question text [4 Marks]
        r'(\d+\.\s+(?!\[CO)[^0-9].*?\[\d+\s*Marks?\])',
    ]
    
    all_matches = []
    for i, pattern in enumerate(patterns, 1):
        matches = re.finditer(pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            all_matches.append(match.group(1))
            print(f"   ✓ Pattern {i} found: {match.group(1)[:80]}...")
    
    if not all_matches:
        print("   ❌ No questions found with regex patterns")
        print("   💡 The OCR text might not contain formatted questions")
        return []
    
    # Process matches
    for question_text in all_matches:
        # Clean OCR artifacts
        question_text = clean_ocr_artifacts(question_text)
        
        # Extract marks
        marks_match = re.search(r'\[(\d+)\s*Marks?\]', question_text)
        if marks_match:
            marks = int(marks_match.group(1))
            
            # Map 4 marks to 5
            if marks == 4:
                marks = 5
            
            if marks not in [2, 3, 5]:
                continue
            
            # Clean the question
            cleaned = clean_question(question_text)
            if cleaned and len(cleaned) > 30:
                extracted.append({
                    'question': cleaned,
                    'marks': marks,
                    'source': 'extracted'
                })
                print(f"   ✅ Extracted {marks}-mark question: {cleaned[:80]}...")
    
    return extracted

def clean_ocr_artifacts(text):
    """Fix common OCR errors"""
    replacements = {
        r'sclution': 'solution',
        r'anays': 'arrays',
        r'algorithim': 'algorithm',
        r'funciton': 'function',
        r'perfrom': 'perform',
        r'conider': 'consider',
        r'folowing': 'following',
        r'<->': '↔',
        r'\s+': ' ',
    }
    
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text

def clean_question(question):
    """Clean and validate question"""
    if not question:
        return None
    
    try:
        # Remove metadata
        metadata_patterns = [
            r'\[CO\d+\]',
            r'Q\d+\.',
            r'^\d+\.',
            r'\[\d+\s*Marks?\]',
        ]
        
        for pattern in metadata_patterns:
            question = re.sub(pattern, '', question, flags=re.IGNORECASE)
        
        # Remove markdown
        question = re.sub(r'\*\*(.*?)\*\*', r'\1', question)
        question = re.sub(r'\s+', ' ', question)
        question = question.strip()
        
        # Validation
        if len(question) < 30:
            return None
        
        # Check for exam metadata garbage
        garbage_keywords = ['enrollment', 'examination', 'course code', 'semester', 
                           'time allowed', 'instructions', 'attempt all']
        if any(kw in question.lower() for kw in garbage_keywords):
            return None
        
        # Reject MCQs
        if re.search(r'\b[a-d]\)', question):
            return None
        
        return question
        
    except Exception as e:
        return None

def detect_topic_with_llm(text_chunk):
    """Detect topic"""
    try:
        sample = text_chunk[:400]
        raw_topic = topic_detection_chain.invoke({"text": sample}).strip()
        
        # Clean up
        topic = re.sub(r'^(topic:?|answer:?)\s*', '', raw_topic, flags=re.IGNORECASE)
        topic = topic.strip().title()
        
        if len(topic) < 3 or len(topic) > 50:
            return "Data Structures"
        
        return topic
    except:
        return "Data Structures"

def detect_subtopic_with_llm(question, topic):
    """Detect subtopic"""
    try:
        raw_subtopic = subtopic_detection_chain.invoke({
            "question": question[:200],  # Truncate for speed
            "topic": topic
        }).strip()
        
        subtopic = re.sub(r'^(subtopic:?|answer:?)\s*', '', raw_subtopic, flags=re.IGNORECASE)
        subtopic = subtopic.strip().title()
        
        if len(subtopic) < 3 or len(subtopic) > 50:
            return "General"
        
        return subtopic
    except:
        return "General"

def generate_questions(text, questions_per_category=3):
    """Main generation function with diagnostics"""
    
    if not text or len(text.strip()) < 100:
        print("⚠️  Text too short")
        return create_empty_structure()
    
    print(f"📄 Processing {len(text)} characters...")
    
    # STEP 1: Try to extract existing questions
    print("\n" + "="*60)
    extracted_questions = extract_existing_questions(text)
    print(f"\n📊 Extraction Result: Found {len(extracted_questions)} questions")
    
    extracted_by_marks = {2: [], 3: [], 5: []}
    for eq in extracted_questions:
        marks = eq['marks']
        extracted_by_marks[marks].append(eq)
    
    # STEP 2: Generate questions
    print("\n" + "="*60)
    print("🔥 STEP 2: Generating questions...")
    
    chunks = splitter.split_text(text)
    print(f"   Created {len(chunks)} text chunks")
    
    if len(chunks) > 0:
        print(f"\n   Sample chunk 1 (first 200 chars):\n   {chunks[0][:200]}...\n")
    
    marks_buckets = {2: [], 3: [], 5: []}
    seen_questions = set()
    
    # Add extracted questions first
    for marks in [2, 3, 5]:
        for eq in extracted_by_marks[marks]:
            question = eq['question']
            if question not in seen_questions:
                seen_questions.add(question)
                
                sample_chunk = chunks[0] if chunks else text[:400]
                topic = detect_topic_with_llm(sample_chunk)
                subtopic = detect_subtopic_with_llm(question, topic)
                
                meta = MARKS_META[marks]
                question_json = {
                    "question": question,
                    "topic": topic,
                    "subtopic": subtopic,
                    "question_type": meta["question_type"],
                    "difficulty_level": meta["difficulty_level"],
                    "cognitive_level": meta["cognitive_level"],
                    "category": topic,
                    "image": None,
                    "source": "extracted"
                }
                
                # Try to get predictions
                try:
                    model_preds = get_predicted_values({
                        "question": question,
                        "question_type": question_json["question_type"],
                        "difficulty_level": question_json["difficulty_level"],
                        "cognitive_level": question_json["cognitive_level"],
                        "topic": question_json["topic"],
                        "subtopic": question_json["subtopic"],
                        "category": question_json["category"]
                    })
                    question_json["predicted_marks"] = model_preds["predicted_marks"]
                    question_json["predicted_estimation_time"] = model_preds["predicted_estimation_time"]
                except Exception as e:
                    print(f"      ⚠️  Prediction failed: {e}")
                    question_json["predicted_marks"] = marks
                    question_json["predicted_estimation_time"] = meta["time"]
                
                marks_buckets[marks].append(question_json)
    
    # Generate new questions
    for marks in [2, 3, 5]:
        current_count = len(marks_buckets[marks])
        needed = questions_per_category - current_count
        
        if needed <= 0:
            print(f"\n   ✅ {marks}-mark: Have {current_count} extracted questions")
            continue
        
        print(f"\n   🔄 Generating {needed} additional {marks}-mark questions...")
        
        generated_count = 0
        max_attempts = min(len(chunks) * 5, 50)
        
        for attempt in range(max_attempts):
            if generated_count >= needed:
                break
            
            # Pick a random chunk
            chunk = random.choice(chunks) if chunks else text[:600]
            
            if len(chunk.strip()) < 50:
                continue
            
            try:
                print(f"      Attempt {attempt + 1}/{max_attempts}...", end=" ")
                
                # Generate question
                raw_output = question_chain.invoke({
                    "text": chunk,
                    "marks": marks
                }).strip()
                
                # Debug: Show what LLM generated
                if attempt == 0:  # Show first attempt
                    print(f"\n      🔍 LLM Output (first 200 chars): {raw_output[:200]}")
                
                # Clean the output
                question = clean_question(raw_output)
                
                if not question:
                    print("❌ Invalid")
                    continue
                
                if question in seen_questions:
                    print("❌ Duplicate")
                    continue
                
                if len(question) > 600:  # Too long
                    print("❌ Too long")
                    continue
                
                # Valid question!
                seen_questions.add(question)
                
                # Detect topic/subtopic
                topic = detect_topic_with_llm(chunk)
                subtopic = detect_subtopic_with_llm(question, topic)
                
                meta = MARKS_META[marks]
                question_json = {
                    "question": question,
                    "topic": topic,
                    "subtopic": subtopic,
                    "question_type": meta["question_type"],
                    "difficulty_level": meta["difficulty_level"],
                    "cognitive_level": meta["cognitive_level"],
                    "category": topic,
                    "image": None,
                    "source": "generated"
                }
                
                # Try predictions
                try:
                    model_preds = get_predicted_values({
                        "question": question,
                        "question_type": question_json["question_type"],
                        "difficulty_level": question_json["difficulty_level"],
                        "cognitive_level": question_json["cognitive_level"],
                        "topic": question_json["topic"],
                        "subtopic": question_json["subtopic"],
                        "category": question_json["category"]
                    })
                    question_json["predicted_marks"] = model_preds["predicted_marks"]
                    question_json["predicted_estimation_time"] = model_preds["predicted_estimation_time"]
                except:
                    question_json["predicted_marks"] = marks
                    question_json["predicted_estimation_time"] = meta["time"]
                
                marks_buckets[marks].append(question_json)
                generated_count += 1
                
                print(f"✅ Success! ({generated_count}/{needed})")
                print(f"         Q: {question[:100]}...")
                
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                continue
    
    # Final summary
    print("\n" + "="*60)
    print("📊 FINAL SUMMARY:")
    total = 0
    for marks, questions in marks_buckets.items():
        extracted_count = sum(1 for q in questions if q.get('source') == 'extracted')
        generated_count = len(questions) - extracted_count
        total += len(questions)
        status = "✅" if len(questions) >= questions_per_category else "⚠️"
        print(f"   {status} {marks}-mark: {len(questions)}/{questions_per_category} "
              f"(Extracted: {extracted_count}, Generated: {generated_count})")
    
    print(f"\n🎯 Total: {total} questions")
    
    return {
        "2_mark": marks_buckets[2],
        "3_mark": marks_buckets[3],
        "5_mark": marks_buckets[5]
    }

def create_empty_structure():
    return {"2_mark": [], "3_mark": [], "5_mark": []}

def save_questions(output, filename):
    """Save questions"""
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        total = sum(len(questions) for questions in output.values())
        print(f"\n✅ Saved {total} questions to {filename}")
        return True
    except Exception as e:
        print(f"❌ Save error: {e}")
        return False

if __name__ == "__main__":
    text_file = "/Users/tanusharma/Downloads/coe-project/output_of_stages/extracted_output.txt"
    output_file = "/Users/tanusharma/Downloads/coe-project/output_of_stages/generated_questions.json"
    
    print("🚀 UNIVERSITY QUESTION GENERATION SYSTEM V2")
    print("=" * 60)
    
    try:
        with open(text_file, "r", encoding="utf-8") as f:
            text = f.read()
        
        if not text.strip():
            print("❌ Input file is empty")
            exit(1)
        
        print(f"✅ Loaded {len(text)} characters")
        
        # Show a sample of the input
        print("\n📄 Input text sample (first 300 chars):")
        print("-" * 60)
        print(text[:300])
        print("-" * 60)
        
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        exit(1)
    
    # Generate
    final_questions = generate_questions(text, questions_per_category=3)
    
    # Save
    if save_questions(final_questions, output_file):
        print("\n🎉 GENERATION COMPLETED!")
        print("=" * 60)
        
        for category, questions in final_questions.items():
            if questions:
                print(f"\n📋 {category}:")
                for i, q in enumerate(questions, 1):
                    source = "📄" if q.get('source') == 'extracted' else "🤖"
                    print(f"\n   {source} Q{i}: {q['question'][:150]}...")
    else:
        print("❌ Failed to save questions")
        exit(1)
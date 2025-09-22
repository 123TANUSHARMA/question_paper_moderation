# import json
# import random
# from transformers import pipeline
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_ollama import OllamaLLM
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser

# # Load question generator
# question_gen = pipeline("text2text-generation", model="iarfmoose/t5-base-question-generator")

# # Load subtopic LLM
# llm = OllamaLLM(model="llama3")
# output_parser = StrOutputParser()

# # Prompt to predict subtopic
# subtopic_prompt = PromptTemplate.from_template("""
# Given the subject: {topic}
# And the question: "{question}"

# What is the best subtopic this question belongs to?
# Respond with a short academic subtopic label like "Neural Networks", "Backpropagation", etc.
# """)

# subtopic_chain = subtopic_prompt | llm | output_parser

# # Configs
# MARKS_META = {
#     1: {"question_type": "mcq", "difficulty_level": "easy", "time": "1 min", "cognitive_level": "remembering"},
#     2: {"question_type": "short", "difficulty_level": "medium", "time": "2-3 min", "cognitive_level": "understanding"},
#     3: {"question_type": "descriptive", "difficulty_level": "medium", "time": "4-5 min", "cognitive_level": "applying"},
#     5: {"question_type": "long", "difficulty_level": "hard", "time": "6-10 min", "cognitive_level": "evaluating"}
# }

# splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# def is_valid_question(q):
#     q = q.lower().strip()
#     return len(q) > 10 and not any(bad in q for bad in ["true", "false", "not_entailment", "entailment"])

# def detect_topic(text, topic_keywords):
#     for topic, keywords in topic_keywords.items():
#         for kw in keywords:
#             if kw.lower() in text.lower():
#                 return topic
#     return "General"

# def generate_questions(text, topic_keywords, questions_per_category=5):
#     chunks = splitter.split_text(text)
#     seen_questions = set()
#     used_chunks = set()
#     marks_buckets = {1: [], 2: [], 3: [], 5: []}

#     for marks in [1, 2, 3, 5]:
#         while len(marks_buckets[marks]) < questions_per_category and len(used_chunks) < len(chunks):
#             chunk = random.choice(chunks)
#             if chunk in used_chunks:
#                 continue
#             used_chunks.add(chunk)

#             result = question_gen(f"generate question: {chunk}", max_length=160 if marks > 2 else 64, do_sample=False)
#             question = result[0]['generated_text'].strip()

#             if not is_valid_question(question) or question in seen_questions:
#                 continue
#             seen_questions.add(question)

#             topic = detect_topic(chunk, topic_keywords)
#             subtopic = subtopic_chain.invoke({"topic": topic, "question": question}).strip()

#             question_json = {
#                 "question": question,
#                 "topic": topic,
#                 "subtopic": subtopic,
#                 "question_type": MARKS_META[marks]["question_type"],
#                 "difficulty_level": MARKS_META[marks]["difficulty_level"],
#                 "time": MARKS_META[marks]["time"],
#                 "cognitive_level": MARKS_META[marks]["cognitive_level"],
#                 "marks": marks,
#                 "image": None
#             }

#             marks_buckets[marks].append(question_json)

#     return {
#         "1_mark": marks_buckets[1],
#         "2_mark": marks_buckets[2],
#         "3_mark": marks_buckets[3],
#         "5_mark": marks_buckets[5]
#     }

# def save_questions(output, filename="output_questions.json"):
#     with open(filename, "w", encoding="utf-8") as f:
#         json.dump(output, f, indent=2, ensure_ascii=False)
#     print(f"✅ Saved structured questions to {filename}")

# if __name__ == "__main__":
#     with open("C:\\Users\\hp\\projects\\Question_Generator\\Data_Preprocessing\\vector_stores\\extracted_output.txt", "r", encoding="utf-8") as f:
#         text = f.read()

#     with open("C:\\Users\\hp\\projects\\Question_Generator\\Data_Preprocessing\\vector_stores\\keyword.json", "r", encoding="utf-8") as kf:
#         topic_keywords = json.load(kf)

#     print("🚀 Generating questions with subtopics...")
#     final_questions = generate_questions(text, topic_keywords)
#     save_questions(final_questions)
#     print("✅ Question generation completed successfully!")

import json
import random
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load Ollama LLM
llm = OllamaLLM(model="gemma3:1b")
output_parser = StrOutputParser()

# Enhanced prompt for better question generation
question_prompt = PromptTemplate.from_template("""
You are an expert university professor creating examination questions.

Given this academic content:
---
{text}
---

Create ONE high-quality university-level question for {marks} marks.

QUESTION CHARACTERISTICS:
- Type: {question_type}
- Difficulty: {difficulty_level} 
- Cognitive Level: {cognitive_level}
- Should test deep understanding, not memorization

UNIVERSITY-LEVEL QUESTION PATTERNS:
✓ "A [scenario] requires [specific requirements]. (a) Suggest [solution] with justification. (b) Implement/Apply..."
✓ "Compare and contrast [A] vs [B] in terms of [criteria]. Justify which is better for [scenario]."
✓ "Analyze the time/space complexity of [algorithm/structure]. Derive the mathematical expression."
✓ "Design an algorithm/system for [problem]. Explain your approach and analyze its efficiency."
✓ "Critically evaluate [approach/method]. Discuss advantages, limitations, and suggest improvements."
✓ "Given [data/scenario], implement [solution] and trace through the execution with examples."

QUESTION REQUIREMENTS:
- Must include specific scenarios, examples, or applications
- Should have multiple parts (a), (b), (c) for higher marks
- Include justification/explanation requirements
- Must be answerable from the given content
- Avoid generic "What is..." or "Define..." questions

For MCQ (1 mark): Create scenario-based questions testing conceptual understanding
For Short (2 marks): Include justification or comparison elements  
For Descriptive (3+ marks): Multi-part questions with implementation/analysis

CRITICAL: Return ONLY the clean question text. No prefixes like "Here's the question:", no bold formatting, no extra explanations.
""")

question_chain = question_prompt | llm | output_parser

# Improved subtopic detection prompt
subtopic_prompt = PromptTemplate.from_template("""
Analyze this question and determine the most specific subtopic:

Question: "{question}"
Main Topic: {topic}

Based on the question content, what is the most relevant subtopic?

Examples of good subtopics:
- For Sorting: "Quick Sort", "Merge Sort", "Heap Sort", "Bubble Sort"
- For Trees: "Binary Search Tree", "AVL Tree", "B-Tree", "Tree Traversal"
- For Hash Tables: "Collision Resolution", "Hash Functions", "Open Addressing"
- For TCP: "Flow Control", "Congestion Control", "Connection Management"
- For Graphs: "Shortest Path", "Minimum Spanning Tree", "Graph Traversal"

Return only the subtopic name (2-4 words maximum). No explanations.
""")

subtopic_chain = subtopic_prompt | llm | output_parser

# Enhanced topic detection with specific academic topics
ACADEMIC_TOPIC_KEYWORDS = {
    "Sorting": [
        "bubble sort", "insertion sort", "selection sort", "merge sort", "quick sort",
        "heap sort", "radix sort", "counting sort", "bucket sort", "sorting algorithm",
        "stable sort", "in-place sort", "comparison sort", "sort", "sorting"
    ],
    "Searching": [
        "linear search", "binary search", "interpolation search", "exponential search",
        "jump search", "ternary search", "search algorithm", "searching technique",
        "search", "searching", "find"
    ],
    "Hash Tables": [
        "hash table", "hash map", "hash function", "collision", "chaining",
        "open addressing", "linear probing", "quadratic probing", "double hashing",
        "hashing", "hash"
    ],
    "Trees": [
        "binary tree", "binary search tree", "bst", "avl tree", "red black tree",
        "b-tree", "b+ tree", "tree traversal", "inorder", "preorder", "postorder",
        "tree rotation", "balanced tree", "tree insertion", "tree deletion", "tree"
    ],
    "Graphs": [
        "graph", "vertex", "edge", "directed graph", "undirected graph",
        "dfs", "bfs", "shortest path", "dijkstra", "bellman ford",
        "minimum spanning tree", "kruskal", "prim", "topological sort"
    ],
    "Dynamic Programming": [
        "dynamic programming", "dp", "memoization", "tabulation", "optimal substructure",
        "overlapping subproblems", "knapsack", "longest common subsequence", "fibonacci"
    ],
    "Greedy Algorithms": [
        "greedy algorithm", "greedy approach", "activity selection", "fractional knapsack",
        "huffman coding", "minimum coin change", "job scheduling", "greedy"
    ],
    "Arrays": [
        "array", "2d array", "matrix", "array operations", "array manipulation",
        "subarray", "sliding window", "two pointers", "array rotation"
    ],
    "Linked Lists": [
        "linked list", "singly linked list", "doubly linked list", "circular linked list",
        "linked list insertion", "linked list deletion", "reverse linked list"
    ],
    "Stacks": [
        "stack", "lifo", "push", "pop", "stack operations", "expression evaluation",
        "parentheses matching", "infix to postfix", "stack implementation"
    ],
    "Queues": [
        "queue", "fifo", "enqueue", "dequeue", "circular queue", "priority queue",
        "queue implementation", "breadth first search"
    ],
    "TCP": [
        "tcp", "transmission control protocol", "tcp connection", "tcp handshake",
        "tcp segments", "tcp flow control", "tcp congestion control", "reliable transport"
    ],
    "UDP": [
        "udp", "user datagram protocol", "udp packet", "unreliable transport",
        "connectionless protocol", "udp vs tcp"
    ],
    "HTTP": [
        "http", "hypertext transfer protocol", "http methods", "get", "post",
        "http status codes", "http headers", "rest api", "web protocols"
    ],
    "Database": [
        "database", "sql", "relational database", "normalization", "acid properties",
        "transactions", "joins", "primary key", "foreign key", "indexing"
    ],
    "Operating System": [
        "operating system", "process", "thread", "cpu scheduling", "memory management",
        "virtual memory", "paging", "deadlock", "synchronization", "file system"
    ],
    "Complexity Analysis": [
        "time complexity", "space complexity", "big o", "big theta", "big omega",
        "asymptotic analysis", "worst case", "best case", "average case", "complexity"
    ],
    "Recursion": [
        "recursion", "recursive", "base case", "recursive call", "recursive function",
        "tail recursion", "recursion tree", "recursive algorithm"
    ],
    "String Algorithms": [
        "string", "string matching", "pattern matching", "substring", "string search",
        "kmp algorithm", "rabin karp", "string processing"
    ]
}

# Configs with better cognitive alignment
MARKS_META = {
    1: {"question_type": "mcq", "difficulty_level": "easy", "time": "1 min", "cognitive_level": "understanding"},
    2: {"question_type": "short", "difficulty_level": "medium", "time": "2-3 min", "cognitive_level": "applying"},
    3: {"question_type": "descriptive", "difficulty_level": "medium", "time": "4-5 min", "cognitive_level": "analyzing"},
    5: {"question_type": "long", "difficulty_level": "hard", "time": "6-10 min", "cognitive_level": "evaluating"}
}

splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)

def detect_topic_enhanced(text, topic_keywords=None):
    """Enhanced topic detection with fallback to academic keywords"""
    text_lower = text.lower()
    
    # First try user-provided keywords
    if topic_keywords and isinstance(topic_keywords, dict):
        topic_scores = {}
        for topic, keywords in topic_keywords.items():
            if isinstance(keywords, list):
                score = sum(1 for kw in keywords if str(kw).lower() in text_lower)
                if score > 0:
                    topic_scores[topic] = score
        
        if topic_scores:
            return max(topic_scores, key=topic_scores.get)
    
    # Fallback to academic keywords
    topic_scores = {}
    for topic, keywords in ACADEMIC_TOPIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in text_lower)
        if score > 0:
            topic_scores[topic] = score
    
    return max(topic_scores, key=topic_scores.get) if topic_scores else "Algorithms"

def clean_question(question):
    """Clean and validate generated questions with comprehensive cleaning"""
    if not question:
        return None
    
    try:
        # Remove unwanted prefixes and formatting - comprehensive cleaning
        prefixes_to_remove = [
            r"Here's the question:\s*",
            r"Here is the question:\s*", 
            r"Question:\s*",
            r"\*\*Question:\*\*\s*",
            r"\*\*Here's the question:\*\*\s*",
            r"\*\*Mark \d+\*\*\s*",
            r"^\s*\*\*(.*?)\*\*\s*",  # Bold formatting at start
            r"^(Q\d+\.|Question \d+:|Answer:|Solution:)\s*",
            r"^\d+\.\s*",  # Question numbers
            r"^[a-z]\)\s*",  # a) b) c) numbering
        ]
        
        for pattern in prefixes_to_remove:
            question = re.sub(pattern, '', question, flags=re.IGNORECASE)
        
        # Clean quotes - handle both straight and smart quotes
        question = re.sub(r'^["\'""''`]+|["\'""''`]+$', '', question)
        
        # Replace newlines and multiple spaces
        question = re.sub(r'\n+', ' ', question)
        question = re.sub(r'\s+', ' ', question)
        question = question.strip()
        
        # Remove remaining markdown formatting
        question = re.sub(r'\*\*(.*?)\*\*', r'\1', question)  # Remove bold
        question = re.sub(r'\*(.*?)\*', r'\1', question)      # Remove italic
        
        # Final cleanup
        question = question.strip()
        
        # Validation checks
        if len(question) < 15:
            return None
            
        # Skip if contains common OCR artifacts
        ocr_artifacts = [
            'enrollment', 'student name', 'examination', 'course code', 'maximum marks',
            'jaypee institute', 'semester', 'b.tech', 'university', 'college'
        ]
        if any(artifact in question.lower() for artifact in ocr_artifacts):
            return None
        
        # Skip if question is too generic or incomplete
        if question.lower().startswith(('what is', 'define', 'list', 'name')) and len(question) < 50:
            return None
            
        return question
        
    except Exception as e:
        print(f"Error cleaning question: {e}")
        return None

def clean_subtopic(subtopic):
    """Clean subtopic with better validation"""
    if not subtopic:
        return "General"
    
    try:
        # Remove quotes and extra formatting
        subtopic = re.sub(r'^["\'""''`]+|["\'""''`]+$', '', subtopic)
        subtopic = re.sub(r'\*\*(.*?)\*\*', r'\1', subtopic)
        subtopic = subtopic.strip()
        
        # Validate length and content
        if not subtopic or len(subtopic) > 50 or len(subtopic) < 2:
            return "General"
            
        # Title case for better formatting
        subtopic = subtopic.title()
        
        return subtopic
        
    except Exception:
        return "General"

def generate_questions(text, topic_keywords=None, questions_per_category=3):
    """Enhanced question generation with better error handling"""
    
    # Validate input
    if not text or len(text.strip()) < 100:
        print("⚠️  Warning: Text too short for meaningful question generation")
        return create_empty_structure()
    
    print(f"📄 Processing text of {len(text)} characters...")
    
    # Split text into chunks
    chunks = splitter.split_text(text)
    if not chunks:
        print("⚠️  Warning: No text chunks created")
        return create_empty_structure()
    
    print(f"📝 Created {len(chunks)} text chunks for processing")
    
    seen_questions = set()
    used_chunks = set()
    marks_buckets = {1: [], 2: [], 3: [], 5: []}
    
    # Track generation attempts to avoid infinite loops
    max_attempts_per_category = min(len(chunks) * 3, 30)
    
    for marks, meta in MARKS_META.items():
        attempts = 0
        success_count = 0
        print(f"\n🔄 Generating {marks}-mark questions...")
        
        while success_count < questions_per_category and attempts < max_attempts_per_category:
            attempts += 1
            
            # Select chunk - prioritize unused chunks
            available_chunks = [c for c in chunks if c not in used_chunks and len(c.strip()) > 50]
            if not available_chunks:
                available_chunks = [c for c in chunks if len(c.strip()) > 50]
            
            if not available_chunks:
                print(f"⚠️  No suitable chunks available for {marks}-mark questions")
                break
                
            chunk = random.choice(available_chunks)
            used_chunks.add(chunk)
            
            try:
                # Generate question with timeout handling
                print(f"   Attempt {attempts}: Generating question...")
                
                raw_question = question_chain.invoke({
                    "text": chunk,
                    "marks": marks,
                    "question_type": meta["question_type"],
                    "difficulty_level": meta["difficulty_level"],
                    "cognitive_level": meta["cognitive_level"]
                }).strip()
                
                # Clean and validate question
                question = clean_question(raw_question)
                if not question:
                    print(f"   ❌ Question cleaning failed")
                    continue
                    
                if question in seen_questions:
                    print(f"   ❌ Duplicate question detected")
                    continue
                
                seen_questions.add(question)
                
                # Detect topic and subtopic
                topic = detect_topic_enhanced(chunk, topic_keywords)
                
                # Generate subtopic with error handling
                try:
                    raw_subtopic = subtopic_chain.invoke({
                        "topic": topic, 
                        "question": question
                    }).strip()
                    subtopic = clean_subtopic(raw_subtopic)
                except Exception as e:
                    print(f"   ⚠️  Subtopic generation failed: {e}")
                    subtopic = "General"
                
                # Create question JSON
                question_json = {
                    "question": question,
                    "topic": topic,
                    "subtopic": subtopic,
                    "question_type": meta["question_type"],
                    "difficulty_level": meta["difficulty_level"],
                    "time": meta["time"],
                    "cognitive_level": meta["cognitive_level"],
                    "marks": marks,
                    "image": None
                }
                
                marks_buckets[marks].append(question_json)
                success_count += 1
                
                print(f"   ✅ Generated {marks}-mark question ({success_count}/{questions_per_category})")
                print(f"      Topic: {topic} | Subtopic: {subtopic}")
                print(f"      Preview: {question[:80]}...")
                
            except Exception as e:
                print(f"   ❌ Error generating {marks}-mark question: {str(e)}")
                continue
    
    # Print final summary
    print("\n📊 GENERATION SUMMARY:")
    total_generated = 0
    for marks, questions in marks_buckets.items():
        count = len(questions)
        total_generated += count
        status = "✅" if count >= questions_per_category else "⚠️ "
        print(f"   {status} {marks}-mark: {count}/{questions_per_category} questions")
    
    print(f"\n🎯 Total questions generated: {total_generated}")
    
    return {
        "1_mark": marks_buckets[1],
        "2_mark": marks_buckets[2],
        "3_mark": marks_buckets[3],
        "5_mark": marks_buckets[5]
    }

def create_empty_structure():
    """Create empty question structure"""
    return {
        "1_mark": [],
        "2_mark": [],
        "3_mark": [],
        "5_mark": []
    }

def load_topic_keywords(filepath):
    """Safely load topic keywords with comprehensive error handling"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()
            
        if not content:
            print("⚠️  Keyword file is empty, using default academic topics")
            return None
            
        data = json.loads(content)
        
        if not data or not isinstance(data, dict):
            print("⚠️  Invalid keyword structure, using default academic topics")
            return None
        
        # Validate structure
        valid_data = {}
        for topic, keywords in data.items():
            if isinstance(keywords, list) and keywords:
                valid_data[topic] = keywords
        
        if not valid_data:
            print("⚠️  No valid topic-keyword pairs found, using defaults")
            return None
            
        print(f"✅ Loaded {len(valid_data)} topic categories from keyword file")
        return valid_data
        
    except FileNotFoundError:
        print("⚠️  Keyword file not found, using default academic topics")
        return None
    except json.JSONDecodeError as e:
        print(f"⚠️  Invalid JSON in keyword file: {e}")
        return None
    except Exception as e:
        print(f"⚠️  Error loading keyword file: {str(e)}")
        return None

def save_questions(output, filename):
    """Save questions with comprehensive error handling"""
    try:
        # Ensure directory exists
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        # Verify file was written correctly
        with open(filename, "r", encoding="utf-8") as f:
            verification_data = json.load(f)
            
        total_questions = sum(len(questions) for questions in verification_data.values())
        print(f"✅ Successfully saved {total_questions} questions to {filename}")
        return True
        
    except Exception as e:
        print(f"❌ Error saving questions: {str(e)}")
        return False

if __name__ == "__main__":
    # File paths
    text_file = "/Users/tanusharma/Downloads/coe-project/output_of_stages/extracted_output.txt"
    keyword_file = "/Users/tanusharma/Downloads/coe-project/output_of_stages/keyword.json"
    output_file = "/Users/tanusharma/Downloads/coe-project/output_of_stages/generated_questions.json"
    
    print("🚀 STARTING ENHANCED QUESTION GENERATION SYSTEM")
    print("=" * 60)
    
    # Load input text with validation
    try:
        with open(text_file, "r", encoding="utf-8") as f:
            text = f.read()
        
        if not text.strip():
            print("❌ Error: Input text file is empty")
            exit(1)
        
        print(f"✅ Loaded input text: {len(text)} characters")
            
    except FileNotFoundError:
        print(f"❌ Error: Text file not found: {text_file}")
        exit(1)
    except Exception as e:
        print(f"❌ Error reading text file: {str(e)}")
        exit(1)
    
    # Load topic keywords with fallback
    topic_keywords = load_topic_keywords(keyword_file)
    
    # Generate questions
    print(f"\n🔥 Starting question generation...")
    final_questions = generate_questions(text, topic_keywords, questions_per_category=3)
    
    # Save results with validation
    if save_questions(final_questions, output_file):
        print("\n🎉 QUESTION GENERATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        # Final summary
        for category, questions in final_questions.items():
            if questions:
                print(f"📋 {category}: {len(questions)} questions")
                # Show preview of first question
                preview = questions[0]['question'][:100] + "..." if len(questions[0]['question']) > 100 else questions[0]['question']
                print(f"   Preview: {preview}")
                print(f"   Topic: {questions[0]['topic']} | Subtopic: {questions[0]['subtopic']}")
                print()
    else:
        print("❌ Failed to save questions")
        exit(1)
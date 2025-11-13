import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json
import os

class DifficultyPredictor:
    """
    Predicts difficulty level of questions using trained BERT model
    """
    
    def __init__(self, model_path="models/difficulty_classifier"):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.config = None
        self.reverse_difficulty_map = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and tokenizer"""
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                "Please train the model first using train_difficulty_model.py"
            )
        
        print(f"📦 Loading difficulty prediction model from {self.model_path}...")
        
        try:
            # Load config
            config_path = os.path.join(self.model_path, 'model_config.json')
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            
            # Create reverse mapping
            self.reverse_difficulty_map = {
                v: k for k, v in self.config['difficulty_map'].items()
            }
            
            # Load tokenizer and model
            self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
            self.model = BertForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Model loaded successfully on {self.device}")
            
        except Exception as e:
            raise Exception(f"Error loading model: {e}")
    
    def predict(self, question, question_type=None, cognitive_level=None, 
                topic=None, marks=None):
        """
        Predict difficulty level for a given question
        
        Args:
            question (str): The question text
            question_type (str, optional): Type of question (short, descriptive, long)
            cognitive_level (str, optional): Cognitive level (applying, analyzing, evaluating)
            topic (str, optional): Topic of the question
            marks (int, optional): Marks assigned to the question
        
        Returns:
            dict: Contains predicted_difficulty, confidence, and probabilities
        """
        
        if not question or len(question.strip()) == 0:
            return {
                'predicted_difficulty': 'medium',
                'confidence': 0.0,
                'probabilities': {}
            }
        
        # Prepare combined text (same format as training)
        combined_text = question
        
        if question_type or cognitive_level or topic or marks:
            metadata_parts = []
            if question_type:
                metadata_parts.append(f"[TYPE: {question_type}]")
            if cognitive_level:
                metadata_parts.append(f"[COGNITIVE: {cognitive_level}]")
            if topic:
                metadata_parts.append(f"[TOPIC: {topic}]")
            if marks:
                metadata_parts.append(f"[MARKS: {marks}]")
            
            combined_text = f"{question} {' '.join(metadata_parts)}"
        
        # Tokenize
        encoding = self.tokenizer(
            combined_text,
            add_special_tokens=True,
            max_length=self.config['max_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Get probabilities
            probabilities = torch.softmax(logits, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()
        
        # Map to difficulty level
        predicted_difficulty = self.reverse_difficulty_map[predicted_class]
        
        # Create probability dict
        prob_dict = {
            self.reverse_difficulty_map[i]: probabilities[i].item()
            for i in range(len(self.reverse_difficulty_map))
        }
        
        return {
            'predicted_difficulty': predicted_difficulty,
            'confidence': confidence,
            'probabilities': prob_dict
        }
    
    def predict_batch(self, questions_data):
        """
        Predict difficulty for multiple questions
        
        Args:
            questions_data (list): List of dicts containing question info
                Each dict should have: question, question_type, cognitive_level, topic, marks
        
        Returns:
            list: List of prediction results
        """
        
        results = []
        
        for q_data in questions_data:
            prediction = self.predict(
                question=q_data.get('question', ''),
                question_type=q_data.get('question_type'),
                cognitive_level=q_data.get('cognitive_level'),
                topic=q_data.get('topic'),
                marks=q_data.get('marks')
            )
            results.append(prediction)
        
        return results

# Global instance for efficient reuse
_predictor_instance = None

def get_difficulty_predictor(model_path="models/difficulty_classifier"):
    """
    Get or create a global DifficultyPredictor instance
    This avoids reloading the model multiple times
    """
    global _predictor_instance
    
    if _predictor_instance is None:
        _predictor_instance = DifficultyPredictor(model_path)
    
    return _predictor_instance

def predict_difficulty(question, question_type=None, cognitive_level=None, 
                       topic=None, marks=None, model_path="models/difficulty_classifier"):
    """
    Convenience function to predict difficulty for a single question
    
    Args:
        question (str): The question text
        question_type (str, optional): Type of question
        cognitive_level (str, optional): Cognitive level
        topic (str, optional): Topic
        marks (int, optional): Marks
        model_path (str): Path to model directory
    
    Returns:
        dict: Prediction results
    """
    
    try:
        predictor = get_difficulty_predictor(model_path)
        return predictor.predict(question, question_type, cognitive_level, topic, marks)
    except Exception as e:
        print(f"⚠️  Difficulty prediction failed: {e}")
        # Fallback to default
        return {
            'predicted_difficulty': 'medium',
            'confidence': 0.0,
            'probabilities': {'easy': 0.33, 'medium': 0.34, 'hard': 0.33}
        }

# Test function
if __name__ == "__main__":
    print("🧪 Testing Difficulty Predictor")
    print("=" * 70)
    
    # Sample questions for testing
    test_questions = [
        {
            "question": "What is a stack data structure?",
            "question_type": "short",
            "cognitive_level": "applying",
            "topic": "Data Structures",
            "marks": 2
        },
        {
            "question": "Implement a binary search tree with insertion, deletion, and traversal operations. Analyze the time complexity of each operation.",
            "question_type": "long",
            "cognitive_level": "evaluating",
            "topic": "Data Structures",
            "marks": 5
        },
        {
            "question": "Explain the difference between BFS and DFS graph traversal algorithms.",
            "question_type": "descriptive",
            "cognitive_level": "analyzing",
            "topic": "Algorithms",
            "marks": 3
        }
    ]
    
    try:
        predictor = DifficultyPredictor()
        
        for i, q_data in enumerate(test_questions, 1):
            print(f"\n📝 Question {i}:")
            print(f"   Text: {q_data['question'][:80]}...")
            print(f"   Marks: {q_data['marks']}")
            
            result = predictor.predict(**q_data)
            
            print(f"\n   🎯 Prediction: {result['predicted_difficulty'].upper()}")
            print(f"   📊 Confidence: {result['confidence']:.2%}")
            print(f"   📈 Probabilities:")
            for difficulty, prob in sorted(result['probabilities'].items()):
                print(f"      {difficulty}: {prob:.2%}")
        
        print("\n✅ Testing completed!")
        
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("\nPlease run train_difficulty_model.py first to train the model.")
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
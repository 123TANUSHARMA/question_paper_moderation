import pandas as pd
import numpy as np
import joblib
import re

class QuestionInference:
    def __init__(self, model_path='models/'):
        """
        Load trained models for inference
        """
        print("Loading models...")
        self.time_model = joblib.load(f'{model_path}time_model.pkl')
        self.marks_model = joblib.load(f'{model_path}marks_model.pkl')
        self.scaler = joblib.load(f'{model_path}scaler.pkl')
        self.label_encoders = joblib.load(f'{model_path}label_encoders.pkl')
        self.feature_columns = joblib.load(f'{model_path}feature_columns.pkl')
        print("Models loaded successfully!")
    
    def extract_text_features(self, text):
        """
        Extract features from question text
        """
        if pd.isna(text) or text == "":
            return {
                'word_count': 0,
                'char_count': 0,
                'avg_word_length': 0,
                'sentence_count': 0,
                'has_numbers': 0,
                'has_equations': 0,
                'question_marks': 0
            }
        
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        avg_word_length = char_count / word_count if word_count > 0 else 0
        sentence_count = len(re.split(r'[.!?]+', text))
        has_numbers = 1 if re.search(r'\d', text) else 0
        has_equations = 1 if re.search(r'[+\-*/=∫∑√]', text) else 0
        question_marks = text.count('?')
        
        return {
            'word_count': word_count,
            'char_count': char_count,
            'avg_word_length': avg_word_length,
            'sentence_count': sentence_count,
            'has_numbers': has_numbers,
            'has_equations': has_equations,
            'question_marks': question_marks
        }
    
    def engineer_features(self, question_data):
        """
        Engineer features for prediction
        """
        if isinstance(question_data, dict):
            df = pd.DataFrame([question_data])
        else:
            df = question_data.copy()
        
        # Extract text features
        text_features = df['question'].apply(self.extract_text_features)
        text_features_df = pd.DataFrame(text_features.tolist())
        df = pd.concat([df.reset_index(drop=True), text_features_df], axis=1)
        
        # Encode difficulty level
        difficulty_mapping = {
            'easy': 1, 'medium': 2, 'hard': 3,
            'Easy': 1, 'Medium': 2, 'Hard': 3
        }
        df['difficulty_encoded'] = df['difficulty_level'].map(difficulty_mapping)
        df['difficulty_encoded'] = df['difficulty_encoded'].fillna(2)
        
        # Encode cognitive level
        cognitive_mapping = {
            'remember': 1, 'understand': 2, 'apply': 3,
            'analyze': 4, 'evaluate': 5, 'create': 6,
            'Remember': 1, 'Understand': 2, 'Apply': 3,
            'Analyze': 4, 'Evaluate': 5, 'Create': 6
        }
        df['cognitive_encoded'] = df['cognitive_level'].map(cognitive_mapping)
        df['cognitive_encoded'] = df['cognitive_encoded'].fillna(2)
        
        # Encode categorical columns
        categorical_cols = ['question_type', 'topic', 'subtopic', 'category']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('unknown')
                # Use stored label encoders
                le = self.label_encoders.get(col)
                if le:
                    # Handle unseen categories
                    df[f'{col}_encoded'] = df[col].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else 0
                    )
                else:
                    df[f'{col}_encoded'] = 0
        
        # Interaction features
        df['difficulty_cognitive'] = df['difficulty_encoded'] * df['cognitive_encoded']
        
        return df
    
    def predict(self, question_data):
        """
        Predict estimation_time and marks for new question(s)
        
        Args:
            question_data: dict or DataFrame with columns:
                - question (str): The question text
                - question_type (str): Type of question
                - difficulty_level (str): easy/medium/hard
                - cognitive_level (str): remember/understand/apply/analyze/evaluate/create
                - topic (str): Topic name
                - subtopic (str): Subtopic name
                - category (str): Category name
        
        Returns:
            dict or list of dicts with predictions
        """
        # Engineer features
        df = self.engineer_features(question_data)
        
        # Select features
        X = df[self.feature_columns].fillna(0)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        time_pred = self.time_model.predict(X_scaled)
        marks_pred = self.marks_model.predict(X_scaled)
        
        # Round predictions
        time_pred = np.round(time_pred, 1)
        marks_pred = np.round(marks_pred, 1)
        
        # Return results
        if len(time_pred) == 1:
            return {
                'estimation_time': float(time_pred[0]),
                'marks': float(marks_pred[0])
            }
        else:
            return [
                {'estimation_time': float(t), 'marks': float(m)}
                for t, m in zip(time_pred, marks_pred)
            ]

def predict_single_question(question, question_type, difficulty_level, 
                           cognitive_level, topic, subtopic, category):
    """
    Helper function to predict for a single question
    """
    inference = QuestionInference()
    
    question_data = {
        'question': question,
        'question_type': question_type,
        'difficulty_level': difficulty_level,
        'cognitive_level': cognitive_level,
        'topic': topic,
        'subtopic': subtopic,
        'category': category
    }
    
    prediction = inference.predict(question_data)
    
    print(f"\nQuestion: {question[:100]}...")
    print(f"Predicted Estimation Time: {prediction['estimation_time']} minutes")
    print(f"Predicted Marks: {prediction['marks']}")
    
    return prediction

def predict_from_csv(csv_path):
    """
    Predict for multiple questions from CSV file
    """
    inference = QuestionInference()
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Make predictions
    predictions = inference.predict(df)
    
    # Add predictions to dataframe
    df['predicted_estimation_time'] = [p['estimation_time'] for p in predictions]
    df['predicted_marks'] = [p['marks'] for p in predictions]
    
    # Save results
    output_path = csv_path.replace('.csv', '_with_predictions.csv')
    df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")
    
    return df


def get_predicted_values(question_data):
    """
    Simplified helper for integration with question_generation.py
    Takes a dict like:
    {
        'question': str,
        'question_type': str,
        'difficulty_level': str,
        'cognitive_level': str,
        'topic': str,
        'subtopic': str,
        'category': str
    }
    Returns:
        {
            'predicted_marks': float,
            'predicted_estimation_time': float
        }
    """
    try:
        inference = QuestionInference()
        preds = inference.predict(question_data)
        return {
            "predicted_marks": preds["marks"],
            "predicted_estimation_time": preds["estimation_time"]
        }
    except Exception as e:
        print(f"⚠️ Prediction error: {e}")
        return {
            "predicted_marks": None,
            "predicted_estimation_time": None
        }

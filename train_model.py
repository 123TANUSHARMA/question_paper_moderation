import pandas as pd
import numpy as np
import sqlite3
import re
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
import seaborn as sns

class QuestionPredictor:
    def __init__(self, db_path):
        """
        Initialize the predictor with database path
        """
        self.db_path = db_path
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.time_model = None
        self.marks_model = None
        self.feature_columns = []
        
    def load_data(self):
        """
        Load data from SQLite database
        """
        print("Loading data from database...")
        conn = sqlite3.connect(self.db_path)
        
        # Load questions table
        query = """
        SELECT 
            question,
            marks,
            question_type,
            difficulty_level,
            cognitive_level,
            topic,
            subtopic,
            category,
            estimation_time
        FROM questions
        WHERE estimation_time IS NOT NULL 
        AND marks IS NOT NULL
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        print(f"Loaded {len(df)} samples from database")
        print(f"\nData shape: {df.shape}")
        print(f"\nColumns: {df.columns.tolist()}")
        
        return df
    
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
        
        # Basic text statistics
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        avg_word_length = char_count / word_count if word_count > 0 else 0
        
        # Count sentences
        sentence_count = len(re.split(r'[.!?]+', text))
        
        # Check for numbers and mathematical symbols
        has_numbers = 1 if re.search(r'\d', text) else 0
        has_equations = 1 if re.search(r'[+\-*/=∫∑√]', text) else 0
        
        # Count question marks
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
    
    def engineer_features(self, df):
        """
        Create features from raw data
        """
        print("\nEngineering features...")
        
        # Create a copy
        df = df.copy()
        
        # Extract text features
        text_features = df['question'].apply(self.extract_text_features)
        text_features_df = pd.DataFrame(text_features.tolist())
        
        # Combine with original dataframe
        df = pd.concat([df, text_features_df], axis=1)
        
        # Encode difficulty level (ordinal)
        difficulty_mapping = {
            'easy': 1,
            'medium': 2,
            'hard': 3,
            'Easy': 1,
            'Medium': 2,
            'Hard': 3
        }
        df['difficulty_encoded'] = df['difficulty_level'].map(difficulty_mapping)
        df['difficulty_encoded'] = df['difficulty_encoded'].fillna(2)  # Default to medium
        
        # Encode cognitive level (ordinal - Bloom's taxonomy)
        cognitive_mapping = {
            'remember': 1,
            'understand': 2,
            'apply': 3,
            'analyze': 4,
            'evaluate': 5,
            'create': 6,
            'Remember': 1,
            'Understand': 2,
            'Apply': 3,
            'Analyze': 4,
            'Evaluate': 5,
            'Create': 6
        }
        df['cognitive_encoded'] = df['cognitive_level'].map(cognitive_mapping)
        df['cognitive_encoded'] = df['cognitive_encoded'].fillna(2)  # Default to understand
        
        # Label encode categorical columns
        categorical_cols = ['question_type', 'topic', 'subtopic', 'category']
        
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                # Handle missing values
                df[col] = df[col].fillna('unknown')
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        # Create interaction features
        df['difficulty_cognitive'] = df['difficulty_encoded'] * df['cognitive_encoded']
        
        # Feature columns for model
        self.feature_columns = [
            'word_count', 'char_count', 'avg_word_length', 'sentence_count',
            'has_numbers', 'has_equations', 'question_marks',
            'difficulty_encoded', 'cognitive_encoded',
            'question_type_encoded', 'topic_encoded', 'subtopic_encoded', 
            'category_encoded', 'difficulty_cognitive'
        ]
        
        print(f"Total features created: {len(self.feature_columns)}")
        print(f"Features: {self.feature_columns}")
        
        return df
    
    def prepare_data(self, df):
        """
        Prepare features and targets
        """
        print("\nPreparing training data...")
        
        # Features
        X = df[self.feature_columns].copy()
        
        # Handle any remaining NaN values
        X = X.fillna(0)
        
        # Targets
        y_time = df['estimation_time'].values
        y_marks = df['marks'].values
        
        print(f"Features shape: {X.shape}")
        print(f"Time target shape: {y_time.shape}")
        print(f"Marks target shape: {y_marks.shape}")
        
        # Check for NaN in targets
        print(f"\nNaN in time: {np.isnan(y_time).sum()}")
        print(f"NaN in marks: {np.isnan(y_marks).sum()}")
        
        return X, y_time, y_marks
    
    def train_models(self, X, y_time, y_marks):
        """
        Train separate models for time and marks prediction
        """
        print("\n" + "="*50)
        print("TRAINING MODELS")
        print("="*50)
        
        # Split data
        X_train, X_test, y_time_train, y_time_test, y_marks_train, y_marks_test = train_test_split(
            X, y_time, y_marks, test_size=0.2, random_state=42
        )
        
        print(f"\nTrain size: {len(X_train)}")
        print(f"Test size: {len(X_test)}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Time Estimation Model
        print("\n--- Training Time Estimation Model ---")
        self.time_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.time_model.fit(X_train_scaled, y_time_train)
        
        # Predictions
        y_time_pred = self.time_model.predict(X_test_scaled)
        
        # Evaluate
        time_mae = mean_absolute_error(y_time_test, y_time_pred)
        time_rmse = np.sqrt(mean_squared_error(y_time_test, y_time_pred))
        time_r2 = r2_score(y_time_test, y_time_pred)
        
        print(f"Time Model Performance:")
        print(f"  MAE: {time_mae:.2f} minutes")
        print(f"  RMSE: {time_rmse:.2f} minutes")
        print(f"  R² Score: {time_r2:.3f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.time_model, X_train_scaled, y_time_train, 
                                     cv=5, scoring='neg_mean_absolute_error')
        print(f"  Cross-val MAE: {-cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})")
        
        # Train Marks Prediction Model
        print("\n--- Training Marks Prediction Model ---")
        self.marks_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.marks_model.fit(X_train_scaled, y_marks_train)
        
        # Predictions
        y_marks_pred = self.marks_model.predict(X_test_scaled)
        
        # Evaluate
        marks_mae = mean_absolute_error(y_marks_test, y_marks_pred)
        marks_rmse = np.sqrt(mean_squared_error(y_marks_test, y_marks_pred))
        marks_r2 = r2_score(y_marks_test, y_marks_pred)
        
        print(f"Marks Model Performance:")
        print(f"  MAE: {marks_mae:.2f} marks")
        print(f"  RMSE: {marks_rmse:.2f} marks")
        print(f"  R² Score: {marks_r2:.3f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.marks_model, X_train_scaled, y_marks_train, 
                                     cv=5, scoring='neg_mean_absolute_error')
        print(f"  Cross-val MAE: {-cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})")
        
        # Feature importance
        self.plot_feature_importance(X.columns)
        
        return {
            'time': {'mae': time_mae, 'rmse': time_rmse, 'r2': time_r2},
            'marks': {'mae': marks_mae, 'rmse': marks_rmse, 'r2': marks_r2}
        }
    
    def plot_feature_importance(self, feature_names):
        """
        Plot feature importance for both models
        """
        print("\nPlotting feature importance...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Time model importance
        time_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.time_model.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        axes[0].barh(time_importance['feature'], time_importance['importance'])
        axes[0].set_xlabel('Importance')
        axes[0].set_title('Top 10 Features - Time Estimation')
        axes[0].invert_yaxis()
        
        # Marks model importance
        marks_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.marks_model.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        axes[1].barh(marks_importance['feature'], marks_importance['importance'])
        axes[1].set_xlabel('Importance')
        axes[1].set_title('Top 10 Features - Marks Prediction')
        axes[1].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("Feature importance plot saved as 'feature_importance.png'")
        plt.close()
    
    def save_models(self, path='models/'):
        """
        Save trained models and preprocessing objects
        """
        import os
        os.makedirs(path, exist_ok=True)
        
        print(f"\nSaving models to {path}...")
        
        # Save models
        joblib.dump(self.time_model, f'{path}time_model.pkl')
        joblib.dump(self.marks_model, f'{path}marks_model.pkl')
        
        # Save preprocessing objects
        joblib.dump(self.scaler, f'{path}scaler.pkl')
        joblib.dump(self.label_encoders, f'{path}label_encoders.pkl')
        joblib.dump(self.feature_columns, f'{path}feature_columns.pkl')
        
        print("Models saved successfully!")
        print(f"  - time_model.pkl")
        print(f"  - marks_model.pkl")
        print(f"  - scaler.pkl")
        print(f"  - label_encoders.pkl")
        print(f"  - feature_columns.pkl")
    
    def predict(self, question_data):
        """
        Predict time and marks for new questions
        
        question_data: dict or DataFrame with columns:
            question, question_type, difficulty_level, cognitive_level, 
            topic, subtopic, category
        """
        if isinstance(question_data, dict):
            question_data = pd.DataFrame([question_data])
        elif isinstance(question_data, pd.Series):
            question_data = pd.DataFrame([question_data])
        
        # Keep only required columns for feature engineering
        required_cols = ['question', 'question_type', 'difficulty_level', 
                        'cognitive_level', 'topic', 'subtopic', 'category']
        question_data = question_data[required_cols].copy()
        
        # Engineer features
        df = self.engineer_features(question_data)
        
        # Prepare features - ensure correct order
        X = df[self.feature_columns].fillna(0)
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Predict
        time_pred = self.time_model.predict(X_scaled)
        marks_pred = self.marks_model.predict(X_scaled)
        
        return {
            'estimation_time': float(time_pred[0]),
            'marks': float(marks_pred[0])
        }

def main():
    """
    Main training pipeline
    """
    # Path to your database
    DB_PATH = '/Users/tanusharma/Downloads/coe-project/question_storage/questions.db'  # Update this path
    
    # Initialize predictor
    predictor = QuestionPredictor(DB_PATH)
    
    # Load data
    df = predictor.load_data()
    
    # Check data statistics
    print("\n" + "="*50)
    print("DATA STATISTICS")
    print("="*50)
    print(f"\nEstimation Time - Mean: {df['estimation_time'].mean():.2f}, "
          f"Std: {df['estimation_time'].std():.2f}, "
          f"Min: {df['estimation_time'].min():.2f}, "
          f"Max: {df['estimation_time'].max():.2f}")
    print(f"Marks - Mean: {df['marks'].mean():.2f}, "
          f"Std: {df['marks'].std():.2f}, "
          f"Min: {df['marks'].min():.2f}, "
          f"Max: {df['marks'].max():.2f}")
    
    # Engineer features
    df = predictor.engineer_features(df)
    
    # Prepare data
    X, y_time, y_marks = predictor.prepare_data(df)
    
    # Train models
    metrics = predictor.train_models(X, y_time, y_marks)
    
    # Save models
    predictor.save_models()
    
    # Test prediction on sample
    print("\n" + "="*50)
    print("SAMPLE PREDICTION TEST")
    print("="*50)
    sample = df.iloc[0].to_dict()
    prediction = predictor.predict(sample)
    print(f"\nSample Question: {sample['question'][:100]}...")
    print(f"Predicted Time: {prediction['estimation_time']:.2f} minutes")
    print(f"Predicted Marks: {prediction['marks']:.2f}")
    print(f"Actual Time: {sample['estimation_time']}")
    print(f"Actual Marks: {sample['marks']}")
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*50)

if __name__ == "__main__":
    main()
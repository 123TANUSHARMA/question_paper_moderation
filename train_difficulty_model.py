import sqlite3
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import json
import os
from tqdm import tqdm

# Configuration
class Config:
    DB_PATH = "question_storage/questions.db"
    MODEL_OUTPUT_DIR = "models/difficulty_classifier"
    MAX_LENGTH = 256
    BATCH_SIZE = 16
    EPOCHS = 10
    LEARNING_RATE = 2e-5
    TRAIN_SPLIT = 0.8
    RANDOM_SEED = 42
    
    # Difficulty levels mapping
    DIFFICULTY_MAP = {
        'easy': 0,
        'medium': 1,
        'hard': 2
    }
    
    REVERSE_DIFFICULTY_MAP = {v: k for k, v in DIFFICULTY_MAP.items()}

# Set random seeds for reproducibility
torch.manual_seed(Config.RANDOM_SEED)
np.random.seed(Config.RANDOM_SEED)

class QuestionDataset(Dataset):
    """Custom Dataset for Question Difficulty Classification"""
    
    def __init__(self, questions, difficulties, tokenizer, max_length):
        self.questions = questions
        self.difficulties = difficulties
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = str(self.questions[idx])
        difficulty = self.difficulties[idx]
        
        # Tokenize the question
        encoding = self.tokenizer(
            question,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(difficulty, dtype=torch.long)
        }

def load_data_from_db(db_path):
    """Load questions and their difficulty levels from SQLite database"""
    
    print(f"📊 Loading data from {db_path}...")
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Query to get questions with all relevant features
        query = """
        SELECT 
            question,
            difficulty_level,
            question_type,
            cognitive_level,
            topic,
            subtopic,
            marks
        FROM questions
        WHERE difficulty_level IS NOT NULL
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        print(f"✅ Loaded {len(df)} questions from database")
        print(f"\n📈 Difficulty Distribution:")
        print(df['difficulty_level'].value_counts())
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading database: {e}")
        return None

def prepare_features(df):
    """Prepare combined features for training"""
    
    print("\n🔧 Preparing features...")
    
    # Combine question with metadata for richer context
    df['combined_text'] = df.apply(
        lambda row: f"{row['question']} [TYPE: {row['question_type']}] [COGNITIVE: {row['cognitive_level']}] [TOPIC: {row['topic']}] [MARKS: {row['marks']}]",
        axis=1
    )
    
    # Map difficulty levels to integers
    df['difficulty_encoded'] = df['difficulty_level'].str.lower().map(Config.DIFFICULTY_MAP)
    
    # Remove any rows with unmapped difficulties
    df = df.dropna(subset=['difficulty_encoded'])
    df['difficulty_encoded'] = df['difficulty_encoded'].astype(int)
    
    print(f"✅ Prepared {len(df)} samples")
    print(f"\n📊 Encoded Difficulty Distribution:")
    print(df['difficulty_encoded'].value_counts().sort_index())
    
    return df

def create_dataloaders(df, tokenizer):
    """Create train and validation dataloaders"""
    
    print("\n📦 Creating dataloaders...")
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['combined_text'].values,
        df['difficulty_encoded'].values,
        test_size=1-Config.TRAIN_SPLIT,
        random_state=Config.RANDOM_SEED,
        stratify=df['difficulty_encoded'].values
    )
    
    print(f"✅ Train samples: {len(train_texts)}")
    print(f"✅ Validation samples: {len(val_texts)}")
    
    # Create datasets
    train_dataset = QuestionDataset(train_texts, train_labels, tokenizer, Config.MAX_LENGTH)
    val_dataset = QuestionDataset(val_texts, val_labels, tokenizer, Config.MAX_LENGTH)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)
    
    return train_loader, val_loader

def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch"""
    
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        predictions = torch.argmax(logits, dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_samples += labels.size(0)
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct_predictions/total_samples:.4f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    
    return avg_loss, accuracy

def evaluate(model, dataloader, device):
    """Evaluate the model"""
    
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    
    return avg_loss, accuracy, all_predictions, all_labels

def train_model():
    """Main training function"""
    
    print("🚀 DIFFICULTY LEVEL PREDICTION MODEL TRAINING")
    print("=" * 70)
    
    # Check if database exists
    if not os.path.exists(Config.DB_PATH):
        print(f"❌ Database not found at {Config.DB_PATH}")
        print("Please ensure the database exists and contains questions.")
        return
    
    # Load data
    df = load_data_from_db(Config.DB_PATH)
    if df is None or len(df) == 0:
        print("❌ No data available for training")
        return
    
    # Check minimum samples
    if len(df) < 20:
        print(f"⚠️  Warning: Only {len(df)} samples available. Recommend at least 50+ for good training.")
    
    # Prepare features
    df = prepare_features(df)
    
    # Initialize tokenizer and model
    print("\n🤖 Initializing BERT model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=len(Config.DIFFICULTY_MAP)
    )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Using device: {device}")
    model.to(device)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(df, tokenizer)
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Training loop
    print(f"\n🏋️ Training for {Config.EPOCHS} epochs...")
    print("=" * 70)
    
    best_val_accuracy = 0
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(Config.EPOCHS):
        print(f"\n📍 Epoch {epoch + 1}/{Config.EPOCHS}")
        print("-" * 70)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        print(f"✅ Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f}")
        
        # Evaluate
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, device)
        print(f"✅ Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f}")
        
        # Store history
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            os.makedirs(Config.MODEL_OUTPUT_DIR, exist_ok=True)
            
            model.save_pretrained(Config.MODEL_OUTPUT_DIR)
            tokenizer.save_pretrained(Config.MODEL_OUTPUT_DIR)
            
            print(f"💾 Best model saved! (Val Accuracy: {val_acc:.4f})")
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("📊 FINAL EVALUATION")
    print("=" * 70)
    
    # Load best model
    model = BertForSequenceClassification.from_pretrained(Config.MODEL_OUTPUT_DIR)
    model.to(device)
    
    _, final_acc, final_preds, final_labels = evaluate(model, val_loader, device)
    
    print(f"\n✅ Best Validation Accuracy: {best_val_accuracy:.4f}")
    
    # Classification report
    print("\n📈 Classification Report:")
    print(classification_report(
        final_labels,
        final_preds,
        target_names=[Config.REVERSE_DIFFICULTY_MAP[i] for i in range(len(Config.DIFFICULTY_MAP))]
    ))
    
    # Confusion matrix
    print("\n📊 Confusion Matrix:")
    cm = confusion_matrix(final_labels, final_preds)
    print(cm)
    
    # Save training history
    history_path = os.path.join(Config.MODEL_OUTPUT_DIR, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # Save configuration
    config_path = os.path.join(Config.MODEL_OUTPUT_DIR, 'model_config.json')
    config_dict = {
        'difficulty_map': Config.DIFFICULTY_MAP,
        'max_length': Config.MAX_LENGTH,
        'num_classes': len(Config.DIFFICULTY_MAP)
    }
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"\n✅ Model saved to: {Config.MODEL_OUTPUT_DIR}")
    print(f"✅ Training history saved to: {history_path}")
    print(f"✅ Model config saved to: {config_path}")
    
    print("\n🎉 Training completed successfully!")

if __name__ == "__main__":
    train_model()
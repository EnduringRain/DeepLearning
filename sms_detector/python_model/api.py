from fastapi import FastAPI, HTTPException, BackgroundTasks
import tensorflow as tf
import numpy as np
import pickle
import re
import os
import json
import pandas as pd
from datetime import datetime
from pydantic import BaseModel
import uvicorn
from typing import List, Optional
import shutil
import importlib.util
from fastapi.middleware.cors import CORSMiddleware
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the train_lstm_model module
spec = importlib.util.spec_from_file_location("train_lstm_model", "train_lstm_model.py")
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Preprocessing function (must match train_lstm_model.py)
def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'\d{10,}', '[PHONE]', text)
        text = re.sub(r'http[s]?://\S+', '[URL]', text)
        text = re.sub(r'£|\$|€', '[CURRENCY]', text)
        text = re.sub(r'!', ' [EXCL] ', text)
        text = re.sub(r'\?', ' [QUES] ', text)
        text = re.sub(r'%', ' [PERCENT] ', text)
        text = re.sub(r'[^\w\s\[\]]', ' ', text)
        abbreviations = {
            'u': 'you', 'r': 'are', 'ur': 'your', '2': 'to', '4': 'for',
            'wont': 'will not', 'cant': 'cannot', 'dont': 'do not',
            'im': 'i am', 'ive': 'i have', 'didnt': 'did not'
        }
        for abbr, full in abbreviations.items():
            text = re.sub(r'\b' + abbr + r'\b', full, text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ''

# Focal Loss (must match train_lstm_model.py)
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -tf.keras.backend.mean(alpha * tf.keras.backend.pow(1. - pt_1, gamma) * tf.keras.backend.log(pt_1 + tf.keras.backend.epsilon())) - \
               tf.keras.backend.mean((1 - alpha) * tf.keras.backend.pow(pt_0, gamma) * tf.keras.backend.log(1. - pt_0 + tf.keras.backend.epsilon()))
    return focal_loss_fixed

# Global variables
model = None
tokenizer = None
word_index = {}
best_threshold = 0.5
max_len = 200
max_words = 15000
model_loading_status = "Not loaded"

# Feedback storage
feedback_dir = "feedback_data"
os.makedirs(feedback_dir, exist_ok=True)
feedback_file = os.path.join(feedback_dir, "user_feedback.json")

if not os.path.exists(feedback_file):
    with open(feedback_file, "w") as f:
        json.dump([], f)

retrain_status = {
    "is_running": False,
    "start_time": None,
    "end_time": None,
    "status": "idle",
    "message": "",
    "progress": 0
}

# Load model, tokenizer, and threshold
try:
    model_path = 'spam_detector_improved.h5'
    logger.info(f"Attempting to load model from {model_path}")
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(
                model_path,
                custom_objects={'focal_loss_fixed': focal_loss(gamma=2.0, alpha=0.75)}
            )
            logger.info(f"Model loaded successfully from {model_path}")
            logger.info(f"Model summary: {model.summary()}")
        except Exception as e:
            raise ValueError(f"Failed to load model: {str(e)}")
    else:
        raise FileNotFoundError(f"Model file {model_path} not found")
    
    tokenizer_path = 'tokenizer_improved.pkl'
    logger.info(f"Attempting to load tokenizer from {tokenizer_path}")
    if os.path.exists(tokenizer_path):
        try:
            with open(tokenizer_path, 'rb') as f:
                tokenizer = pickle.load(f)
                logger.info(f"Tokenizer type: {type(tokenizer)}")
                if hasattr(tokenizer, 'word_index'):
                    logger.info(f"Word index type: {type(tokenizer.word_index)}")
                    if isinstance(tokenizer.word_index, dict):
                        word_index = tokenizer.word_index
                        logger.info(f"Tokenizer loaded from {tokenizer_path}, word index size: {len(word_index)}")
                        logger.info(f"Sample word_index entries: {dict(list(word_index.items())[:5])}")
                        for key, value in list(word_index.items())[:5]:
                            if not isinstance(key, str) or not isinstance(value, (int, float)):
                                raise ValueError(f"Invalid word_index entry: key={key} (type={type(key)}), value={value} (type={type(value)})")
                    else:
                        raise ValueError(f"word_index is not a dictionary, found type: {type(tokenizer.word_index)}")
                else:
                    raise ValueError("Tokenizer object does not have a word_index attribute")
        except Exception as e:
            raise ValueError(f"Failed to load or parse tokenizer: {str(e)}")
    else:
        raise FileNotFoundError(f"Tokenizer file {tokenizer_path} not found")
    
    threshold_path = 'best_threshold.pkl'
    logger.info(f"Attempting to load threshold from {threshold_path}")
    if os.path.exists(threshold_path):
        try:
            with open(threshold_path, 'rb') as f:
                best_threshold = pickle.load(f)
            logger.info(f"Best threshold loaded: {best_threshold}")
        except Exception as e:
            logger.warning(f"Failed to load threshold: {str(e)}, using default: {best_threshold}")
    else:
        logger.warning(f"Threshold file {threshold_path} not found, using default: {best_threshold}")
        
    if model is not None and word_index:
        model_loading_status = "Loaded successfully"
        logger.info("All resources loaded successfully")
    else:
        model_loading_status = "Partial loading: " + (
            "Model loaded" if model is not None else "Model missing"
        ) + ", " + (
            f"{len(word_index)} words in vocabulary" if word_index else "Vocabulary missing"
        )
        logger.info(f"Loading status: {model_loading_status}")
except Exception as e:
    model_loading_status = f"Failed: {str(e)}"
    logger.error(f"Error loading model or tokenizer: {e}")
    raise

# Tokenization function
def tokenize_text(text, max_sequence_len=200):
    if not word_index:
        raise ValueError("Word index is empty or not loaded")
    
    if not isinstance(word_index, dict):
        raise ValueError(f"word_index is not a dictionary, found type: {type(word_index)}")
    
    processed_text = preprocess_text(text)
    if not processed_text:
        raise ValueError("Text is empty or invalid after preprocessing")
    
    words = processed_text.split()
    sequence = []
    for word in words:
        try:
            if word in word_index:
                idx = word_index[word]
                if not isinstance(idx, (int, float)):
                    raise ValueError(f"Index for word '{word}' is not an integer: {idx} (type={type(idx)})")
                sequence.append(idx)
            else:
                oov_idx = word_index.get('<OOV>', 1)
                if not isinstance(oov_idx, (int, float)):
                    raise ValueError(f"OOV index is not an integer: {oov_idx} (type={type(oov_idx)})")
                sequence.append(oov_idx)
        except Exception as e:
            raise ValueError(f"Error tokenizing word '{word}': {str(e)}")
    
    if len(sequence) > max_sequence_len:
        sequence = sequence[:max_sequence_len]
    else:
        sequence = sequence + [0] * (max_sequence_len - len(sequence))
    
    logger.debug(f"Tokenized sequence: {sequence[:10]}... (length={len(sequence)})")
    return np.array([sequence])

# Pydantic models
class PredictionRequest(BaseModel):
    text: str

class FeedbackItem(BaseModel):
    message: str
    actual_label: str
    predicted_label: str
    confidence: float
    timestamp: Optional[str] = None

class IncrementalLearnRequest(BaseModel):
    messages: List[FeedbackItem]
    epochs: int = 3
    batch_size: int = 32

# Save feedback to file
def save_feedback(feedback_item: FeedbackItem):
    try:
        if not feedback_item.timestamp:
            feedback_item.timestamp = datetime.now().isoformat()
            
        feedback_data = []
        if os.path.exists(feedback_file) and os.path.getsize(feedback_file) > 0:
            with open(feedback_file, "r") as f:
                feedback_data = json.load(f)
        
        feedback_data.append(feedback_item.dict())
        
        with open(feedback_file, "w") as f:
            json.dump(feedback_data, f, indent=2)
            
        return True
    except Exception as e:
        logger.error(f"Error saving feedback: {e}")
        return False

# Incremental learning task
async def incremental_learn_task(messages: List[FeedbackItem], epochs: int = 3, batch_size: int = 32):
    global model, tokenizer, word_index, best_threshold, retrain_status
    
    try:
        retrain_status["is_running"] = True
        retrain_status["start_time"] = datetime.now().isoformat()
        retrain_status["status"] = "running"
        retrain_status["message"] = "Bắt đầu quá trình incremental learning"
        retrain_status["progress"] = 5
        
        data = []
        for item in messages:
            data.append({
                "message": item.message,
                "label": 1 if item.actual_label.lower() == "spam" else 0,
                "processed_message": preprocess_text(item.message)
            })
        
        df = pd.DataFrame(data)
        
        retrain_status["progress"] = 20
        retrain_status["message"] = f"Đã chuẩn bị {len(df)} mẫu cho incremental learning"
        
        sequences = tokenizer.texts_to_sequences(df["processed_message"])
        X = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len, padding="post")
        y = df["label"].values
        
        retrain_status["progress"] = 40
        retrain_status["message"] = "Đang thực hiện incremental learning..."
        
        optimizer = tf.keras.optimizers.AdamW(learning_rate=0.0005, weight_decay=0.0001)
        model.compile(
            optimizer=optimizer,
            loss=focal_loss(gamma=2.0, alpha=0.75),
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        class_weights = {}
        if len(np.unique(y)) > 1:
            from sklearn.utils.class_weight import compute_class_weight
            weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
            class_weights = {i: w for i, w in enumerate(weights)}
        
        history = model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights if class_weights else None,
            verbose=1
        )
        
        retrain_status["progress"] = 80
        retrain_status["message"] = "Đã hoàn thành incremental learning, đang lưu mô hình..."
        
        model.save("spam_detector_improved.h5")
        
        for item in messages:
            save_feedback(item)
        
        retrain_status["progress"] = 100
        retrain_status["status"] = "completed"
        retrain_status["end_time"] = datetime.now().isoformat()
        retrain_status["message"] = "Incremental learning thành công!"
        
        return True
    except Exception as e:
        retrain_status["status"] = "failed"
        retrain_status["end_time"] = datetime.now().isoformat()
        retrain_status["message"] = f"Lỗi khi thực hiện incremental learning: {str(e)}"
        logger.error(f"Error during incremental learning: {e}")
        return False
    finally:
        retrain_status["is_running"] = False

# Updated /predict endpoint for LSTM model
@app.post("/predict")
async def predict_spam(request: PredictionRequest):
    try:
        if model is None or tokenizer is None:
            raise ValueError("Model or tokenizer not loaded")
        
        text = request.text
        logger.info(f"Received text for prediction: {text}")
        
        # Preprocess and tokenize the text
        processed_text = preprocess_text(text)
        if not processed_text:
            raise ValueError("Text is empty or invalid after preprocessing")
        
        sequences = tokenizer.texts_to_sequences([processed_text])
        padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(
            sequences, maxlen=max_len, padding="post"
        )
        
        logger.debug(f"Padded sequence: {padded_sequence[0][:10]}... (length={len(padded_sequence[0])})")
        
        # Make prediction
        confidence = float(model.predict(padded_sequence, verbose=0)[0][0])
        label = "spam" if confidence > best_threshold else "ham"
        
        logger.info(f"Prediction: {label}, Confidence: {confidence}, Threshold: {best_threshold}")
        
        return {
            "prediction": label,
            "confidence": confidence,
            "threshold": best_threshold,
            "processed_text": processed_text
        }
    except Exception as e:
        logger.error(f"Error in predict_spam: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackItem):
    try:
        success = save_feedback(feedback)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save feedback")
        
        return {
            "status": "success",
            "message": "Feedback received and saved",
            "feedback_count": len(json.load(open(feedback_file, "r"))) if os.path.exists(feedback_file) else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing feedback: {str(e)}")

@app.get("/feedback_stats")
async def get_feedback_stats():
    try:
        if not os.path.exists(feedback_file) or os.path.getsize(feedback_file) == 0:
            return {
                "total_feedback": 0,
                "spam_count": 0,
                "ham_count": 0,
                "correct_predictions": 0,
                "incorrect_predictions": 0,
                "accuracy": 0
            }
        
        with open(feedback_file, "r") as f:
            feedback_data = json.load(f)
        
        total = len(feedback_data)
        spam_count = sum(1 for item in feedback_data if item["actual_label"].lower() == "spam")
        ham_count = total - spam_count
        
        correct_predictions = sum(1 for item in feedback_data 
                                if item["actual_label"].lower() == item["predicted_label"].lower())
        
        return {
            "total_feedback": total,
            "spam_count": spam_count,
            "ham_count": ham_count,
            "correct_predictions": correct_predictions,
            "incorrect_predictions": total - correct_predictions,
            "accuracy": correct_predictions / total if total > 0 else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting feedback stats: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "word_index_size": len(word_index),
        "model_loading_status": model_loading_status,
        "threshold": best_threshold,
        "feedback_count": len(json.load(open(feedback_file, "r"))) if os.path.exists(feedback_file) and os.path.getsize(feedback_file) > 0 else 0,
        "retrain_status": retrain_status["status"]
    }
# Pydantic model cho batch predict
class BatchPredictionRequest(BaseModel):
    messages: List[str]

@app.post("/batch_predict")
async def batch_predict(request: BatchPredictionRequest):
    try:
        if model is None or tokenizer is None:
            raise ValueError("Model or tokenizer not loaded")

        messages = request.messages
        if not messages:
            raise ValueError("No messages provided for prediction")

        logger.info(f"Received {len(messages)} messages for batch prediction")

        # Preprocess và tokenize tất cả tin nhắn
        processed_texts = [preprocess_text(text) for text in messages]
        sequences = tokenizer.texts_to_sequences(processed_texts)
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
            sequences, maxlen=max_len, padding="post"
        )

        logger.debug(f"Padded sequences shape: {padded_sequences.shape}")

        # Dự đoán hàng loạt
        confidences = model.predict(padded_sequences, verbose=0)
        predictions = [
            {
                "prediction": "spam" if conf > best_threshold else "ham",
                "confidence": float(conf),
                "threshold": best_threshold,
                "processed_text": processed_text
            }
            for conf, processed_text in zip(confidences, processed_texts)
        ]

        logger.info(f"Batch prediction completed: {len(predictions)} results")
        return predictions

    except Exception as e:
        logger.error(f"Error in batch_predict: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
    except Exception as e:
        logger.error(f"Failed to start Uvicorn server: {str(e)}")
        raise
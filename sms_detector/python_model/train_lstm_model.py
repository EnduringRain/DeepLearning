import pandas as pd
import numpy as np
import re
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, Dropout, Bidirectional,
    MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import random

# Hàm chuẩn hóa văn bản
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

# Tạo hàm augment dữ liệu đơn giản
def simple_augment(text, num_augmentations=1):
    augmented_texts = []
    
    spam_words = ['free', 'win', 'winner', 'cash', 'prize', 'call', 'text', 
                  'claim', 'offer', 'deal', 'limited', 'congrat', 'urgent',
                  'discount', 'guaranteed', 'click', 'link', 'reply', 'bonus']
    
    for _ in range(num_augmentations):
        words = text.split()
        
        if len(words) >= 4 and random.random() > 0.5:
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        
        if random.random() > 0.5:
            random_spam_word = random.choice(spam_words)
            insert_pos = random.randint(0, len(words))
            words.insert(insert_pos, random_spam_word)
        
        if len(words) > 3 and random.random() > 0.7:
            del_idx = random.randint(0, len(words) - 1)
            words.pop(del_idx)
            
        augmented_text = ' '.join(words)
        augmented_texts.append(augmented_text)
    
    return augmented_texts

# Focal Loss để xử lý mất cân bằng lớp
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1 + K.epsilon())) - \
               K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
    return focal_loss_fixed

# Callback tính F1-score
class F1ScoreCallback(Callback):
    def __init__(self, validation_data):
        super(F1ScoreCallback, self).__init__()
        self.validation_data = validation_data
        self.best_f1 = 0
        self.best_threshold = 0.5
        self.best_weights = None
        
    def on_epoch_end(self, epoch, logs=None):
        x_val, y_val = self.validation_data
        y_pred_proba = self.model.predict(x_val)
        
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_f1 = 0
        best_thresh = 0.5
        
        for thresh in thresholds:
            y_pred = (y_pred_proba > thresh).astype("int32")
            f1 = f1_score(y_val, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        print(f" - val_f1: {best_f1:.4f} (threshold: {best_thresh:.2f})")
        logs["val_f1"] = best_f1
        
        if best_f1 > self.best_f1:
            self.best_f1 = best_f1
            self.best_threshold = best_thresh
            self.best_weights = self.model.get_weights()
            print(f" - Saved best model with F1: {best_f1:.4f}")

# Hàm chính
def train_spam_detector(include_feedback=False, feedback_file=None, epochs=20, batch_size=64):
    # Đọc dữ liệu gốc
    df = pd.read_csv('data/SMSSpamCollection', sep='\t', header=None, encoding='latin-1')
    df.columns = ['label', 'message']
    df = df.dropna()
    
    # Mã hóa nhãn (spam=1, ham=0)
    df['label'] = df['label'].map({'spam': 1, 'ham': 0})
    
    print(f"Phân phối ban đầu: {df['label'].value_counts().to_dict()}")
    print(f"Tỉ lệ Spam/Ham: 1:{df['label'].value_counts()[0]/df['label'].value_counts()[1]:.1f}")
    
    # Tiền xử lý văn bản
    df['processed_message'] = df['message'].apply(preprocess_text)

    # Nếu sử dụng feedback, thêm dữ liệu feedback
    if include_feedback and feedback_file and os.path.exists(feedback_file):
        feedback_df = pd.read_csv(feedback_file)
        feedback_df = feedback_df[['processed_message', 'label']]
        # Kết hợp dữ liệu gốc và feedback
        df = pd.concat([df[['processed_message', 'label']], feedback_df], ignore_index=True)
        print(f"Đã thêm dữ liệu feedback: {len(feedback_df)} mẫu")
        print(f"Phân phối sau khi thêm feedback: {df['label'].value_counts().to_dict()}")
    
    # Tăng cường dữ liệu cho lớp thiểu số (spam)
    spam_df = df[df['label'] == 1]
    ham_df = df[df['label'] == 0]
    
    augmented_texts = []
    augmented_labels = []
    
    num_augmentations = min(5, int(len(ham_df) / len(spam_df)))
    
    for _, row in spam_df.iterrows():
        text = row['processed_message']
        aug_texts = simple_augment(text, num_augmentations)
        augmented_texts.extend(aug_texts)
        augmented_labels.extend([1] * len(aug_texts))
    
    aug_df = pd.DataFrame({
        'processed_message': augmented_texts,
        'label': augmented_labels
    })
    
    combined_df = pd.concat([df[['processed_message', 'label']], aug_df], ignore_index=True)
    
    print(f"Phân phối sau augmentation: {combined_df['label'].value_counts().to_dict()}")
    
    # Chuẩn bị dữ liệu và vectorization
    max_words = 15000
    max_len = 200
    
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(combined_df['processed_message'])
    sequences = tokenizer.texts_to_sequences(combined_df['processed_message'])
    word_index = tokenizer.word_index
    print(f"Kích thước từ điển: {len(word_index)} từ")
    
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
    
    X = padded_sequences
    y = combined_df['label'].values
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
    )
    
    print(f"Train set: {X_train.shape[0]} mẫu")
    print(f"Validation set: {X_val.shape[0]} mẫu")
    print(f"Test set: {X_test.shape[0]} mẫu")
    
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"Phân phối sau SMOTE: {np.bincount(y_train_smote)}")
    
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    print(f"Class weights: {class_weight_dict}")
    
    # Xây dựng mô hình
    embedding_dim = 200
    max_words = 15000
    max_len = 200

    inputs = tf.keras.Input(shape=(max_len,))
    embedding = Embedding(max_words, embedding_dim)(inputs)
    lstm_out = Bidirectional(LSTM(128, return_sequences=True))(embedding)
    lstm_dropout = Dropout(0.3)(lstm_out)
    attention_out = MultiHeadAttention(num_heads=8, key_dim=32)(query=lstm_dropout, value=lstm_dropout, key=lstm_dropout)
    attention_norm = LayerNormalization()(attention_out)
    pooled = GlobalAveragePooling1D()(attention_norm)

    dense1 = Dense(128, activation='relu')(pooled)
    dropout1 = Dropout(0.5)(dense1)
    dense2 = Dense(64, activation='relu')(dropout1)
    dropout2 = Dropout(0.4)(dense2)
    outputs = Dense(1, activation='sigmoid')(dropout2)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    optimizer = AdamW(learning_rate=0.001, weight_decay=0.0001)
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
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=0.00001,
        verbose=1
    )
    
    f1_callback = F1ScoreCallback(validation_data=(X_val, y_val))
    
    # Huấn luyện với epochs và batch_size được truyền vào
    history = model.fit(
        X_train_smote, y_train_smote,
        epochs=epochs,  # Sử dụng giá trị epochs truyền vào
        batch_size=batch_size,  # Sử dụng giá trị batch_size truyền vào
        validation_data=(X_val, y_val),
        class_weight=class_weight_dict,
        callbacks=[early_stopping, reduce_lr, f1_callback],
        verbose=2
    )
    
    if f1_callback.best_weights is not None:
        model.set_weights(f1_callback.best_weights)
        print(f"Đã áp dụng trọng số tốt nhất với F1: {f1_callback.best_f1:.4f}")
    
    print("\nĐánh giá mô hình trên tập test:")
    best_threshold = f1_callback.best_threshold
    print(f"Sử dụng ngưỡng tối ưu: {best_threshold:.2f}")
    
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > best_threshold).astype(int)
    
    report = classification_report(y_test, y_pred, target_names=['Ham', 'Spam'])
    print(report)
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'], 
                yticklabels=['Ham', 'Spam'])
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.title('Ma trận nhầm lẫn')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    model.save('spam_detector_improved.h5')
    
    with open('tokenizer_improved.pkl', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('best_threshold.pkl', 'wb') as handle:
        pickle.dump(best_threshold, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return model, tokenizer, best_threshold

if __name__ == "__main__":
    model, tokenizer, threshold = train_spam_detector()
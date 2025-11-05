"""
neural_network.py
Deep Learning baseline for Smart Detection of Phishing Attacks in Encrypted Network Traffic
Uses a simple fully connected neural network (DNN)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_clean_data():
    df = pd.read_csv("cleaned_data.csv")
    print(f"Loaded cleaned dataset — shape: {df.shape}")
    return df

def preprocess_data(df):
    drop_cols = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    le = LabelEncoder()
    df['Label'] = le.fit_transform(df['Label'])

    X = df.drop('Label', axis=1)
    y = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, le

def build_dnn(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate(model, X_train, X_test, y_train, y_test, le):
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, y_train, validation_split=0.2, epochs=25, batch_size=64,
        callbacks=[early_stop], verbose=1
    )

    y_pred = (model.predict(X_test) > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Test Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Neural Network - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('NN_confusion_matrix.png', dpi=300)
    plt.close()

    # Plot training history
    plt.figure(figsize=(6,4))
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('NN_training_curve.png', dpi=300)
    plt.close()
    print("📊 Saved training curve and confusion matrix plots")

def main():
    df = load_clean_data()
    X_train, X_test, y_train, y_test, le = preprocess_data(df)
    model = build_dnn(X_train.shape[1])
    train_and_evaluate(model, X_train, X_test, y_train, y_test, le)

if __name__ == "__main__":
    main()

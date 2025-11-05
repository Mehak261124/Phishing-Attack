"""
baseline_model.py
Baseline ML models for Smart Detection of Phishing Attacks in Encrypted Network Traffic
Goal: Train and evaluate traditional ML classifiers before deep models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_clean_data():
    df = pd.read_csv("cleaned_data.csv")
    print(f"Loaded cleaned dataset — shape: {df.shape}")
    return df

def preprocess_data(df):
    # Drop non-numeric IDs that don’t help prediction
    drop_cols = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # Encode labels
    le = LabelEncoder()
    df['Label'] = le.fit_transform(df['Label'])
    print(f"Classes encoded as: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    X = df.drop('Label', axis=1)
    y = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, le

def train_baseline_models(X_train, X_test, y_train, y_test, le):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}

    for name, model in models.items():
        print(f"\n=== Training {name} ===")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
        print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title(f'{name} - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(f'{name.replace(" ", "_")}_cm.png', dpi=300)
        plt.close()
        print(f"Saved confusion matrix as {name.replace(' ', '_')}_cm.png")

        results[name] = acc

    print("\n=== MODEL COMPARISON ===")
    for model, acc in results.items():
        print(f"{model}: {acc:.4f}")

def main():
    df = load_clean_data()
    X_train, X_test, y_train, y_test, le = preprocess_data(df)
    train_baseline_models(X_train, X_test, y_train, y_test, le)

if __name__ == "__main__":
    main()

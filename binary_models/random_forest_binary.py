"""
Random Forest Classifier for Binary Classification
Uses:  preprocessed_binary_10k/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc)
import time
import pickle
import warnings
warnings.filterwarnings("ignore")
import os


class RandomForestTrainer:
    def __init__(self, data_dir="preprocessed_binary_10k"):
        self.data_dir = data_dir
        self.model_name = "Random Forest"
        self.load_data()

    # ============================================================
    # 1. LOAD DATA
    # ============================================================
    def load_data(self):
        print("=" * 70)
        print(f"TRAINING {self.model_name.upper()} FOR BINARY CLASSIFICATION")
        print("=" * 70)

        print("\nLoading preprocessed data...")

        csv_dir = f"{self.data_dir}/csv"

        train_df = pd.read_csv(f"{csv_dir}/binary_train.csv")
        test_df = pd.read_csv(f"{csv_dir}/binary_test.csv")

        self.X_train = train_df.drop(columns=["Binary_Label"])
        self.y_train = train_df["Binary_Label"]

        self.X_test = test_df.drop(columns=["Binary_Label"])
        self.y_test = test_df["Binary_Label"]

        print(f"✓ Training set: {self.X_train.shape}")
        print(f"✓ Test set:     {self.X_test.shape}")
        print(f"✓ Features:     {self.X_train.shape[1]}")

    # ============================================================
    # 2. TRAIN MODEL
    # ============================================================
    def train(self):
        print("\n" + "=" * 70)
        print("TRAINING MODEL")
        print("=" * 70)

        self.model = RandomForestClassifier(
            n_estimators=120,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        print("\nTraining Random Forest (120 Trees)...")
        start = time.time()
        self.model.fit(self.X_train, self.y_train)
        self.train_time = time.time() - start

        print(f"✓ Training completed in {self.train_time:.2f}s")

        train_pred = self.model.predict(self.X_train)
        print(f"Train Accuracy: {accuracy_score(self.y_train, train_pred):.4f}")

    # ============================================================
    # 3. EVALUATE
    # ============================================================
    def evaluate(self):
        print("\n" + "=" * 70)
        print("EVALUATING ON TEST SET")
        print("=" * 70)

        start = time.time()
        self.y_pred = self.model.predict(self.X_test)
        self.test_time = time.time() - start

        self.y_proba = self.model.predict_proba(self.X_test)[:, 1]

        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        self.precision = precision_score(self.y_test, self.y_pred)
        self.recall = recall_score(self.y_test, self.y_pred)
        self.f1 = f1_score(self.y_test, self.y_pred)
        self.cm = confusion_matrix(self.y_test, self.y_pred)

        print("\nTest Set Performance:")
        print(f"  Accuracy : {self.accuracy:.4f}")
        print(f"  Precision: {self.precision:.4f}")
        print(f"  Recall   : {self.recall:.4f}")
        print(f"  F1-Score : {self.f1:.4f}")
        print(f"  Test Time: {self.test_time:.4f}s")

        print("\nConfusion Matrix:\n", self.cm)

    # ============================================================
    # 4. SAVE MODEL
    # ============================================================
    def save_model(self):
        model_dir = f"{self.data_dir}/models"
        os.makedirs(model_dir, exist_ok=True)

        file_path = f"{model_dir}/random_forest.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(self.model, f)

        print(f"\n✓ Model saved at: {file_path}")

    # ============================================================
    # 5. VISUALIZATIONS
    # ============================================================
    def plot_results(self):
        print("\n" + "=" * 70)
        print("GENERATING VISUALIZATIONS")
        print("=" * 70)

        plots_dir = f"{self.data_dir}/plots_model"
        os.makedirs(plots_dir, exist_ok=True)

        fig = plt.figure(figsize=(17, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

        # ---------------------------------------------------------
        # Confusion Matrix
        # ---------------------------------------------------------
        ax1 = fig.add_subplot(gs[0, 0])
        sns.heatmap(self.cm, annot=True, fmt='d', cmap='Greens',
                    xticklabels=['BENIGN', 'ATTACK'],
                    yticklabels=['BENIGN', 'ATTACK'],
                    ax=ax1, cbar=False)
        ax1.set_title(f"Confusion Matrix (Acc: {self.accuracy:.4f})")

        # ---------------------------------------------------------
        # Metrics Bar Plot
        # ---------------------------------------------------------
        ax2 = fig.add_subplot(gs[0, 1])
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
        scores = [self.accuracy, self.precision, self.recall, self.f1]
        ax2.bar(metrics, scores, color=["#2ecc71", "#27ae60", "#16a085", "#1abc9c"])
        ax2.set_title("Performance Metrics")
        ax2.set_ylim(0, 1.1)

        # ---------------------------------------------------------
        # ROC Curve
        # ---------------------------------------------------------
        ax3 = fig.add_subplot(gs[0, 2])
        fpr, tpr, _ = roc_curve(self.y_test, self.y_proba)
        roc_auc = auc(fpr, tpr)
        ax3.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}", color='green')
        ax3.plot([0, 1], [0, 1], 'k--')
        ax3.legend()
        ax3.set_title("ROC Curve")

        # ---------------------------------------------------------
        # Feature Importance
        # ---------------------------------------------------------
        ax4 = fig.add_subplot(gs[1, :])
        importance = pd.DataFrame({
            "feature": self.X_train.columns,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False).head(15)

        ax4.barh(importance["feature"], importance["importance"],
                 color=plt.cm.Greens(np.linspace(0.4, 0.9, 15)))
        ax4.set_title("Top 15 Most Important Features")
        ax4.invert_yaxis()

        # ---------------------------------------------------------
        # Summary Text
        # ---------------------------------------------------------
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis("off")

        summary = f"""
RANDOM FOREST CLASSIFICATION SUMMARY
===============================================
Accuracy : {self.accuracy:.4f}
Precision: {self.precision:.4f}
Recall   : {self.recall:.4f}
F1 Score : {self.f1:.4f}

Train Time: {self.train_time:.2f}s
Test Time : {self.test_time:.4f}s

Confusion Matrix:
TN: {self.cm[0,0]}
FP: {self.cm[0,1]}
FN: {self.cm[1,0]}
TP: {self.cm[1,1]}

Random Forest Advantages:
• Reduces overfitting compared to single decision tree
• High accuracy and stability
• Works well on large feature sets
"""
        ax5.text(0.03, 0.95, summary, fontsize=10, verticalalignment="top",
                 fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.3))

        out_path = f"{plots_dir}/random_forest_results.png"
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved visualization at: {out_path}")

    # ============================================================
    # RUN COMPLETE PIPELINE
    # ============================================================
    def run_complete_pipeline(self):
        self.train()
        self.evaluate()
        self.save_model()
        self.plot_results()

        print("\n" + "=" * 70)
        print("RANDOM FOREST TRAINING COMPLETE!")
        print("=" * 70)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    trainer = RandomForestTrainer(data_dir="preprocessed_binary_10k")
    trainer.run_complete_pipeline()
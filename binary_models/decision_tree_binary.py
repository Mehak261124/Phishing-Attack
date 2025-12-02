"""
Decision Tree Classifier for Binary Classification
Uses:  preprocessed_binary_10k/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc)
import time
import pickle
import warnings
warnings.filterwarnings("ignore")
import os


class DecisionTreeTrainer:
    def __init__(self, data_dir='preprocessed_binary_10k'):
        """Initialize Decision Tree Trainer"""
        self.data_dir = data_dir
        self.model_name = 'Decision Tree'
        self.load_data()

    # ============================================================
    # 1. LOAD DATA
    # ============================================================
    def load_data(self):
        print("=" * 70)
        print(f"TRAINING {self.model_name.upper()} FOR BINARY CLASSIFICATION")
        print("=" * 70)

        print("\nLoading preprocessed data...")

        # correct folder structure
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
    # 2. TRAIN
    # ============================================================
    def train(self):
        print("\n" + "=" * 70)
        print("TRAINING MODEL")
        print("=" * 70)

        self.model = DecisionTreeClassifier(
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )

        start = time.time()
        self.model.fit(self.X_train, self.y_train)
        self.train_time = time.time() - start

        print(f"\n✓ Training completed in {self.train_time:.2f}s")

        # Validation Metrics = Test metrics (no validation split)
        y_pred_val = self.model.predict(self.X_train)
        print(f"Train Accuracy: {accuracy_score(self.y_train, y_pred_val):.4f}")

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

        print(f"\nAccuracy : {self.accuracy:.4f}")
        print(f"Precision: {self.precision:.4f}")
        print(f"Recall   : {self.recall:.4f}")
        print(f"F1-Score : {self.f1:.4f}")
        print("\nConfusion Matrix:\n", self.cm)
        print("\nClassification Report:")
        print(classification_report(self.y_test, self.y_pred))

    # ============================================================
    # 4. SAVE MODEL
    # ============================================================
    def save_model(self):
        os.makedirs(f"{self.data_dir}/models", exist_ok=True)
        model_path = f"{self.data_dir}/models/decision_tree.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

        print(f"\n✓ Model saved at: {model_path}")

    # ============================================================
    # 5. VISUALIZATIONS (FULL PROFESSIONAL LAYOUT)
    # ============================================================
    def plot_results(self):
        print("\n" + "=" * 70)
        print("GENERATING VISUALIZATIONS")
        print("=" * 70)

        plots_dir = f"{self.data_dir}/plots_model"
        os.makedirs(plots_dir, exist_ok=True)

        fig = plt.figure(figsize=(17, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

        # ------------------------------------------------------------
        # 1. Confusion Matrix
        # ------------------------------------------------------------
        ax1 = fig.add_subplot(gs[0, 0])
        sns.heatmap(self.cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['BENIGN', 'ATTACK'],
                    yticklabels=['BENIGN', 'ATTACK'],
                    ax=ax1, cbar=False)
        ax1.set_title(f'Confusion Matrix\nAccuracy: {self.accuracy:.4f}', fontweight='bold')

        # ------------------------------------------------------------
        # 2. Metrics Bar Chart
        # ------------------------------------------------------------
        ax2 = fig.add_subplot(gs[0, 1])
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
        values = [self.accuracy, self.precision, self.recall, self.f1]
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
        bars = ax2.bar(metrics, values, color=colors, alpha=0.8)
        ax2.set_ylim(0, 1.1)
        ax2.set_title('Performance Metrics', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        for bar, val in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.03,
                     f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')

        # ------------------------------------------------------------
        # 3. ROC Curve
        # ------------------------------------------------------------
        ax3 = fig.add_subplot(gs[0, 2])
        fpr, tpr, _ = roc_curve(self.y_test, self.y_proba)
        roc_auc = auc(fpr, tpr)
        ax3.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'AUC = {roc_auc:.4f}')
        ax3.plot([0, 1], [0, 1], 'k--')
        ax3.set_title('ROC Curve', fontweight='bold')
        ax3.legend()

        # ------------------------------------------------------------
        # 4. Feature Importance (Top 15)
        # ------------------------------------------------------------
        ax4 = fig.add_subplot(gs[1, :])
        importance = pd.DataFrame({
            "feature": self.X_train.columns,
            "importance": self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(15)

        ax4.barh(importance['feature'], importance['importance'],
                 color=plt.cm.viridis(np.linspace(0.3, 0.9, 15)))
        ax4.set_title('Top 15 Important Features', fontweight='bold')
        ax4.invert_yaxis()
        ax4.grid(axis='x', alpha=0.3)

        # ------------------------------------------------------------
        # 5. Summary Text
        # ------------------------------------------------------------
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis("off")

        summary = f"""
MODEL SUMMARY - DECISION TREE
{'='*60}
Accuracy : {self.accuracy:.4f}
Precision: {self.precision:.4f}
Recall   : {self.recall:.4f}
F1-Score : {self.f1:.4f}
Train Time: {self.train_time:.3f}s
Test Time : {self.test_time:.4f}s

CONFUSION MATRIX DETAILS
TN (Benign correct):      {self.cm[0,0]}
FP (Benign→Attack):       {self.cm[0,1]}
FN (Attack→Benign):       {self.cm[1,0]}
TP (Attack correct):      {self.cm[1,1]}
"""
        ax5.text(0.02, 0.95, summary, fontsize=11,
                 verticalalignment="top",
                 fontfamily="monospace",
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

        output_path = f"{plots_dir}/decision_tree_results.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved visualization at: {output_path}")

    # ============================================================
    # 6. RUN PIPELINE
    # ============================================================
    def run_complete_pipeline(self):
        self.train()
        self.evaluate()
        self.save_model()
        self.plot_results()

        print("\n" + "=" * 70)
        print("DECISION TREE TRAINING COMPLETE!")
        print("=" * 70)
        print(f"Best F1 Score: {self.f1:.4f}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    trainer = DecisionTreeTrainer(data_dir="preprocessed_binary_10k")
    trainer.run_complete_pipeline()
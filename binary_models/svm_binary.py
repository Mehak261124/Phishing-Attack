"""
Support Vector Machine (SVM) Classifier for Binary Classification
Uses: preprocessed_binary_10k/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc,
                             classification_report)
import time
import pickle
import warnings
warnings.filterwarnings('ignore')
import os


class SVMTrainer:
    def __init__(self, data_dir="preprocessed_binary_10k"):
        """Initialize SVM Trainer"""
        self.data_dir = data_dir
        self.model_name = "Support Vector Machine (SVM)"
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

        self.model = SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=42,
            cache_size=1500
        )

        print("\nTraining SVM (RBF Kernel)... MAY TAKE TIME...")
        start = time.time()
        self.model.fit(self.X_train, self.y_train)
        self.train_time = time.time() - start

        print(f"✓ Training complete in {self.train_time:.2f}s")

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

        print("\nTest Performance:")
        print(f"  Accuracy : {self.accuracy:.4f}")
        print(f"  Precision: {self.precision:.4f}")
        print(f"  Recall   : {self.recall:.4f}")
        print(f"  F1-Score : {self.f1:.4f}")
        print(f"  Test Time: {self.test_time:.4f}s")
        print("\nConfusion Matrix:")
        print(self.cm)

    # ============================================================
    # 4. SAVE MODEL
    # ============================================================
    def save_model(self):
        model_dir = f"{self.data_dir}/models"
        os.makedirs(model_dir, exist_ok=True)

        file_path = f"{model_dir}/svm.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(self.model, f)

        print(f"\n✓ Model saved at: {file_path}")

    # ============================================================
    # 5. VISUALIZATION
    # ============================================================
    def plot_results(self):
        print("\n" + "=" * 70)
        print("GENERATING VISUALIZATIONS")
        print("=" * 70)

        plots_dir = f"{self.data_dir}/plots_model"
        os.makedirs(plots_dir, exist_ok=True)

        fig = plt.figure(figsize=(16, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.5, wspace=0.3)

        # --------------------------------------------------
        # Confusion Matrix
        # --------------------------------------------------
        ax1 = fig.add_subplot(gs[0, 0])
        sns.heatmap(self.cm, annot=True, fmt="d", cmap="Oranges",
                    xticklabels=["BENIGN", "ATTACK"],
                    yticklabels=["BENIGN", "ATTACK"], ax=ax1)
        ax1.set_title(f"Confusion Matrix\nAcc: {self.accuracy:.4f}")

        # --------------------------------------------------
        # Metrics
        # --------------------------------------------------
        ax2 = fig.add_subplot(gs[0, 1])
        metrics = ["Accuracy", "Precision", "Recall", "F1"]
        values = [self.accuracy, self.precision, self.recall, self.f1]
        ax2.bar(metrics, values, color=['#e67e22', '#d35400', '#ba4a00', '#a04000'])
        ax2.set_ylim(0, 1.1)
        ax2.set_title("Performance Metrics")

        # --------------------------------------------------
        # ROC Curve
        # --------------------------------------------------
        ax3 = fig.add_subplot(gs[0, 2])
        fpr, tpr, _ = roc_curve(self.y_test, self.y_proba)
        roc_auc = auc(fpr, tpr)
        ax3.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        ax3.plot([0, 1], [0, 1], "k--")
        ax3.legend()
        ax3.set_title("ROC Curve")

        # --------------------------------------------------
        # MODEL INFO: Support Vectors
        # --------------------------------------------------
        ax4 = fig.add_subplot(gs[1, :])
        ax4.axis("off")
        n_support = self.model.n_support_
        total_sv = sum(n_support)

        info = f"""
SUPPORT VECTOR MACHINE - MODEL DETAILS
===============================================
Kernel: {self.model.kernel.upper()}
Gamma : {self.model.gamma}
C     : {self.model.C}

Support Vectors:
  • BENIGN: {n_support[0]}
  • ATTACK: {n_support[1]}
  • TOTAL : {total_sv} ({total_sv / len(self.X_train) * 100:.2f}% of training data)

Notes:
  • Only support vectors define decision boundary
  • RBF Kernel enables nonlinear class separation
"""
        ax4.text(0.02, 0.98, info, va="top", fontsize=10,
                 fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.4))

        # --------------------------------------------------
        # Summary
        # --------------------------------------------------
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis("off")

        summary = f"""
DATASET SUMMARY
===============================================
Train Samples: {len(self.X_train)}
Test Samples : {len(self.X_test)}
Features     : {self.X_train.shape[1]}

PERFORMANCE:
  Accuracy : {self.accuracy:.4f}
  Precision: {self.precision:.4f}
  Recall   : {self.recall:.4f}
  F1-Score : {self.f1:.4f}
"""
        ax5.text(0.02, 0.98, summary, va="top",
                 fontfamily="monospace",
                 fontsize=10,
                 bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5))

        # --------------------------------------------------
        # Footer
        # --------------------------------------------------
        ax6 = fig.add_subplot(gs[3, :])
        ax6.axis("off")
        ax6.text(0.5, 0.5,
                 "SVM Binary Classification — Visualization Complete",
                 ha="center", fontsize=13, fontweight="bold")

        out_path = f"{plots_dir}/svm_results.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
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
        print("SVM TRAINING COMPLETE!")
        print("=" * 70)
        print(f"Best F1 Score: {self.f1:.4f}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    trainer = SVMTrainer(data_dir="preprocessed_binary_10k")
    trainer.run_complete_pipeline()
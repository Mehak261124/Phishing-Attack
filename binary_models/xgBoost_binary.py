"""
XGBoost Classifier for Binary Classification
Uses: preprocessed_binary_10k/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc,
    classification_report
)
import time
import pickle
import warnings
warnings.filterwarnings("ignore")
import os


class XGBoostTrainer:
    def __init__(self, data_dir="preprocessed_binary_10k"):
        self.data_dir = data_dir
        self.model_name = "XGBoost"
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
    # 2. TRAIN
    # ============================================================
    def train(self):
        print("\n" + "=" * 70)
        print("TRAINING MODEL")
        print("=" * 70)

        self.model = XGBClassifier(
            n_estimators=150,
            max_depth=8,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
            reg_lambda=1.0,
            reg_alpha=0.1
        )

        print(f"\nTraining {self.model_name}...")

        start = time.time()
        self.model.fit(self.X_train, self.y_train)
        self.train_time = time.time() - start

        print(f"✓ Training completed in {self.train_time:.2f}s")

        # Train accuracy
        y_train_pred = self.model.predict(self.X_train)
        print(f"Train Accuracy: {accuracy_score(self.y_train, y_train_pred):.4f}")

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
        print("\nConfusion Matrix:\n", self.cm)

    # ============================================================
    # 4. SAVE MODEL
    # ============================================================
    def save_model(self):
        model_dir = f"{self.data_dir}/models"
        os.makedirs(model_dir, exist_ok=True)

        filepath = f"{model_dir}/xgboost.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)

        print(f"\n✓ Model saved at: {filepath}")

    # ============================================================
    # 5. VISUALIZATION
    # ============================================================
    def plot_results(self):
        print("\n" + "=" * 70)
        print("GENERATING VISUALIZATIONS")
        print("=" * 70)

        plots_dir = f"{self.data_dir}/plots_model"
        os.makedirs(plots_dir, exist_ok=True)

        fig = plt.figure(figsize=(16, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

        # --------------------------------------------------
        # Confusion Matrix
        # --------------------------------------------------
        ax1 = fig.add_subplot(gs[0, 0])
        sns.heatmap(self.cm, annot=True, fmt="d", cmap="Purples",
                    xticklabels=["BENIGN", "ATTACK"],
                    yticklabels=["BENIGN", "ATTACK"],
                    ax=ax1)
        ax1.set_title(f"Confusion Matrix\nAcc: {self.accuracy:.4f}")

        # --------------------------------------------------
        # Metrics
        # --------------------------------------------------
        ax2 = fig.add_subplot(gs[0, 1])
        metrics = ["Accuracy", "Precision", "Recall", "F1"]
        values = [self.accuracy, self.precision, self.recall, self.f1]
        bars = ax2.bar(metrics, values, color=["#9b59b6", "#8e44ad", "#7d3c98", "#6c3483"])
        ax2.set_ylim(0, 1.1)
        ax2.set_title("Performance Metrics")

        for bar, val in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                     f"{val:.4f}", ha="center", fontsize=9)

        # --------------------------------------------------
        # ROC Curve
        # --------------------------------------------------
        ax3 = fig.add_subplot(gs[0, 2])
        fpr, tpr, _ = roc_curve(self.y_test, self.y_proba)
        roc_auc = auc(fpr, tpr)

        ax3.plot(fpr, tpr, color="purple", lw=2, label=f"AUC = {roc_auc:.4f}")
        ax3.plot([0, 1], [0, 1], "k--")
        ax3.legend()
        ax3.set_title("ROC Curve")

        # --------------------------------------------------
        # Feature Importance
        # --------------------------------------------------
        ax4 = fig.add_subplot(gs[1, :])
        feat = pd.DataFrame({
            "feature": self.X_train.columns,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False).head(15)

        bars = ax4.barh(feat["feature"], feat["importance"],
                        color=plt.cm.Purples(np.linspace(0.4, 0.9, len(feat))))
        ax4.set_title("Top 15 Feature Importances")
        ax4.invert_yaxis()

        # --------------------------------------------------
        # Summary
        # --------------------------------------------------
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis("off")

        summary = f"""
MODEL SUMMARY - XGBOOST
===========================================
Accuracy : {self.accuracy:.4f}
Precision: {self.precision:.4f}
Recall   : {self.recall:.4f}
F1-Score : {self.f1:.4f}
Train Time : {self.train_time:.3f}s
Test Time  : {self.test_time:.4f}s

CONFUSION MATRIX:
{self.cm}

INTERPRETATION:
  • Strong performance on structured intrusion detection dataset
  • XGBoost handles imbalance + complexity better
"""
        ax5.text(0.02, 0.98, summary, va="top",
                 fontfamily="monospace",
                 bbox=dict(facecolor="lavender", alpha=0.5))

        output_path = f"{plots_dir}/xgboost_results.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"✓ Saved visualization at: {output_path}")

    # ============================================================
    # RUN FULL PIPELINE
    # ============================================================
    def run_complete_pipeline(self):
        self.train()
        self.evaluate()
        self.save_model()
        self.plot_results()

        print("\n" + "=" * 70)
        print("XGBOOST TRAINING COMPLETE!")
        print("=" * 70)
        print(f"Best F1 Score: {self.f1:.4f}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    trainer = XGBoostTrainer(data_dir="preprocessed_binary_10k")
    trainer.run_complete_pipeline()
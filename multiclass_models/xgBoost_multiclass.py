"""
XGBoost Classifier for MULTICLASS Classification
For 20K Balanced Attack-Only CICIDS Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

import time
import pickle
import warnings
import os
warnings.filterwarnings("ignore")


class XGBoostMulticlassTrainer:
    def __init__(self, data_dir="preprocessed_multiclass_20k"):
        """Initialize Trainer"""

        self.data_dir = data_dir

        # Folder structure
        self.csv_dir = os.path.join(data_dir, "csv")
        self.models_dir = os.path.join(data_dir, "models")
        self.plots_model_dir = os.path.join(data_dir, "plots_model")

        # Create model-plots folder
        os.makedirs(self.plots_model_dir, exist_ok=True)

        self.model_name = "XGBoost Multiclass"
        self.load_data()

    # ============================================================
    # 1. LOAD DATA
    # ============================================================
    def load_data(self):
        print("=" * 80)
        print("TRAINING XGBOOST (MULTICLASS)")
        print("=" * 80)

        print("\nLoading preprocessed multiclass dataset...")

        train_df = pd.read_csv(os.path.join(self.csv_dir, "multiclass_train.csv"))
        test_df = pd.read_csv(os.path.join(self.csv_dir, "multiclass_test.csv"))

        self.X_train = train_df.drop(columns=["Label"])
        self.y_train = train_df["Label"]

        self.X_test = test_df.drop(columns=["Label"])
        self.y_test = test_df["Label"]

        self.class_names = sorted(self.y_train.unique())

        print(f"✓ Training set: {self.X_train.shape}")
        print(f"✓ Test set:     {self.X_test.shape}")
        print(f"✓ Features:     {self.X_train.shape[1]}")
        print(f"✓ Classes:      {len(self.class_names)}\n")

    # ============================================================
    # 2. TRAIN MODEL
    # ============================================================
    def train(self):
        print("=" * 80)
        print("TRAINING MODEL")
        print("=" * 80)

        self.model = XGBClassifier(
            objective="multi:softprob",
            num_class=len(self.class_names),
            eval_metric="mlogloss",
            n_estimators=250,
            learning_rate=0.08,
            max_depth=10,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            random_state=42,
            n_jobs=-1
        )

        print(f"\nTraining {self.model_name}...")
        start = time.time()
        self.model.fit(self.X_train, self.y_train)
        self.train_time = time.time() - start

        print(f"✓ Training completed in {self.train_time:.2f}s")

    # ============================================================
    # 3. EVALUATE MODEL
    # ============================================================
    def evaluate(self):
        print("=" * 80)
        print("EVALUATING MODEL")
        print("=" * 80)

        start = time.time()
        self.y_pred = self.model.predict(self.X_test)
        self.y_proba = self.model.predict_proba(self.X_test)
        self.test_time = time.time() - start

        # Metrics
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        self.precision_macro = precision_score(self.y_test, self.y_pred, average="macro")
        self.recall_macro = recall_score(self.y_test, self.y_pred, average="macro")
        self.f1_macro = f1_score(self.y_test, self.y_pred, average="macro")

        self.cm = confusion_matrix(self.y_test, self.y_pred)

        print(f"\nAccuracy:      {self.accuracy:.4f}")
        print(f"Precision(M):  {self.precision_macro:.4f}")
        print(f"Recall(M):     {self.recall_macro:.4f}")
        print(f"F1-Score(M):   {self.f1_macro:.4f}")
        print(f"Test Time:     {self.test_time:.4f}s\n")

        print("Classification Report:")
        print(classification_report(self.y_test, self.y_pred))

    # ============================================================
    # 4. SAVE MODEL
    # ============================================================
    def save_model(self):
        path = os.path.join(self.models_dir, "xgboost_multiclass.pkl")
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"\n✓ Model saved at: {path}")

    # ============================================================
    # 5. VISUALIZATIONS
    # ============================================================
    def plot_results(self):
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)

        # Output directory
        self.output_plot_dir = os.path.join(
            self.plots_model_dir,
            "xgboost_multiclass_results"
        )
        os.makedirs(self.output_plot_dir, exist_ok=True)

        # ------------------------------------------
        # 1. CONFUSION MATRIX
        # ------------------------------------------
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title(f"Confusion Matrix (Accuracy: {self.accuracy:.4f})", fontsize=16)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(f"{self.output_plot_dir}/confusion_matrix.png")
        plt.close()

        # ------------------------------------------
        # 2. PERFORMANCE METRICS
        # ------------------------------------------
        plt.figure(figsize=(10, 6))
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
        values = [
            self.accuracy,
            self.precision_macro,
            self.recall_macro,
            self.f1_macro
        ]
        sns.barplot(x=metrics, y=values, palette="viridis")
        plt.title("Performance Metrics (Macro Avg)", fontsize=16)
        plt.ylim(0, 1)
        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f"{v:.4f}", ha="center")
        plt.tight_layout()
        plt.savefig(f"{self.output_plot_dir}/performance_metrics.png")
        plt.close()

        # ------------------------------------------
        # 3. ROC CURVES (OVR)
        # ------------------------------------------
        y_test_bin = label_binarize(self.y_test, classes=self.class_names)

        plt.figure(figsize=(11, 10))
        for i, cls in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], self.y_proba[:, i])
            auc_score = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"Class {cls} (AUC={auc_score:.2f})")

        plt.plot([0, 1], [0, 1], "k--")
        plt.title("ROC Curve (One-vs-Rest)", fontsize=16)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(f"{self.output_plot_dir}/roc_curves.png")
        plt.close()

        # ------------------------------------------
        # 4. FEATURE IMPORTANCE
        # ------------------------------------------
        plt.figure(figsize=(14, 10))
        importance_df = pd.DataFrame({
            "Feature": self.X_train.columns,
            "Importance": self.model.feature_importances_
        }).sort_values("Importance", ascending=False).head(20)

        sns.barplot(
            x="Importance",
            y="Feature",
            data=importance_df,
            palette="viridis"
        )
        plt.title("Top 20 Feature Importances", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{self.output_plot_dir}/feature_importance.png")
        plt.close()

        # ------------------------------------------
        # 5. MODEL SUMMARY
        # ------------------------------------------
        plt.figure(figsize=(12, 6))
        plt.axis("off")
        summary = f"""
MODEL SUMMARY – XGBOOST (MULTICLASS)
============================================

Accuracy     : {self.accuracy:.4f}
Precision(M) : {self.precision_macro:.4f}
Recall(M)    : {self.recall_macro:.4f}
F1-Score(M)  : {self.f1_macro:.4f}

Train Time : {self.train_time:.3f}s
Test Time  : {self.test_time:.4f}s
Classes    : {len(self.class_names)}
"""
        plt.text(0.05, 0.95, summary, fontsize=14, fontfamily="monospace")
        plt.tight_layout()
        plt.savefig(f"{self.output_plot_dir}/model_summary.png")
        plt.close()

        print(f"\n✓ All plots saved in: {self.output_plot_dir}")

    # ============================================================
    # RUN ALL
    # ============================================================
    def run_complete_pipeline(self):
        self.train()
        self.evaluate()
        self.save_model()
        self.plot_results()

        print("\n" + "=" * 80)
        print("XGBOOST MULTICLASS TRAINING COMPLETE!")
        print("=" * 80)
        print(f"Final Macro F1: {self.f1_macro:.4f}")


# MAIN
if __name__ == "__main__":
    trainer = XGBoostMulticlassTrainer("preprocessed_multiclass_20k")
    trainer.run_complete_pipeline()
"""
Random Forest Classifier for MULTICLASS Classification
For 20K Attack-Only Multiclass Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
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


class RandomForestMulticlassTrainer:
    def __init__(self, data_dir="preprocessed_multiclass_20k"):
        """Initialize Random Forest Trainer"""

        self.data_dir = data_dir

        # Dataset folders
        self.csv_dir = os.path.join(data_dir, "csv")
        self.models_dir = os.path.join(data_dir, "models")

        # Model plots folder (same level as plots/)
        self.plots_model_dir = os.path.join(data_dir, "plots_model")
        os.makedirs(self.plots_model_dir, exist_ok=True)

        self.model_name = "Random Forest (Multiclass)"
        self.load_data()

    # ============================================================
    # 1. LOAD DATA
    # ============================================================
    def load_data(self):
        print("=" * 80)
        print(f"TRAINING {self.model_name.upper()}")
        print("=" * 80)
        print("\nLoading preprocessed multiclass dataset...")

        train_df = pd.read_csv(os.path.join(self.csv_dir, "multiclass_train.csv"))
        test_df = pd.read_csv(os.path.join(self.csv_dir, "multiclass_test.csv"))

        self.X_train = train_df.drop(columns=["Label"])
        self.y_train = train_df["Label"]

        self.X_test = test_df.drop(columns=["Label"])
        self.y_test = test_df["Label"]

        # Class names
        self.class_names = sorted(self.y_train.unique())

        print(f"✓ Training set: {self.X_train.shape}")
        print(f"✓ Test set:     {self.X_test.shape}")
        print(f"✓ Features:     {self.X_train.shape[1]}\n")

    # ============================================================
    # 2. TRAIN MODEL
    # ============================================================
    def train(self):
        print("=" * 80)
        print("TRAINING MODEL")
        print("=" * 80)

        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=25,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        )

        print(f"\nTraining {self.model_name}...")
        start = time.time()
        self.model.fit(self.X_train, self.y_train)
        self.train_time = time.time() - start

        print(f"✓ Training completed in {self.train_time:.2f}s")

        y_train_pred = self.model.predict(self.X_train)
        print(f"Train Accuracy: {accuracy_score(self.y_train, y_train_pred):.4f}")

    # ============================================================
    # 3. EVALUATE MODEL
    # ============================================================
    def evaluate(self):
        print("=" * 80)
        print("EVALUATING MODEL")
        print("=" * 80)

        start = time.time()
        self.y_pred = self.model.predict(self.X_test)
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
        path = os.path.join(self.models_dir, "random_forest_multiclass.pkl")
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

        print(f"\n✓ Model saved at: {path}")

    # ============================================================
    # 5. VISUALIZATIONS
    # ============================================================
    def plot_results(self):
        print("\n" + "=" * 70)
        print("GENERATING PROFESSIONAL VISUALIZATIONS")
        print("=" * 70)

        self.output_plot_dir = os.path.join(
            self.plots_model_dir,
            "random_forest_multiclass_results"
        )
        os.makedirs(self.output_plot_dir, exist_ok=True)

        # -------------------- 1. Confusion Matrix --------------------
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f"Confusion Matrix (Accuracy: {self.accuracy:.4f})", fontsize=16)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(f"{self.output_plot_dir}/confusion_matrix.png")
        plt.close()

        # -------------------- 2. Metrics Chart --------------------
        plt.figure(figsize=(10, 6))
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
        values = [
            self.accuracy,
            self.precision_macro,
            self.recall_macro,
            self.f1_macro
        ]
        sns.barplot(x=metrics, y=values, palette="viridis")
        plt.ylim(0, 1)
        plt.title("Performance Metrics (Macro Avg)", fontsize=16)
        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f"{v:.4f}", ha="center")
        plt.tight_layout()
        plt.savefig(f"{self.output_plot_dir}/performance_metrics.png")
        plt.close()

        # -------------------- 3. ROC Curves --------------------
        y_test_bin = label_binarize(self.y_test, classes=self.class_names)
        y_proba = self.model.predict_proba(self.X_test)

        plt.figure(figsize=(12, 10))
        for i, cls in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
            auc_val = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"{cls} (AUC={auc_val:.2f})")

        plt.plot([0, 1], [0, 1], 'k--')
        plt.title("ROC Curve (One-vs-Rest)", fontsize=16)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_plot_dir}/roc_curves.png")
        plt.close()

        # -------------------- 4. Feature Importance --------------------
        plt.figure(figsize=(14, 10))
        importance_df = pd.DataFrame({
            "Feature": self.X_train.columns,
            "Importance": self.model.feature_importances_
        }).sort_values("Importance", ascending=False).head(20)

        sns.barplot(y="Feature", x="Importance", data=importance_df, palette="viridis")
        plt.title("Top 20 Important Features", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{self.output_plot_dir}/feature_importance.png")
        plt.close()

        # -------------------- 5. Summary Text --------------------
        plt.figure(figsize=(12, 6))
        plt.axis("off")
        summary = f"""
MODEL SUMMARY – RANDOM FOREST (MULTICLASS)
================================================
Accuracy     : {self.accuracy:.4f}
Precision(M) : {self.precision_macro:.4f}
Recall(M)    : {self.recall_macro:.4f}
F1-Score(M)  : {self.f1_macro:.4f}

Train Time   : {self.train_time:.3f}s
Test Time    : {self.test_time:.4f}s
Classes      : {len(self.class_names)}
"""
        plt.text(0.05, 0.95, summary, fontsize=14, fontfamily="monospace", va="top")
        plt.tight_layout()
        plt.savefig(f"{self.output_plot_dir}/model_summary.png")
        plt.close()

        print("\n✓ All visualizations saved in:", self.output_plot_dir)

    # ============================================================
    # 6. RUN ALL
    # ============================================================
    def run_complete_pipeline(self):
        self.train()
        self.evaluate()
        self.save_model()
        self.plot_results()

        print("\n" + "=" * 80)
        print("MULTICLASS RANDOM FOREST TRAINING COMPLETE!")
        print("=" * 80)
        print(f"Best F1 (Macro): {self.f1_macro:.4f}")


# MAIN
if __name__ == "__main__":
    trainer = RandomForestMulticlassTrainer("preprocessed_multiclass_20k")
    trainer.run_complete_pipeline()
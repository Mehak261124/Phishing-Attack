"""
TabNet Classifier for MULTICLASS Classification
For 20K Attack-Only Multiclass Dataset
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from pytorch_tabnet.tab_model import TabNetClassifier

warnings.filterwarnings("ignore")


class TabNetMulticlassTrainer:
    def __init__(self, data_dir="preprocessed_multiclass_20k"):
        """
        Initialize TabNet Multiclass Trainer
        Uses:
          data_dir/csv/multiclass_train.csv
          data_dir/csv/multiclass_test.csv
        Saves:
          data_dir/models/tabnet_multiclass.zip
          data_dir/plots_model/tabnet_multiclass_results/*.png
        """

        self.data_dir = data_dir

        # Dataset folders
        self.csv_dir = os.path.join(data_dir, "csv")
        self.models_dir = os.path.join(data_dir, "models")
        self.plots_model_dir = os.path.join(data_dir, "plots_model")

        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.plots_model_dir, exist_ok=True)

        # Folder for this model's plots
        self.output_plot_dir = os.path.join(
            self.plots_model_dir, "tabnet_multiclass_results"
        )
        os.makedirs(self.output_plot_dir, exist_ok=True)

        self.model_name = "TabNet (Multiclass)"

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

        # Separate features / labels
        self.X_train_full = train_df.drop(columns=["Label"])
        self.y_train_full = train_df["Label"].astype(int)

        self.X_test = test_df.drop(columns=["Label"])
        self.y_test = test_df["Label"].astype(int)

        # Create small validation split from training set
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train_full,
            self.y_train_full,
            test_size=0.2,
            random_state=42,
            stratify=self.y_train_full,
        )

        # Class names: 0..n-1
        self.class_names = sorted(self.y_train_full.unique())
        self.n_classes = len(self.class_names)

        print(f"✓ Train set : {self.X_train.shape}")
        print(f"✓ Val set   : {self.X_val.shape}")
        print(f"✓ Test set  : {self.X_test.shape}")
        print(f"✓ Features  : {self.X_train.shape[1]}")
        print(f"✓ Classes   : {self.n_classes}\n")

        # Convert to numpy (float32 for TabNet)
        self.X_train_np = self.X_train.values.astype(np.float32)
        self.X_val_np = self.X_val.values.astype(np.float32)
        self.X_test_np = self.X_test.values.astype(np.float32)

        self.y_train_np = self.y_train.values.astype(int)
        self.y_val_np = self.y_val.values.astype(int)
        self.y_test_np = self.y_test.values.astype(int)

    # ============================================================
    # 2. BUILD & TRAIN TABNET
    # ============================================================
    def train(self):
        print("=" * 80)
        print("TRAINING TABNET MODEL")
        print("=" * 80)

        # TabNet hyperparameters (reasonable defaults)
        self.model = TabNetClassifier(
            n_d=32,
            n_a=32,
            n_steps=5,
            gamma=1.5,
            lambda_sparse=1e-4,
            n_independent=2,
            n_shared=2,
            seed=42,
            verbose=10,
        )

        print(f"\nTraining {self.model_name}...")
        start = time.time()

        self.model.fit(
            X_train=self.X_train_np,
            y_train=self.y_train_np,
            eval_set=[(self.X_val_np, self.y_val_np)],
            eval_name=["valid"],
            eval_metric=["accuracy"],
            max_epochs=100,
            patience=10,
            batch_size=2048,
            virtual_batch_size=256,
            num_workers=0,
            drop_last=False,
        )

        self.train_time = time.time() - start
        print(f"\n✓ Training completed in {self.train_time:.3f}s")

        # Optional: training accuracy (on full train set)
        train_preds = self.model.predict(self.X_train_np)
        train_acc = accuracy_score(self.y_train_np, train_preds)
        print(f"Train Accuracy: {train_acc:.4f}")

    # ============================================================
    # 3. EVALUATE
    # ============================================================
    def evaluate(self):
        print("=" * 80)
        print("EVALUATING TABNET MODEL")
        print("=" * 80)

        start = time.time()
        self.y_pred = self.model.predict(self.X_test_np)
        self.test_time = time.time() - start

        # Proba for ROC curves
        self.y_proba = self.model.predict_proba(self.X_test_np)

        # Metrics (macro)
        self.accuracy = accuracy_score(self.y_test_np, self.y_pred)
        self.precision_macro = precision_score(
            self.y_test_np, self.y_pred, average="macro"
        )
        self.recall_macro = recall_score(
            self.y_test_np, self.y_pred, average="macro"
        )
        self.f1_macro = f1_score(self.y_test_np, self.y_pred, average="macro")

        self.cm = confusion_matrix(self.y_test_np, self.y_pred)

        print(f"\nAccuracy:      {self.accuracy:.4f}")
        print(f"Precision(M):  {self.precision_macro:.4f}")
        print(f"Recall(M):     {self.recall_macro:.4f}")
        print(f"F1-Score(M):   {self.f1_macro:.4f}")
        print(f"Test Time:     {self.test_time:.4f}s\n")

        print("Classification Report:")
        print(
            classification_report(
                self.y_test_np,
                self.y_pred,
                digits=4,
            )
        )

    # ============================================================
    # 4. SAVE MODEL
    # ============================================================
    def save_model(self):
        model_path = os.path.join(self.models_dir, "tabnet_multiclass")
        # TabNet saves several files with this prefix
        self.model.save_model(model_path)
        print(f"\n✓ Model saved with prefix: {model_path}*")

    # ============================================================
    # 5. VISUALIZATIONS
    # ============================================================
    def plot_results(self):
        print("\n" + "=" * 70)
        print("GENERATING TABNET VISUALIZATIONS")
        print("=" * 70)

        # ----------------- 1. Confusion Matrix -----------------
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            self.cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            linewidths=0.5,
            square=True,
            cbar=True,
        )
        plt.title(
            f"Confusion Matrix (Accuracy: {self.accuracy:.4f})",
            fontsize=16,
            fontweight="bold",
        )
        plt.xlabel("Predicted", fontsize=14)
        plt.ylabel("True", fontsize=14)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_plot_dir, "confusion_matrix.png"))
        plt.close()

        # ----------------- 2. Performance Metrics Bar -----------------
        plt.figure(figsize=(10, 6))
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
        values = [
            self.accuracy,
            self.precision_macro,
            self.recall_macro,
            self.f1_macro,
        ]
        sns.barplot(x=metrics, y=values, palette="viridis")
        plt.title("Performance Metrics (Macro Avg)", fontsize=16, fontweight="bold")
        plt.ylim(0, 1)
        plt.grid(axis="y", alpha=0.3)
        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f"{v:.4f}", ha="center", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_plot_dir, "performance_metrics.png"))
        plt.close()

        # ----------------- 3. ROC Curves (One-vs-Rest) -----------------
        # Binarize labels
        y_test_bin = label_binarize(self.y_test_np, classes=self.class_names)

        plt.figure(figsize=(12, 10))
        for i, cls in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], self.y_proba[:, i])
            auc_score = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=1.5, label=f"Class {cls} (AUC={auc_score:.2f})")

        # Micro-average
        fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), self.y_proba.ravel())
        plt.plot(
            fpr_micro,
            tpr_micro,
            "k--",
            lw=2,
            label=f"Micro Avg AUC={auc(fpr_micro, tpr_micro):.2f}",
        )

        plt.plot([0, 1], [0, 1], "r--", lw=1)
        plt.xlabel("False Positive Rate", fontsize=14)
        plt.ylabel("True Positive Rate", fontsize=14)
        plt.title("ROC Curve (One-vs-Rest)", fontsize=16, fontweight="bold")
        plt.legend(loc="lower right", fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_plot_dir, "roc_curves.png"))
        plt.close()

        # ----------------- 4. Feature Importances -----------------
        try:
            importances = self.model.feature_importances_
            plt.figure(figsize=(14, 10))
            importance_df = pd.DataFrame(
                {"Feature": self.X_train.columns, "Importance": importances}
            ).sort_values("Importance", ascending=False).head(20)

            sns.barplot(
                x="Importance",
                y="Feature",
                data=importance_df,
                palette="viridis",
            )
            plt.title(
                "Top 20 Feature Importances (TabNet)",
                fontsize=16,
                fontweight="bold",
            )
            plt.xlabel("Importance")
            plt.ylabel("Feature")
            plt.tight_layout()
            plt.savefig(
                os.path.join(self.output_plot_dir, "feature_importance.png")
            )
            plt.close()
        except Exception as e:
            print("⚠ Could not compute TabNet feature_importances_:", e)

        # ----------------- 5. Model Summary Text -----------------
        plt.figure(figsize=(12, 6))
        plt.axis("off")
        summary = f"""
MODEL SUMMARY – TABNET (MULTICLASS)
===============================================
Accuracy(M)   : {self.accuracy:.4f}
Precision(M)  : {self.precision_macro:.4f}
Recall(M)     : {self.recall_macro:.4f}
F1-Score(M)   : {self.f1_macro:.4f}

Train Time    : {self.train_time:.3f}s
Test Time     : {self.test_time:.4f}s
Classes       : {self.n_classes}
"""
        plt.text(
            0.05,
            0.95,
            summary,
            fontsize=14,
            fontfamily="monospace",
            verticalalignment="top",
        )
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_plot_dir, "model_summary.png"))
        plt.close()

        print("\n✓ All TabNet visualizations saved in:", self.output_plot_dir)

    # ============================================================
    # 6. RUN COMPLETE PIPELINE
    # ============================================================
    def run_complete_pipeline(self):
        self.train()
        self.evaluate()
        self.save_model()
        self.plot_results()

        print("\n" + "=" * 80)
        print("TABNET MULTICLASS TRAINING COMPLETE!")
        print("=" * 80)
        print(f"Best F1 (Macro): {self.f1_macro:.4f}")


# MAIN
if __name__ == "__main__":
    trainer = TabNetMulticlassTrainer(data_dir="preprocessed_multiclass_20k")
    trainer.run_complete_pipeline()
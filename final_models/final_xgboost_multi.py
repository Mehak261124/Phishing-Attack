# ================================
# FINAL XGBOOST MULTICLASS PIPELINE
# ================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve
)
from sklearn.preprocessing import label_binarize

from xgboost import XGBClassifier


# ================================
# FOLDER STRUCTURE
# ================================
BASE_DIR = "final_preprocessed_data"
CSV_PATH = f"{BASE_DIR}/csv/multiclass_preprocessed.csv"

PLOT_DIR = f"{BASE_DIR}/plot_models/multiclass_xgb"
MODEL_DIR = f"{BASE_DIR}/models"

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"\nSaving all visualizations to: {PLOT_DIR}/")
print(f"Saving trained model to: {MODEL_DIR}/xgb_multiclass.pkl")


# ================================
# STEP 1 — LOAD DATA
# ================================
print("\nLoading final preprocessed multiclass dataset...")
df = pd.read_csv(CSV_PATH)
print("Dataset Shape:", df.shape)

X = df.drop("Multiclass_Label", axis=1).values
y = df["Multiclass_Label"].values

classes = np.unique(y)
n_classes = len(classes)
print("Number of classes:", n_classes)


# ================================
# STEP 2 — TRAIN/TEST SPLIT
# ================================
print("\nPerforming Train/Test Split...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print("Train Shape:", X_train.shape)
print("Test Shape:", X_test.shape)


# ================================
# STEP 3 — TRAIN XGBOOST
# ================================
print("\nTraining XGBoost Multiclass Model...")

model = XGBClassifier(
    objective="multi:softprob",
    num_class=n_classes,
    n_estimators=300,
    max_depth=10,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="mlogloss",
    tree_method="hist"
)

model.fit(X_train, y_train)
print("\nTraining Complete!")


# ================================
# SAVE MODEL
# ================================
pickle.dump(model, open(f"{MODEL_DIR}/xgb_multiclass.pkl", "wb"))
print("Model saved as xgb_multiclass.pkl")


# ================================
# STEP 4 — PREDICTIONS
# ================================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)  # shape: [n_samples, n_classes]

print("\nClassification Report:")
cr = classification_report(y_test, y_pred)
print(cr)

with open(f"{PLOT_DIR}/xgb_multiclass_classification_report.txt", "w") as f:
    f.write(cr)


# ================================
# STEP 5 — CONFUSION MATRIX
# ================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=False, fmt='d', cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("XGBoost Multiclass Confusion Matrix")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/xgb_multiclass_confusion_matrix.png")
plt.close()


# ================================
# STEP 6 — MULTICLASS ROC & PR (MICRO-AVERAGED)
# ================================
print("\nComputing micro-averaged ROC and PR curves...")

y_test_bin = label_binarize(y_test, classes=classes)  # (n_samples, n_classes)

# ROC (micro)
fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_prob.ravel())
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"Micro-AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multiclass ROC Curve (Micro-Average)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/xgb_multiclass_roc_micro.png")
plt.close()

# Precision–Recall (micro)
prec, rec, _ = precision_recall_curve(y_test_bin.ravel(), y_prob.ravel())

plt.figure(figsize=(6, 5))
plt.plot(rec, prec)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Multiclass Precision–Recall Curve (Micro-Average)")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/xgb_multiclass_pr_micro.png")
plt.close()


# ================================
# STEP 7 — FEATURE IMPORTANCE
# ================================
importance = model.feature_importances_
indices = np.argsort(importance)[-15:]  # top 15 PCs

plt.figure(figsize=(10, 6))
plt.barh([f"PC{i+1}" for i in indices], importance[indices])
plt.xlabel("Importance Score")
plt.title("Top 15 PCA Feature Importances (XGBoost Multiclass)")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/xgb_multiclass_feature_importance.png")
plt.close()


# ================================
# DONE
# ================================
print("\nSaved in:", PLOT_DIR)
for file in os.listdir(PLOT_DIR):
    print("  -", file)

print("\n========== XGBOOST MULTICLASS TRAINING COMPLETE ==========")

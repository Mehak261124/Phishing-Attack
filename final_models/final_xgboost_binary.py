# ================================
# FINAL XGBOOST BINARY CLASSIFICATION PIPELINE
# ================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve
)

from xgboost import XGBClassifier


# ================================
# FOLDER STRUCTURE
# ================================
BASE_DIR = "final_preprocessed_data"
CSV_PATH = f"{BASE_DIR}/csv/binary_preprocessed.csv"

PLOT_DIR = f"{BASE_DIR}/plot_models/binary"
MODEL_DIR = f"{BASE_DIR}/models"

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"\nSaving all visualizations to: {PLOT_DIR}/")
print(f"Saving trained model to: {MODEL_DIR}/xgb_binary.pkl")


# ================================
# STEP 1 — LOAD DATA
# ================================
print("\nLoading final preprocessed binary dataset...")
df = pd.read_csv(CSV_PATH)
print("Dataset Shape:", df.shape)

X = df.drop("Binary_Label", axis=1).values
y = df["Binary_Label"].values


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
print("\nTraining XGBoost Model...")

model = XGBClassifier(
    n_estimators=250,
    max_depth=8,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss",
    tree_method="hist"
)

model.fit(X_train, y_train)
print("\nTraining Complete!")


# ================================
# SAVE MODEL
# ================================
pickle.dump(model, open(f"{MODEL_DIR}/xgb_binary.pkl", "wb"))
print("Model saved as xgb_binary.pkl")


# ================================
# STEP 4 — PREDICTIONS
# ================================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
cr = classification_report(y_test, y_pred)
print(cr)

with open(f"{PLOT_DIR}/xgb_classification_report.txt", "w") as f:
    f.write(cr)


# ================================
# STEP 5 — VISUALIZATIONS
# ================================

# ----- CONFUSION MATRIX -----
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/xgb_confusion_matrix.png")
plt.close()


# ----- ROC CURVE -----
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0,1], [0,1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/xgb_roc_curve.png")
plt.close()


# ----- PRECISION–RECALL CURVE -----
prec, rec, _ = precision_recall_curve(y_test, y_prob)

plt.figure(figsize=(6,5))
plt.plot(rec, prec, color="purple")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/xgb_pr_curve.png")
plt.close()


# ----- FEATURE IMPORTANCE -----
importance = model.feature_importances_
indices = np.argsort(importance)[-15:]  # top 15 features

plt.figure(figsize=(10,6))
plt.barh([f"PC{i+1}" for i in indices], importance[indices], color="green")
plt.xlabel("Importance Score")
plt.title("Top 15 PCA Feature Importances (XGBoost)")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/xgb_feature_importance.png")
plt.close()


# ================================
# DONE
# ================================
print("\nSaved in:", PLOT_DIR)
for file in os.listdir(PLOT_DIR):
    print("  -", file)

print("\n========== XGBOOST BINARY TRAINING COMPLETE ==========")

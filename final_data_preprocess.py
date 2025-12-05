# ========= FINAL CICIDS2017 PREPROCESSING PIPELINE (CPU SAFE) ==========
# Includes:
# • Load & clean dataset
# • Binary + Multiclass labels
# • Benign downsampling (memory safe)
# • Scaling → scaler.pkl  (with feature_names_in_)
# • Binary oversampling
# • Multiclass capping (NO oversampling)
# • KMeans clustering embedding
# • PCA (20 components) → pca_bin.pkl, pca_multi.pkl
# • Train/Test split saved as CSV
# • ALL visualizations saved in separate folders
# =======================================================================

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler


# ========= Create folder structure ==========
os.makedirs("final_preprocessed_data", exist_ok=True)
os.makedirs("final_preprocessed_data/models", exist_ok=True)
os.makedirs("final_preprocessed_data/csv", exist_ok=True)
os.makedirs("final_preprocessed_data/pca_visualizations", exist_ok=True)
os.makedirs("final_preprocessed_data/stats_visualizations", exist_ok=True)

print("Folders created successfully.")


# ===========================
# STEP 1: LOAD CSV FILES
# ===========================
folder = "TrafficLabelling"
files = glob.glob(os.path.join(folder, "*.csv"))

df_list = []
print("\nLoading CSV files...\n")
for file in files:
    print("Loading:", file)
    df_list.append(pd.read_csv(file, encoding="ISO-8859-1", low_memory=False))

df = pd.concat(df_list, ignore_index=True)
print("\nDataset Combined Shape:", df.shape)


# ===========================
# STEP 2: CLEAN DATA
# ===========================
df.columns = df.columns.str.strip().str.replace(" ", "_")
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()
df = df.drop_duplicates()

print("After Cleaning:", df.shape)


# ===========================
# STEP 3: LABEL GENERATION
# ===========================
enc = LabelEncoder()
df["Multiclass_Label"] = enc.fit_transform(df["Label"])
df["Binary_Label"] = df["Label"].apply(lambda x: 0 if x.lower() == "benign" else 1)

print("\nBinary Counts:\n", df["Binary_Label"].value_counts())
print("\nMulticlass Counts:\n", df["Multiclass_Label"].value_counts())


# ===========================
# STEP 4: BENIGN REDUCTION
# ===========================
benign_df = df[df["Binary_Label"] == 0]
attack_df = df[df["Binary_Label"] == 1]

target_benign = int(len(attack_df) * 1.2)
benign_df = benign_df.sample(n=target_benign, random_state=42)

df_reduced = pd.concat([benign_df, attack_df], ignore_index=True)
df_reduced = df_reduced.sample(frac=1, random_state=42)

print("\nAfter Benign Reduction:", df_reduced.shape)


# ===========================
# STEP 5: NUMERIC FEATURES
# ===========================
# Keep as DataFrame (IMPORTANT: for feature_names_in_)
features_df = df_reduced.drop(["Label", "Binary_Label", "Multiclass_Label"], axis=1)
numeric_features = features_df.select_dtypes(include=["float64", "int64"])

feature_names = numeric_features.columns.tolist()
print(f"\nNumber of numeric features: {len(feature_names)}")

# X as DataFrame here
X_df = numeric_features


# ===========================
# STEP 6: SCALING  (FIXED)
# ===========================
scaler = StandardScaler()

# Fit on DataFrame (so scaler.feature_names_in_ is available)
scaler.fit(X_df)

# Get scaled numpy array
X_scaled = scaler.transform(X_df)

# Save scaler
pickle.dump(scaler, open("final_preprocessed_data/models/scaler.pkl", "wb"))
# Optionally also save feature names
pickle.dump(feature_names, open("final_preprocessed_data/models/feature_names.pkl", "wb"))

print("\nScaler saved → final_preprocessed_data/models/scaler.pkl")
print("Feature names saved → final_preprocessed_data/models/feature_names.pkl")


# ===========================
# STEP 7A: BINARY OVERSAMPLING
# ===========================
y_bin = df_reduced["Binary_Label"].values
oversampler = RandomOverSampler()
X_bin, y_bin = oversampler.fit_resample(X_scaled, y_bin)
print("Binary After Oversampling:", X_bin.shape)


# ===========================
# STEP 7B: MULTICLASS CAPPING
# ===========================
max_samples_per_class = 60000
y_multi = df_reduced["Multiclass_Label"].values

df_multi = pd.DataFrame(X_scaled)
df_multi["label"] = y_multi

chunks = []
for lbl, group in df_multi.groupby("label"):
    if len(group) > max_samples_per_class:
        group = group.sample(max_samples_per_class, random_state=42)
    chunks.append(group)

df_multi_bal = pd.concat(chunks, ignore_index=True)

X_multi = df_multi_bal.drop("label", axis=1).values
y_multi = df_multi_bal["label"].values

print("Multiclass After Balancing:", X_multi.shape)


# ===========================
# STEP 8: KMEANS EMBEDDING
# ===========================
kmeans_bin = KMeans(n_clusters=6, random_state=42)
cluster_bin = kmeans_bin.fit_predict(X_bin)
X_bin_embed = np.hstack([X_bin, cluster_bin.reshape(-1, 1)])

kmeans_multi = KMeans(n_clusters=10, random_state=42)
cluster_multi = kmeans_multi.fit_predict(X_multi)
X_multi_embed = np.hstack([X_multi, cluster_multi.reshape(-1, 1)])

print("Clustering embedded.")

# Save KMeans models for future live inference (OPTIONAL BUT USEFUL)
pickle.dump(kmeans_bin, open("final_preprocessed_data/models/kmeans_bin.pkl", "wb"))
pickle.dump(kmeans_multi, open("final_preprocessed_data/models/kmeans_multi.pkl", "wb"))
print("KMeans models saved.")


# IMPORTANT: Add cluster names
feature_names_bin = feature_names + ["cluster_bin"]
feature_names_multi = feature_names + ["cluster_multi"]


# ===========================
# STEP 9: PCA (FIXED: SAVE MODELS)
# ===========================
pca_bin = PCA(n_components=20)
X_bin_pca = pca_bin.fit_transform(X_bin_embed)

pca_multi = PCA(n_components=20)
X_multi_pca = pca_multi.fit_transform(X_multi_embed)

print("PCA applied.")

# Save PCA models for later (live inference)
pickle.dump(pca_bin, open("final_preprocessed_data/models/pca_bin.pkl", "wb"))
pickle.dump(pca_multi, open("final_preprocessed_data/models/pca_multi.pkl", "wb"))
print("PCA models saved → pca_bin.pkl, pca_multi.pkl")


# ===========================
# STEP 10: TRAIN/TEST SPLIT
# ===========================
binary_cols = [f"PC{i+1}" for i in range(20)] + ["Binary_Label"]
multi_cols  = [f"PC{i+1}" for i in range(20)] + ["Multiclass_Label"]

# Binary
Xb_train, Xb_test, yb_train, yb_test = train_test_split(
    X_bin_pca, y_bin, test_size=0.2, random_state=42
)

# Multiclass
Xm_train, Xm_test, ym_train, ym_test = train_test_split(
    X_multi_pca, y_multi, test_size=0.2, random_state=42
)

# Save CSVs
pd.DataFrame(np.column_stack([Xb_train, yb_train]), columns=binary_cols) \
    .to_csv("final_preprocessed_data/csv/binary_train.csv", index=False)

pd.DataFrame(np.column_stack([Xb_test, yb_test]), columns=binary_cols) \
    .to_csv("final_preprocessed_data/csv/binary_test.csv", index=False)

pd.DataFrame(np.column_stack([Xm_train, ym_train]), columns=multi_cols) \
    .to_csv("final_preprocessed_data/csv/multiclass_train.csv", index=False)

pd.DataFrame(np.column_stack([Xm_test, ym_test]), columns=multi_cols) \
    .to_csv("final_preprocessed_data/csv/multiclass_test.csv", index=False)

# Full processed CSVs
pd.DataFrame(np.column_stack([X_bin_pca, y_bin]), columns=binary_cols) \
    .to_csv("final_preprocessed_data/csv/binary_preprocessed.csv", index=False)

pd.DataFrame(np.column_stack([X_multi_pca, y_multi]), columns=multi_cols) \
    .to_csv("final_preprocessed_data/csv/multiclass_preprocessed.csv", index=False)

print("All CSVs saved.")


# ===========================
# STEP 11: VISUALIZATIONS
# ===========================
print("\nGenerating visualizations...")

# 1) Binary class distribution
plt.figure(figsize=(10,6))
sns.countplot(x=df_reduced["Binary_Label"])
plt.title("Binary Class Distribution")
plt.savefig("final_preprocessed_data/stats_visualizations/binary_class_dist.png")
plt.close()

# 2) Multiclass class distribution
plt.figure(figsize=(12,6))
sns.countplot(x=df_multi_bal["label"])
plt.title("Multiclass Class Distribution")
plt.savefig("final_preprocessed_data/stats_visualizations/multiclass_class_dist.png")
plt.close()

# 3) Correlation heatmap
corr = pd.DataFrame(X_scaled, columns=feature_names).iloc[:, :20].corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr, cmap="coolwarm")
plt.title("Correlation Heatmap (Top 20 Features)")
plt.savefig("final_preprocessed_data/stats_visualizations/corr_heatmap_top20.png")
plt.close()

# 4) PCA Variance
plt.figure(figsize=(10,5))
plt.plot(range(1,21), pca_bin.explained_variance_ratio_, marker='o')
plt.title("PCA Variance Explained (Binary Dataset)")
plt.savefig("final_preprocessed_data/pca_visualizations/viz_pca_variance.png")
plt.close()

# 5) PCA Scatter Binary
plt.figure(figsize=(8,6))
plt.scatter(X_bin_pca[:,0], X_bin_pca[:,1], c=y_bin, cmap="coolwarm", s=3)
plt.title("PCA Scatter (Binary)")
plt.savefig("final_preprocessed_data/pca_visualizations/viz_pca_scatter_binary.png")
plt.close()

# 6) PCA Scatter Multiclass
plt.figure(figsize=(8,6))
plt.scatter(X_multi_pca[:,0], X_multi_pca[:,1], c=y_multi, cmap="viridis", s=3)
plt.title("PCA Scatter (Multiclass)")
plt.savefig("final_preprocessed_data/pca_visualizations/viz_pca_scatter_multiclass.png")
plt.close()

# 7) PCA Feature Contribution for PC1–PC20
loadings = pca_bin.components_
top_n = 12

for comp in range(20):
    pc_vec = loadings[comp]
    idx = np.argsort(np.abs(pc_vec))[::-1][:top_n]

    names = [feature_names_bin[i] for i in idx]

    plt.figure(figsize=(10,6))
    plt.barh(names, pc_vec[idx], color=["green" if v > 0 else "red" for v in pc_vec[idx]])
    plt.title(f"Top {top_n} Features → PC{comp+1}")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"final_preprocessed_data/pca_visualizations/viz_pc{comp+1}_top_features.png")
    plt.close()

print("All visualizations saved!")
print("\n========== PREPROCESSING COMPLETE ==========")

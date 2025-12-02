import pandas as pd
import numpy as np
import os
import pickle
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")


class CICIDS_Multiclass20k_NoBenign:
    def __init__(self, folder="TrafficLabelling", output_folder="preprocessed_multiclass_20k", total_size=20000):
        self.folder = folder
        self.output_folder = output_folder
        self.total_size = total_size
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()
        self.df = None

        # FINAL DEPLOYMENT STRUCTURE
        self.csv_dir = os.path.join(self.output_folder, "csv")
        self.models_dir = os.path.join(self.output_folder, "models")
        self.plots_dir = os.path.join(self.output_folder, "plots")
        self.meta_dir = os.path.join(self.output_folder, "meta")

        # Create folder tree
        for d in [self.output_folder, self.csv_dir, self.models_dir, self.plots_dir, self.meta_dir]:
            os.makedirs(d, exist_ok=True)

    # ============================================================
    # 1. LOAD ALL CSV FILES
    # ============================================================
    def load_all(self):
        print("\n[1] Loading CSV files...")

        files = [f for f in os.listdir(self.folder) if f.endswith(".csv")]
        print(f"   → Found {len(files)} files.\n")

        frames = []
        for f in files:
            path = os.path.join(self.folder, f)
            print(f"   Loading: {f}")

            try:
                df = pd.read_csv(path, low_memory=False)
            except:
                try:
                    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
                except:
                    df = pd.read_csv(path, encoding="cp1252", low_memory=False)

            df.columns = df.columns.str.strip()
            frames.append(df)
            print(f"      ✓ Loaded ({df.shape[0]} rows)\n")

        self.df = pd.concat(frames, ignore_index=True)
        self.df["Label"] = self.df["Label"].astype(str).str.strip()

        print(f"   TOTAL MERGED SHAPE: {self.df.shape}\n")
        return self.df

    # ============================================================
    # 2. CLEAN DATA & REMOVE BENIGN
    # ============================================================
    def clean(self):
        print("[2] Cleaning dataset...")

        drop_cols = [
            "Flow ID", "Flow_ID", "Source IP", "Destination IP",
            "Timestamp", "Protocol", "Source Port", "Destination Port"
        ]

        X = self.df.copy()

        # REMOVE BENIGN
        before = len(X)
        X = X[X["Label"] != "BENIGN"]
        print(f"   Removed BENIGN rows: {before - len(X)}")

        # Drop useless columns
        X.drop(columns=[c for c in drop_cols if c in X.columns], inplace=True)

        # Convert to numeric
        for col in X.columns:
            if col != "Label":
                X[col] = pd.to_numeric(X[col], errors="coerce")

        # Remove invalid numeric rows
        before2 = len(X)
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.dropna(inplace=True)
        print(f"   Removed {before2 - len(X)} invalid rows")

        print(f"   Cleaned shape: {X.shape}\n")

        self.df = X
        return self.df

    # ============================================================
    # 3. BALANCE ATTACK CLASSES TO EXACTLY 20K
    # ============================================================
    def balance_dataset(self):
        print("[3] Creating balanced 20k dataset (ATTACKS ONLY)...")

        labels = self.df["Label"].unique()
        num_classes = len(labels)

        target_per_class = self.total_size // num_classes

        print(f"   → {num_classes} attack classes detected")
        print(f"   → Target per class = {target_per_class}\n")

        balanced_frames = []

        for label in labels:
            class_rows = self.df[self.df["Label"] == label]
            n = len(class_rows)

            print(f"   Class: {label} → {n} rows")

            if n >= target_per_class:
                sampled = class_rows.sample(target_per_class, random_state=42)
            else:
                sampled = class_rows.sample(target_per_class, replace=True, random_state=42)
                print(f"     ⚠ Rare class oversampled to {target_per_class}")

            balanced_frames.append(sampled)

        final_df = pd.concat(balanced_frames, ignore_index=True)
        final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"\n   Final attack-only balanced dataset = {final_df.shape}\n")

        self.df = final_df
        return self.df

    # ============================================================
    # 4. ENCODE + SCALE
    # ============================================================
    def encode_and_scale(self):
        print("[4] Encoding labels and scaling features...\n")

        y = self.df["Label"]
        X = self.df.drop(columns=["Label"])

        y_encoded = self.encoder.fit_transform(y)
        X_scaled = self.scaler.fit_transform(X)

        self.X = pd.DataFrame(X_scaled, columns=X.columns)
        self.y = y_encoded

        # Save metadata
        with open(os.path.join(self.meta_dir, "class_mapping.txt"), "w") as f:
            for idx, name in enumerate(self.encoder.classes_):
                f.write(f"{idx} : {name}\n")

        with open(os.path.join(self.meta_dir, "feature_names.txt"), "w") as f:
            for col in X.columns:
                f.write(col + "\n")

        # Save model artifacts
        with open(os.path.join(self.models_dir, "label_encoder.pkl"), "wb") as f:
            pickle.dump(self.encoder, f)

        with open(os.path.join(self.models_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(self.scaler, f)

        print("   ✓ Encoding + Scaling saved.\n")

        return self.X, self.y

    # ============================================================
    # 5. TRAIN/TEST SPLIT
    # ============================================================
    def split(self, test_size=0.2):
        print("[5] Splitting into train & test...\n")

        return train_test_split(
            self.X, self.y,
            test_size=test_size,
            random_state=42,
            stratify=self.y
        )

    # ============================================================
    # 6. SAVE CSV FILES
    # ============================================================
    def save(self, X_train, X_test, y_train, y_test):
        print("[6] Saving CSV files...")

        train_df = X_train.copy()
        train_df["Label"] = y_train

        test_df = X_test.copy()
        test_df["Label"] = y_test

        train_df.to_csv(os.path.join(self.csv_dir, "multiclass_train.csv"), index=False)
        test_df.to_csv(os.path.join(self.csv_dir, "multiclass_test.csv"), index=False)

        combined = pd.concat([train_df, test_df])
        combined.to_csv(os.path.join(self.csv_dir, "multiclass_preprocessed.csv"), index=False)

        print("   ✓ CSV files saved.\n")

    # ============================================================
    # 7. VISUALIZATIONS
    # ============================================================
    def visualizations(self):
        print("[7] Creating plots...\n")

        # Class Distribution
        plt.figure(figsize=(12, 5))
        sns.countplot(x=self.encoder.inverse_transform(self.y))
        plt.xticks(rotation=90)
        plt.title("Class Distribution (20K Balanced Attack-Only Dataset)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "class_distribution.png"))
        plt.close()

        # Correlation Heatmap
        plt.figure(figsize=(12, 9))
        sns.heatmap(pd.DataFrame(self.X).iloc[:, :25].corr(),
                    cmap="coolwarm", annot=False)
        plt.title("Correlation Heatmap (Top 25 Features)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "correlation_heatmap.png"))
        plt.close()

        # PCA Visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X)

        plt.figure(figsize=(10, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=self.y, s=5, cmap="tab20")
        plt.title("PCA Scatter Plot (Attack-Only Multiclass Data)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "pca_plot.png"))
        plt.close()

        print("   ✓ Visualizations saved.\n")


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    pre = CICIDS_Multiclass20k_NoBenign()

    pre.load_all()
    pre.clean()
    pre.balance_dataset()
    pre.encode_and_scale()
    X_train, X_test, y_train, y_test = pre.split()
    pre.save(X_train, X_test, y_train, y_test)
    pre.visualizations()

    print("\nALL DONE ✔ Your attack-only multiclass dataset (20K) is ready.\n")
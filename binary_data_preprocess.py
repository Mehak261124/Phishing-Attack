import pandas as pd
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


class BinaryPreprocessAndVisualize10K:
    def __init__(self, folder="TrafficLabelling", output_folder="preprocessed_binary_10k"):
        self.folder = folder
        self.output_folder = output_folder
        self.total_size = 10000
        self.scaler = StandardScaler()

        # Final Deployment Folder Structure
        self.csv_dir = os.path.join(self.output_folder, "csv")
        self.models_dir = os.path.join(self.output_folder, "models")
        self.plots_dir = os.path.join(self.output_folder, "plots")
        self.meta_dir = os.path.join(self.output_folder, "meta")

        for d in [self.output_folder, self.csv_dir, self.models_dir, self.plots_dir, self.meta_dir]:
            os.makedirs(d, exist_ok=True)

    # ============================================================
    # 1. LOAD ALL FILES
    # ============================================================
    def load_all(self):
        print("\n[1] Loading all CSV files...")

        files = [f for f in os.listdir(self.folder) if f.endswith(".csv")]
        frames = []

        for f in files:
            path = os.path.join(self.folder, f)
            print("   →", f)

            try:
                df = pd.read_csv(path, low_memory=False)
            except:
                df = pd.read_csv(path, encoding="latin-1", low_memory=False)

            df.columns = df.columns.str.strip()

            if "Label" not in df.columns:
                print(f"❌ Skipping (no Label column): {f}")
                continue

            df["Label"] = df["Label"].astype(str).str.strip()
            frames.append(df)

        self.df = pd.concat(frames, ignore_index=True)
        print("   MERGED SHAPE:", self.df.shape)
        return self.df

    # ============================================================
    # 2. CLEAN
    # ============================================================
    def clean(self):
        print("\n[2] Cleaning dataset...")

        drop_cols = [
            "Flow ID", "Flow_ID",
            "Source IP", "Destination IP",
            "Timestamp", "Protocol",
            "Source Port", "Destination Port"
        ]

        X = self.df.copy()
        X.drop(columns=[c for c in drop_cols if c in X.columns], inplace=True)

        for col in X.columns:
            if col != "Label":
                X[col] = pd.to_numeric(X[col], errors="coerce")

        before = len(X)
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.dropna(inplace=True)
        print(f"   Removed {before - len(X)} invalid rows")

        self.df = X
        print("   Cleaned shape:", self.df.shape)
        return self.df

    # ============================================================
    # 3. BALANCE TO 10K (5000 BENIGN + 5000 ATTACK)
    # ============================================================
    def balance(self):
        print("\n[3] Balancing (5000 benign + 5000 attack)...")

        benign = self.df[self.df["Label"] == "BENIGN"]
        attack = self.df[self.df["Label"] != "BENIGN"]

        benign_sample = benign.sample(5000, random_state=42)
        attack_sample = attack.sample(5000, random_state=42)

        df_final = pd.concat([benign_sample, attack_sample], ignore_index=True)
        df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

        df_final["Binary_Label"] = df_final["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)
        df_final.drop(columns=["Label"], inplace=True)

        print("   Balanced shape:", df_final.shape)

        self.df = df_final
        return self.df

    # ============================================================
    # 4. SCALING
    # ============================================================
    def scale(self):
        print("\n[4] Scaling features...")

        y = self.df["Binary_Label"]
        X = self.df.drop(columns=["Binary_Label"])

        X_scaled = self.scaler.fit_transform(X)

        # Save scaler
        with open(os.path.join(self.models_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(self.scaler, f)

        # Save feature names
        with open(os.path.join(self.meta_dir, "feature_names.txt"), "w") as f:
            for c in X.columns:
                f.write(c + "\n")

        self.X = pd.DataFrame(X_scaled, columns=X.columns)
        self.y = y

        print("   ✓ Scaling done.")
        return self.X, self.y

    # ============================================================
    # 5. SPLIT 80/20
    # ============================================================
    def split(self):
        print("\n[5] Splitting 80/20...")

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y,
            test_size=0.2,
            random_state=42,
            stratify=self.y
        )

        print(f"   Train shape: {X_train.shape}")
        print(f"   Test shape : {X_test.shape}")

        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

        return X_train, X_test, y_train, y_test

    # ============================================================
    # 6. SAVE CSV FILES
    # ============================================================
    def save(self):
        print("\n[6] Saving CSV files...")

        train_df = self.X_train.copy()
        train_df["Binary_Label"] = self.y_train
        train_df.to_csv(os.path.join(self.csv_dir, "binary_train.csv"), index=False)

        test_df = self.X_test.copy()
        test_df["Binary_Label"] = self.y_test
        test_df.to_csv(os.path.join(self.csv_dir, "binary_test.csv"), index=False)

        full_df = pd.concat([train_df, test_df])
        full_df.to_csv(os.path.join(self.csv_dir, "binary_preprocessed.csv"), index=False)

        print("   ✓ Saved all CSV files.")

    # ============================================================
    # 7. VISUALIZATIONS
    # ============================================================
    def visualize(self):
        print("\n[7] Generating visualizations...")

        df = self.df
        X = df.drop(columns=["Binary_Label"])
        y = df["Binary_Label"]

        # 1️⃣ Class Distribution
        plt.figure(figsize=(6, 5))
        sns.countplot(x=y)
        plt.title("Class Distribution (Binary 10K)")
        plt.savefig(f"{self.plots_dir}/class_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 2️⃣ Correlation Heatmap
        variances = X.var().sort_values(ascending=False)
        top_features = variances.head(15).index
        X_top = X[top_features]

        plt.figure(figsize=(12, 10))
        sns.heatmap(X_top.corr(), cmap="coolwarm", annot=False, square=True)
        plt.title("Correlation Heatmap (Top 15 High-Variance Features)")
        plt.savefig(f"{self.plots_dir}/correlation_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 3️⃣ PCA Visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        plt.figure(figsize=(8, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="coolwarm", s=10, alpha=0.7)
        plt.title("PCA Scatter Plot (Binary)")
        plt.savefig(f"{self.plots_dir}/pca_plot.png", dpi=300, bbox_inches='tight')
        plt.close()

        print("   ✓ Saved all visualizations in /plots folder.")

    # ============================================================
    # RUN FULL PIPELINE
    # ============================================================
    def run(self):
        self.load_all()
        self.clean()
        self.balance()
        self.scale()
        self.split()
        self.save()
        self.visualize()

        print("\nDONE ✔ Binary preprocessing + visualization complete.\n")


if __name__ == "__main__":
    BinaryPreprocessAndVisualize10K().run()
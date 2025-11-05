"""
data_explore.py
Exploratory Data Analysis (EDA) for Smart Detection of Phishing Attacks
in Encrypted Network Traffic

Dataset: TrafficLabelling/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
Goal: Understand traffic features, visualize distributions, check imbalance,
and prepare cleaned data for modeling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

def load_data(file_path):
    print(f"\nLoading dataset from: {file_path}")
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    print(f"✅ Dataset loaded successfully — shape: {df.shape}")
    return df

def basic_info(df):
    print("\n=== BASIC INFO ===")
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head())

def check_missing(df):
    print("\n=== MISSING VALUES ===")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("No missing values ✅")
    else:
        print(missing)

def analyze_labels(df):
    print("\n=== LABEL DISTRIBUTION ===")
    if 'Label' not in df.columns:
        print("⚠️ 'Label' column not found.")
        return
    print(df['Label'].value_counts())
    plt.figure(figsize=(6,4))
    sns.countplot(x='Label', data=df, palette='Set2', edgecolor='black')
    plt.title('Traffic Class Distribution')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('label_distribution.png', dpi=300)
    print("📊 Saved plot as label_distribution.png")

def numeric_summary(df):
    print("\n=== NUMERIC SUMMARY ===")
    num_cols = df.select_dtypes(include=np.number).columns
    print(df[num_cols].describe().T.head(10))

def prepare_data(df):
    print("\n=== PREPARING DATA ===")
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)
    df.dropna(subset=['Label'], inplace=True)
    df.to_csv('cleaned_data.csv', index=False)
    print("✅ Cleaned data saved as cleaned_data.csv")
    return df

def main():
    file_path = "TrafficLabelling/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    df = load_data(file_path)
    basic_info(df)
    check_missing(df)
    analyze_labels(df)
    numeric_summary(df)
    prepare_data(df)

if __name__ == "__main__":
    main()

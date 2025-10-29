import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
import os
import glob
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

def find_csv_files():
    """
    Search for CSV files in common locations
    """
    print("Searching for CSV files in current directory and subdirectories...")
    
    # Search patterns
    patterns = [
        '*.csv',
        '*/*.csv',
        '*/*/*.csv',
        'data/*.csv',
        'dataset/*.csv',
        'datasets/*.csv',
        'TrafficLabelling/*.csv'
    ]
    
    found_files = []
    for pattern in patterns:
        found_files.extend(glob.glob(pattern))
    
    # Remove duplicates
    found_files = list(set(found_files))
    
    if found_files:
        print(f"\n✓ Found {len(found_files)} CSV file(s):")
        for i, file in enumerate(found_files, 1):
            file_size = os.path.getsize(file) / (1024 * 1024)  # Size in MB
            print(f"  {i}. {file} ({file_size:.2f} MB)")
        return found_files
    else:
        print("\n❌ No CSV files found!")
        return []

def select_file():
    """
    Let user select a file or provide a path
    """
    files = find_csv_files()
    
    if not files:
        print("\n" + "="*70)
        print("PLEASE PROVIDE YOUR CSV FILE PATH")
        print("="*70)
        file_path = input("\nEnter the full path to your CSV file: ").strip()
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        return file_path
    
    if len(files) == 1:
        print(f"\n✓ Using file: {files[0]}")
        return files[0]
    
    print("\nWhich file would you like to analyze?")
    while True:
        try:
            choice = input(f"Enter number (1-{len(files)}) or press Enter for file #1: ").strip()
            if choice == "":
                return files[0]
            choice = int(choice)
            if 1 <= choice <= len(files):
                return files[choice - 1]
            else:
                print(f"Please enter a number between 1 and {len(files)}")
        except ValueError:
            print("Please enter a valid number")

def load_cic_ids_data(file_path, sample_size=None):
    """
    Load CIC-IDS2017 dataset with optional sampling
    """
    print("\nLoading dataset...")
    
    if sample_size:
        # Load in chunks and sample
        chunks = []
        for chunk in pd.read_csv(file_path, chunksize=50000):
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
        
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
            print(f"Sampled {sample_size} rows from dataset")
    else:
        df = pd.read_csv(file_path)
    
    # Clean column names (remove leading/trailing spaces)
    df.columns = df.columns.str.strip()
    
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def basic_info(df):
    """
    Display basic information about dataset
    """
    print("\n" + "="*70)
    print("BASIC DATASET INFORMATION")
    print("="*70)
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"\nFirst few column names:\n{df.columns.tolist()[:10]}")
    print(f"...\nLast few column names:\n{df.columns.tolist()[-5:]}")
    print(f"\nTotal columns: {len(df.columns)}")
    print(f"\nData Types:\n{df.dtypes.value_counts()}")
    print(f"\nMemory Usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
    
    return df.info()

def check_missing_values(df):
    """
    Check for missing values and infinite values
    """
    print("\n" + "="*70)
    print("MISSING & INFINITE VALUES ANALYSIS")
    print("="*70)
    
    # Check missing values
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    
    # Check infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = {}
    for col in numeric_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            inf_counts[col] = inf_count
    
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing_Count': missing.values,
        'Missing_%': missing_percent.values
    })
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
    
    if len(missing_df) > 0:
        print("\nMissing Values:")
        print(missing_df.head(10))
    else:
        print("\nNo missing values found!")
    
    if inf_counts:
        print("\nInfinite Values Found:")
        for col, count in sorted(inf_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {col}: {count} infinite values")
    else:
        print("\nNo infinite values found!")
    
    return missing_df

def analyze_target_distribution(df):
    """
    Analyze target variable distribution
    """
    print("\n" + "="*70)
    print("TARGET VARIABLE (LABEL) DISTRIBUTION")
    print("="*70)
    
    # Try to find label column (case-insensitive)
    label_col = None
    for col in df.columns:
        if col.lower() == 'label':
            label_col = col
            break
    
    if not label_col:
        print("ERROR: 'Label' column not found!")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    print(f"\nUnique Labels: {df[label_col].nunique()}")
    print(f"\nClass Distribution:")
    print(df[label_col].value_counts())
    print(f"\nClass Percentages:")
    class_pct = df[label_col].value_counts(normalize=True) * 100
    for label, pct in class_pct.items():
        print(f"  {label}: {pct:.2f}%")
    
    # Plot distribution
    plt.figure(figsize=(14, 6))
    counts = df[label_col].value_counts()
    
    # Use different colors for BENIGN vs attacks
    colors = ['green' if 'BENIGN' in str(label).upper() else 'red' for label in counts.index]
    
    counts.plot(kind='bar', color=colors, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Traffic Classes', fontsize=14, fontweight='bold')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for i, v in enumerate(counts):
        plt.text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n✓ Plot saved as 'class_distribution.png'")
    
    # Check class imbalance
    imbalance_ratio = counts.max() / counts.min()
    print(f"\nClass Imbalance Ratio: {imbalance_ratio:.2f}:1")
    if imbalance_ratio > 10:
        print("⚠️  WARNING: Significant class imbalance detected!")

def statistical_summary(df):
    """
    Generate statistical summary for key features
    """
    print("\n" + "="*70)
    print("STATISTICAL SUMMARY")
    print("="*70)
    
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"\nNumber of numeric features: {len(numeric_cols)}")
    
    # Key features for network traffic analysis
    key_features = [
        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Flow Bytes/s', 'Flow Packets/s', 'Packet Length Mean',
        'Average Packet Size', 'Fwd Packet Length Mean', 'Bwd Packet Length Mean'
    ]
    
    available_key_features = [f for f in key_features if f in df.columns]
    
    if available_key_features:
        print("\nKey Features Summary:")
        summary = df[available_key_features].describe()
        print(summary)
    else:
        print("\nGeneral Numeric Summary (first 10 features):")
        summary = df[numeric_cols[:10]].describe()
        print(summary)
    
    return summary

def plot_feature_distributions(df):
    """
    Plot distributions of key features, comparing BENIGN vs ATTACK traffic
    """
    print("\n" + "="*70)
    print("FEATURE DISTRIBUTIONS (BENIGN vs ATTACK)")
    print("="*70)
    
    # Find label column
    label_col = None
    for col in df.columns:
        if col.lower() == 'label':
            label_col = col
            break
    
    if not label_col:
        print("Label column not found, skipping distribution plots")
        return
    
    # Key features for DDoS detection
    key_features = [
        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Flow Bytes/s', 'Flow Packets/s', 'Packet Length Mean'
    ]
    
    available_features = [f for f in key_features if f in df.columns][:6]
    
    if not available_features:
        print("Key features not found, using first 6 numeric columns")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        available_features = numeric_cols[:6]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()
    
    # Create binary label for visualization
    df_plot = df.copy()
    df_plot['Traffic_Type'] = df_plot[label_col].apply(lambda x: 'BENIGN' if 'BENIGN' in str(x).upper() else 'ATTACK')
    
    for idx, col in enumerate(available_features):
        if idx < len(axes):
            # Get data and remove infinite/nan values
            benign_data = df_plot[df_plot['Traffic_Type'] == 'BENIGN'][col].replace([np.inf, -np.inf], np.nan).dropna()
            attack_data = df_plot[df_plot['Traffic_Type'] == 'ATTACK'][col].replace([np.inf, -np.inf], np.nan).dropna()
            
            # Skip if no valid data
            if len(benign_data) == 0 and len(attack_data) == 0:
                axes[idx].text(0.5, 0.5, f'No valid data\nfor {col}', 
                              ha='center', va='center', transform=axes[idx].transAxes)
                continue
            
            # Use log scale if values span multiple orders of magnitude
            if len(benign_data) > 0 and benign_data.max() > 0 and (benign_data.max() / (benign_data.min() + 1)) > 1000:
                benign_data = np.log10(benign_data + 1)
                attack_data = np.log10(attack_data + 1)
                col_label = f'log10({col})'
            else:
                col_label = col
            
            # Plot with error handling
            try:
                if len(benign_data) > 0:
                    axes[idx].hist(benign_data, bins=50, alpha=0.6, color='green', label='BENIGN', edgecolor='black')
                if len(attack_data) > 0:
                    axes[idx].hist(attack_data, bins=50, alpha=0.6, color='red', label='ATTACK', edgecolor='black')
            except Exception as e:
                axes[idx].text(0.5, 0.5, f'Error plotting\n{col}', 
                              ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_title(f'{col_label}', fontsize=10, fontweight='bold')
            axes[idx].set_xlabel('Value', fontsize=9)
            axes[idx].set_ylabel('Frequency', fontsize=9)
            axes[idx].legend(fontsize=8)
            axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Feature distribution plots saved as 'feature_distributions.png'")

def correlation_analysis(df):
    """
    Correlation analysis of key features
    """
    print("\n" + "="*70)
    print("CORRELATION ANALYSIS")
    print("="*70)
    
    # Select key features for correlation
    key_features = [
        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
        'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std',
        'Packet Length Mean', 'Average Packet Size', 'Fwd Packet Length Mean',
        'Bwd Packet Length Mean', 'Fwd Packets/s', 'Bwd Packets/s'
    ]
    
    available_features = [f for f in key_features if f in df.columns]
    
    if len(available_features) < 2:
        print("Not enough key features found, using first 15 numeric columns")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        available_features = numeric_cols[:15].tolist()
    
    # Replace infinite values for correlation
    df_corr = df[available_features].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    correlation_matrix = df_corr.corr()
    
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
    sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix (Lower Triangle)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Correlation matrix saved as 'correlation_matrix.png'")
    
    # Print highly correlated features
    print("\nHighly correlated feature pairs (|correlation| > 0.8):")
    high_corr_found = False
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > 0.8:
                print(f"  {correlation_matrix.columns[i]:<35} <-> {correlation_matrix.columns[j]:<35}: {correlation_matrix.iloc[i, j]:>6.3f}")
                high_corr_found = True
    
    if not high_corr_found:
        print("  No feature pairs with |correlation| > 0.8 found")

def analyze_attack_patterns(df):
    """
    Analyze patterns in attack traffic
    """
    print("\n" + "="*70)
    print("ATTACK PATTERN ANALYSIS")
    print("="*70)
    
    # Find label column
    label_col = None
    for col in df.columns:
        if col.lower() == 'label':
            label_col = col
            break
    
    if not label_col:
        print("Label column not found")
        return
    
    # Separate benign and attack traffic
    benign = df[df[label_col].str.upper().str.contains('BENIGN', na=False)]
    attack = df[~df[label_col].str.upper().str.contains('BENIGN', na=False)]
    
    print(f"\nBenign traffic samples: {len(benign):,}")
    print(f"Attack traffic samples: {len(attack):,}")
    
    # Compare key metrics
    comparison_features = [
        'Flow Duration', 'Flow Packets/s', 'Flow Bytes/s', 
        'Average Packet Size', 'Total Fwd Packets', 'Total Backward Packets'
    ]
    
    available = [f for f in comparison_features if f in df.columns]
    
    if available:
        print("\nKey Metrics Comparison (Mean values):")
        print(f"{'Feature':<30} {'BENIGN':>15} {'ATTACK':>15} {'Ratio':>10}")
        print("-" * 75)
        
        for feature in available:
            benign_mean = benign[feature].replace([np.inf, -np.inf], np.nan).mean()
            attack_mean = attack[feature].replace([np.inf, -np.inf], np.nan).mean()
            ratio = attack_mean / (benign_mean + 1e-10)
            print(f"{feature:<30} {benign_mean:>15.2f} {attack_mean:>15.2f} {ratio:>10.2f}x")

def prepare_for_modeling(df, output_file='processed_data.csv'):
    """
    Prepare data for machine learning modeling
    """
    print("\n" + "="*70)
    print("PREPARING DATA FOR MODELING")
    print("="*70)
    
    df_processed = df.copy()
    
    # Clean column names
    df_processed.columns = df_processed.columns.str.strip()
    
    # Handle infinite values
    print("Replacing infinite values...")
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    df_processed[numeric_cols] = df_processed[numeric_cols].replace([np.inf, -np.inf], np.nan)
    
    # Fill missing values
    print("Filling missing values with median...")
    for col in numeric_cols:
        if df_processed[col].isnull().sum() > 0:
            median_val = df_processed[col].median()
            df_processed[col].fillna(median_val, inplace=True)
    
    # Remove duplicate rows
    initial_rows = len(df_processed)
    df_processed.drop_duplicates(inplace=True)
    removed_duplicates = initial_rows - len(df_processed)
    if removed_duplicates > 0:
        print(f"Removed {removed_duplicates:,} duplicate rows")
    
    print(f"\nProcessed dataset shape: {df_processed.shape}")
    print(f"Missing values after processing: {df_processed.isnull().sum().sum()}")
    print(f"Infinite values after processing: {np.isinf(df_processed[numeric_cols]).sum().sum()}")
    
    # Save processed data
    df_processed.to_csv(output_file, index=False)
    print(f"\n✓ Processed data saved as '{output_file}'")
    
    return df_processed

def main():
    """
    Main function to run complete EDA
    """
    print("="*70)
    print("CIC-IDS2017 DATASET - EXPLORATORY DATA ANALYSIS")
    print("="*70)
    
    try:
        # Find and select CSV file
        file_path = select_file()
        
        print("\n" + "="*70)
        print(f"Analyzing: {os.path.basename(file_path)}")
        print("="*70)
        
        # Option to sample large datasets (set to None to load all data)
        sample_size = None  # Change to 100000 for faster processing
        
        # Load data
        df = load_cic_ids_data(file_path, sample_size=sample_size)
        
        # Run analysis pipeline
        basic_info(df)
        check_missing_values(df)
        analyze_target_distribution(df)
        statistical_summary(df)
        analyze_attack_patterns(df)
        plot_feature_distributions(df)
        correlation_analysis(df)
        
        # Prepare for modeling
        df_processed = prepare_for_modeling(df)
        
        print("\n" + "="*70)
        print("✓ DATA EXPLORATION COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nGenerated files:")
        print("  1. class_distribution.png - Attack vs Benign distribution")
        print("  2. feature_distributions.png - Key feature comparisons")
        print("  3. correlation_matrix.png - Feature correlations")
        print("  4. processed_data.csv - Cleaned dataset ready for modeling")
        print("\nNext steps:")
        print("  → Review the generated visualizations")
        print("  → Use processed_data.csv for machine learning models")
        print("  → Consider feature engineering based on correlation analysis")
        
        return df_processed
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    df = main()
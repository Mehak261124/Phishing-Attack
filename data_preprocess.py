import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class CICIDSPreprocessor:
    def __init__(self, file_path, n_samples=10000, samples_per_class=5000):
        """
        Simplified preprocessor for CIC-IDS2017 dataset
        
        Parameters:
        -----------
        file_path : str
            Path to the CSV file
        n_samples : int
            Total number of samples to extract (default: 10000)
        samples_per_class : int
            Samples per class for binary classification (default: 5000)
        """
        self.file_path = file_path
        self.n_samples = n_samples
        self.samples_per_class = samples_per_class
        self.scaler = StandardScaler()
        
    def load_and_sample_data(self):
        """
        Load data and extract balanced samples
        """
        print("="*70)
        print("LOADING AND SAMPLING DATA")
        print("="*70)
        
        df = pd.read_csv(self.file_path)
        print(f"Original dataset shape: {df.shape}")
        
        # Identify label column
        label_column = df.columns[-1]
        print(f"Label column: '{label_column}'")
        print(f"Unique labels: {df[label_column].unique()}")
        
        # Create binary labels: BENIGN (0) vs ATTACK (1)
        df['Binary_Label'] = df[label_column].apply(
            lambda x: 0 if x == 'BENIGN' else 1
        )
        
        print(f"\nClass distribution in original data:")
        print(df['Binary_Label'].value_counts())
        
        # Sample balanced data
        benign_samples = df[df['Binary_Label'] == 0].sample(
            n=min(self.samples_per_class, len(df[df['Binary_Label'] == 0])),
            random_state=42
        )
        attack_samples = df[df['Binary_Label'] == 1].sample(
            n=min(self.samples_per_class, len(df[df['Binary_Label'] == 1])),
            random_state=42
        )
        
        # Combine and shuffle
        self.df = pd.concat([benign_samples, attack_samples], axis=0)
        self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"\nSampled dataset shape: {self.df.shape}")
        print(f"Class distribution after sampling:")
        print(self.df['Binary_Label'].value_counts())
        
        return self.df
    
    def clean_and_prepare(self):
        """
        Clean data and prepare features
        """
        print("\n" + "="*70)
        print("DATA CLEANING AND PREPARATION")
        print("="*70)
        
        # Clean column names
        self.df.columns = self.df.columns.str.strip().str.replace(' ', '_')
        print("✓ Cleaned column names")
        
        # Separate features and labels
        y = self.df['Binary_Label']
        X = self.df.drop(columns=['Binary_Label'])
        
        # Remove non-numeric columns
        non_numeric_cols = []
        for col in X.columns:
            if X[col].dtype == 'object' or col in ['Flow_ID', 'Source_IP', 'Source_Port', 
                                                     'Destination_IP', 'Destination_Port', 
                                                     'Protocol', 'Timestamp', 'Label']:
                non_numeric_cols.append(col)
        
        if non_numeric_cols:
            print(f"✓ Removing {len(non_numeric_cols)} non-numeric columns")
            X = X.drop(columns=non_numeric_cols)
        
        # Handle missing values and infinities
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Drop rows with missing values
        valid_mask = ~X.isnull().any(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]
        
        missing_rows = len(self.df) - len(X)
        if missing_rows > 0:
            print(f"✓ Removed {missing_rows} rows with missing/inf values")
        
        # Remove duplicates
        dup_mask = ~X.duplicated(keep='first')
        X = X[dup_mask]
        y = y[dup_mask]
        
        dup_rows = len(valid_mask) - len(X)
        if dup_rows > 0:
            print(f"✓ Removed {dup_rows} duplicate rows")
        
        print(f"\nFinal dataset shape: {X.shape}")
        print(f"Number of features: {X.shape[1]}")
        
        self.X = X
        self.y = y
        self.feature_names = X.columns.tolist()
        
        return self.X, self.y
    
    def normalize_features(self):
        """
        Apply standardization (Z-score normalization)
        """
        print("\n" + "="*70)
        print("FEATURE NORMALIZATION")
        print("="*70)
        
        # Standardization: (x - μ) / σ
        X_scaled = self.scaler.fit_transform(self.X)
        self.X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        print(f"✓ Applied Z-score normalization")
        print(f"  Features: {self.X_scaled.shape[1]}")
        print(f"  Mean: {X_scaled.mean():.6f}")
        print(f"  Std: {X_scaled.std():.6f}")
        
        return self.X_scaled, self.y
    
    def split_data(self, test_size=0.2, val_size=0.1):
        """
        Split data into train, validation, and test sets
        """
        print("\n" + "="*70)
        print("DATA SPLITTING")
        print("="*70)
        
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.X_scaled, self.y, 
            test_size=test_size, 
            random_state=42, 
            stratify=self.y
        )
        
        # Second split: train and val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, 
            test_size=val_ratio, 
            random_state=42, 
            stratify=y_temp
        )
        
        print(f"Training set:   {X_train.shape[0]} samples ({y_train.value_counts()[0]} BENIGN, {y_train.value_counts()[1]} ATTACK)")
        print(f"Validation set: {X_val.shape[0]} samples ({y_val.value_counts()[0]} BENIGN, {y_val.value_counts()[1]} ATTACK)")
        print(f"Test set:       {X_test.shape[0]} samples ({y_test.value_counts()[0]} BENIGN, {y_test.value_counts()[1]} ATTACK)")
        
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_data(self):
        """
        Get all preprocessed datasets
        """
        return {
            'X_train': self.X_train,
            'X_val': self.X_val,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_val': self.y_val,
            'y_test': self.y_test,
            'feature_names': self.feature_names
        }
    
    def save_preprocessed_data(self, output_dir='preprocessed_data'):
        """
        Save all preprocessed datasets
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save train set
        train_df = self.X_train.copy()
        train_df['Binary_Label'] = self.y_train.values
        train_df.to_csv(f'{output_dir}/train_data.csv', index=False)
        
        # Save validation set
        val_df = self.X_val.copy()
        val_df['Binary_Label'] = self.y_val.values
        val_df.to_csv(f'{output_dir}/val_data.csv', index=False)
        
        # Save test set
        test_df = self.X_test.copy()
        test_df['Binary_Label'] = self.y_test.values
        test_df.to_csv(f'{output_dir}/test_data.csv', index=False)
        
        # Save feature names
        with open(f'{output_dir}/feature_names.txt', 'w') as f:
            f.write("FEATURE NAMES\n")
            f.write("="*70 + "\n")
            for i, feat in enumerate(self.feature_names, 1):
                f.write(f"{i}. {feat}\n")
        
        print("\n" + "="*70)
        print("DATA SAVED")
        print("="*70)
        print(f"✓ Training data:   {output_dir}/train_data.csv")
        print(f"✓ Validation data: {output_dir}/val_data.csv")
        print(f"✓ Test data:       {output_dir}/test_data.csv")
        print(f"✓ Feature names:   {output_dir}/feature_names.txt")
    
    def run_pipeline(self, test_size=0.2, val_size=0.1):
        """
        Run the complete simplified preprocessing pipeline
        """
        print("\n" + "="*70)
        print("CIC-IDS2017 SIMPLIFIED PREPROCESSING PIPELINE")
        print("="*70 + "\n")
        
        # Step 1: Load and sample
        self.load_and_sample_data()
        
        # Step 2: Clean and prepare
        self.clean_and_prepare()
        
        # Step 3: Normalize
        self.normalize_features()
        
        # Step 4: Split data
        self.split_data(test_size=test_size, val_size=val_size)
        
        print("\n" + "="*70)
        print("PREPROCESSING COMPLETE!")
        print("="*70)
        print(f"Total features: {len(self.feature_names)}")
        print(f"Ready for model training!")
        
        return self.get_data()


# Example usage:
if __name__ == "__main__":
    # Initialize preprocessor
    file_path = "/Users/mehakjain/Desktop/Phishing Attack/TrafficLabelling/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    
    preprocessor = CICIDSPreprocessor(
        file_path=file_path,
        n_samples=10000,
        samples_per_class=5000
    )
    
    # Run the simplified pipeline
    data = preprocessor.run_pipeline(
        test_size=0.2,   # 20% for testing
        val_size=0.1     # 10% for validation
    )
    
    # Save preprocessed data
    preprocessor.save_preprocessed_data('preprocessed_data')
    
    # Access the data
    print(f"\nData shapes:")
    print(f"X_train: {data['X_train'].shape}")
    print(f"X_val: {data['X_val'].shape}")
    print(f"X_test: {data['X_test'].shape}")
    print(f"\nNumber of features: {len(data['feature_names'])}")
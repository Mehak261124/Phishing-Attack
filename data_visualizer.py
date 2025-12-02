import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class SimpleVisualizer:
    def __init__(self, file_path, preprocessed_dir='preprocessed_data'):
        """
        Simple visualizer showing only what we actually did in preprocessing
        
        Parameters:
        -----------
        file_path : str
            Path to original CSV file
        preprocessed_dir : str
            Directory containing preprocessed data
        """
        self.file_path = file_path
        self.preprocessed_dir = preprocessed_dir
        self.load_data()
        
    def load_data(self):
        """Load both original and preprocessed data"""
        print("="*70)
        print("LOADING DATA FOR VISUALIZATION")
        print("="*70)
        
        # Load original data
        print("\nLoading original data...")
        df_original = pd.read_csv(self.file_path)
        
        # Create binary labels
        label_col = df_original.columns[-1]
        df_original['Binary_Label'] = df_original[label_col].apply(
            lambda x: 0 if x == 'BENIGN' else 1
        )
        
        # Sample same amount (10k)
        benign = df_original[df_original['Binary_Label'] == 0].sample(5000, random_state=42)
        attack = df_original[df_original['Binary_Label'] == 1].sample(5000, random_state=42)
        self.df_before = pd.concat([benign, attack]).sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"✓ Original data: {self.df_before.shape}")
        
        # Load preprocessed data
        print("\nLoading preprocessed data...")
        train = pd.read_csv(f'{self.preprocessed_dir}/train_data.csv')
        val = pd.read_csv(f'{self.preprocessed_dir}/val_data.csv')
        test = pd.read_csv(f'{self.preprocessed_dir}/test_data.csv')
        self.df_after = pd.concat([train, val, test], axis=0).reset_index(drop=True)
        
        print(f"✓ Preprocessed data: {self.df_after.shape}")
        
    def plot_normalization_effect(self, save_path='plots/normalization_effect.png'):
        """
        Show the main effect: NORMALIZATION
        Compare feature scales before and after Z-score normalization
        """
        
        # Get numeric columns from before (raw data)
        before_numeric = self.df_before.select_dtypes(include=[np.number]).drop(columns=['Binary_Label'], errors='ignore')
        before_numeric = before_numeric.replace([np.inf, -np.inf], np.nan)
        
        # Get after data (normalized)
        after_numeric = self.df_after.drop(columns=['Binary_Label'])
        
        # Find common features
        common_features = list(set(before_numeric.columns) & set(after_numeric.columns))
        
        # Select features with valid variance (not all zeros, not all NaN)
        valid_features = []
        for feat in common_features:
            before_data = before_numeric[feat].dropna()
            if len(before_data) > 100 and before_data.std() > 0:  # Must have data and variance
                valid_features.append((feat, before_data.var()))
        
        # Sort by variance and take top 3
        valid_features.sort(key=lambda x: x[1], reverse=True)
        top_features = [f[0] for f in valid_features[:3]]
        
        print(f"\nTop 3 most variant features selected:")
        for i, (feat, var) in enumerate(valid_features[:3], 1):
            print(f"  {i}. {feat} (variance: {var:.2e})")
        
        if len(top_features) < 3:
            print(f"⚠️  Warning: Only found {len(top_features)} valid features!")
            return
        
        fig, axes = plt.subplots(3, 4, figsize=(18, 12))
        
        for idx, feature in enumerate(top_features):
            row = idx
            
            # Get BEFORE data (raw, cleaned)
            before_data = before_numeric[feature].dropna()
            before_indices = before_data.index
            y_before = self.df_before.loc[before_indices, 'Binary_Label']
            
            # BEFORE - Original Values
            axes[row, 0].hist(before_data, bins=40, color='coral', alpha=0.7, edgecolor='black')
            axes[row, 0].set_title(f'{feature[:35]}\nBEFORE (Raw Values)', fontsize=10, fontweight='bold')
            axes[row, 0].set_ylabel('Frequency', fontsize=9)
            axes[row, 0].tick_params(labelsize=8)
            axes[row, 0].grid(alpha=0.3)
            
            # Stats box
            axes[row, 0].text(0.98, 0.98, 
                             f'Mean: {before_data.mean():.1f}\nStd: {before_data.std():.1f}\nMin: {before_data.min():.1f}\nMax: {before_data.max():.1f}',
                             transform=axes[row, 0].transAxes, fontsize=8,
                             verticalalignment='top', horizontalalignment='right',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
            
            # AFTER - Normalized Values
            after_data = after_numeric[feature]
            y_after = self.df_after['Binary_Label']
            
            axes[row, 1].hist(after_data, bins=40, color='lightgreen', alpha=0.7, edgecolor='black')
            axes[row, 1].set_title(f'{feature[:35]}\nAFTER (Normalized)', fontsize=10, fontweight='bold')
            axes[row, 1].set_ylabel('Frequency', fontsize=9)
            axes[row, 1].tick_params(labelsize=8)
            axes[row, 1].grid(alpha=0.3)
            
            # Stats box
            axes[row, 1].text(0.98, 0.98, 
                             f'Mean: {after_data.mean():.4f}\nStd: {after_data.std():.4f}\nMin: {after_data.min():.2f}\nMax: {after_data.max():.2f}',
                             transform=axes[row, 1].transAxes, fontsize=8,
                             verticalalignment='top', horizontalalignment='right',
                             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))
            
            # Class Comparison BEFORE
            benign_before = before_data[y_before == 0]
            attack_before = before_data[y_before == 1]
            
            if len(benign_before) > 0 and len(attack_before) > 0:
                axes[row, 2].hist(benign_before, bins=30, alpha=0.6, label='BENIGN', color='green', density=True)
                axes[row, 2].hist(attack_before, bins=30, alpha=0.6, label='ATTACK', color='red', density=True)
                axes[row, 2].set_title(f'Class Separation\nBEFORE', fontsize=10, fontweight='bold')
                axes[row, 2].set_ylabel('Density', fontsize=9)
                axes[row, 2].tick_params(labelsize=8)
                axes[row, 2].legend(fontsize=8)
                axes[row, 2].grid(alpha=0.3)
            
            # Class Comparison AFTER
            benign_after = after_data[y_after == 0]
            attack_after = after_data[y_after == 1]
            
            if len(benign_after) > 0 and len(attack_after) > 0:
                axes[row, 3].hist(benign_after, bins=30, alpha=0.6, label='BENIGN', color='green', density=True)
                axes[row, 3].hist(attack_after, bins=30, alpha=0.6, label='ATTACK', color='red', density=True)
                axes[row, 3].set_title(f'Class Separation\nAFTER', fontsize=10, fontweight='bold')
                axes[row, 3].set_ylabel('Density', fontsize=9)
                axes[row, 3].tick_params(labelsize=8)
                axes[row, 3].legend(fontsize=8)
                axes[row, 3].grid(alpha=0.3)
        
        plt.suptitle('Normalization Effect: Before vs After (Top 3 Most Variant Features)', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        self._save_plot(save_path)
        plt.show()
        
        print("\n✓ Normalization visualization complete!")
        print(f"✓ Successfully plotted {len(top_features)} features")
        
    def plot_preprocessing_summary(self, save_path='plots/preprocessing_summary.png'):
        """
        Summary of what preprocessing actually did
        """
        
        # Get numeric data
        before_numeric = self.df_before.select_dtypes(include=[np.number]).drop(columns=['Binary_Label'], errors='ignore')
        before_numeric = before_numeric.replace([np.inf, -np.inf], np.nan)
        after_numeric = self.df_after.drop(columns=['Binary_Label'])
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Data Quality Improvement
        metrics = {
            'Missing\nValues': [before_numeric.isnull().sum().sum(), after_numeric.isnull().sum().sum()],
            'Infinite\nValues': [np.isinf(before_numeric).sum().sum(), np.isinf(after_numeric).sum().sum()],
            'Duplicate\nRows': [before_numeric.duplicated().sum(), after_numeric.duplicated().sum()],
            'Total\nSamples': [len(before_numeric), len(after_numeric)]
        }
        
        x = np.arange(len(metrics))
        width = 0.35
        
        before_vals = [metrics[k][0] for k in metrics.keys()]
        after_vals = [metrics[k][1] for k in metrics.keys()]
        
        bars1 = axes[0, 0].bar(x - width/2, before_vals, width, label='Before', color='coral', alpha=0.7)
        bars2 = axes[0, 0].bar(x + width/2, after_vals, width, label='After', color='lightgreen', alpha=0.7)
        
        axes[0, 0].set_ylabel('Count', fontsize=10)
        axes[0, 0].set_title('Data Quality Metrics', fontsize=12, fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(metrics.keys(), fontsize=9)
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                              f'{int(height)}', ha='center', va='bottom', fontsize=8)
        
        # 2. Feature Scale Normalization
        common_features = list(set(before_numeric.columns) & set(after_numeric.columns))[:10]
        
        before_means = [before_numeric[f].mean() for f in common_features]
        after_means = [after_numeric[f].mean() for f in common_features]
        
        y_pos = np.arange(len(common_features))
        axes[0, 1].barh(y_pos - width/2, before_means, width, label='Before (Raw)', color='coral', alpha=0.7)
        axes[0, 1].barh(y_pos + width/2, after_means, width, label='After (Normalized)', color='lightgreen', alpha=0.7)
        
        axes[0, 1].set_yticks(y_pos)
        axes[0, 1].set_yticklabels([f[:20] for f in common_features], fontsize=8)
        axes[0, 1].set_xlabel('Mean Value', fontsize=10)
        axes[0, 1].set_title('Feature Means: Before vs After', fontsize=12, fontweight='bold')
        axes[0, 1].legend(fontsize=9)
        axes[0, 1].grid(axis='x', alpha=0.3)
        
        # 3. Standard Deviation Comparison
        before_stds = [before_numeric[f].std() for f in common_features]
        after_stds = [after_numeric[f].std() for f in common_features]
        
        axes[1, 0].barh(y_pos - width/2, before_stds, width, label='Before (Raw)', color='coral', alpha=0.7)
        axes[1, 0].barh(y_pos + width/2, after_stds, width, label='After (Normalized)', color='lightgreen', alpha=0.7)
        
        axes[1, 0].set_yticks(y_pos)
        axes[1, 0].set_yticklabels([f[:20] for f in common_features], fontsize=8)
        axes[1, 0].set_xlabel('Standard Deviation', fontsize=10)
        axes[1, 0].set_title('Feature Std Dev: Before vs After', fontsize=12, fontweight='bold')
        axes[1, 0].legend(fontsize=9)
        axes[1, 0].grid(axis='x', alpha=0.3)
        
        # 4. Summary Text
        axes[1, 1].axis('off')
        
        summary_text = f"""
PREPROCESSING STEPS APPLIED
{'='*45}

1. SAMPLING
   • Extracted 10,000 balanced samples
   • 5,000 BENIGN + 5,000 ATTACK
   
2. DATA CLEANING
   • Removed non-numeric columns
   • Cleaned missing/infinite values
   • Removed duplicates
   • Samples: {len(before_numeric)} → {len(after_numeric)}
   
3. NORMALIZATION (Z-score)
   • Applied: (x - μ) / σ
   • All features now have:
     - Mean ≈ 0.0000
     - Std Dev ≈ 1.0000
   
4. TRAIN/VAL/TEST SPLIT
   • Training: 70% ({len(pd.read_csv(f'{self.preprocessed_dir}/train_data.csv'))} samples)
   • Validation: 10% ({len(pd.read_csv(f'{self.preprocessed_dir}/val_data.csv'))} samples)
   • Test: 20% ({len(pd.read_csv(f'{self.preprocessed_dir}/test_data.csv'))} samples)
   
RESULT
{'='*45}
✓ Clean data with no quality issues
✓ All features on same scale
✓ Ready for ML model training
✓ Total features: {after_numeric.shape[1]}
        """
        
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                       fontsize=9, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.suptitle('Preprocessing Summary: What We Actually Did', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        self._save_plot(save_path)
        plt.show()
        
        print("\n✓ Preprocessing summary complete!")
        
    def plot_class_balance(self, save_path='plots/class_balance.png'):
        """
        Show class distribution across splits
        """
        
        # Load individual splits
        train = pd.read_csv(f'{self.preprocessed_dir}/train_data.csv')
        val = pd.read_csv(f'{self.preprocessed_dir}/val_data.csv')
        test = pd.read_csv(f'{self.preprocessed_dir}/test_data.csv')
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        datasets = [
            ('Training Set', train),
            ('Validation Set', val),
            ('Test Set', test)
        ]
        
        for idx, (name, df) in enumerate(datasets):
            counts = df['Binary_Label'].value_counts().sort_index()
            colors = ['#2ecc71', '#e74c3c']
            
            bars = axes[idx].bar(['BENIGN', 'ATTACK'], counts.values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
            axes[idx].set_title(f'{name}\n({len(df)} samples)', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Count', fontsize=11)
            axes[idx].grid(axis='y', alpha=0.3)
            axes[idx].set_ylim(0, max(counts.values) * 1.15)
            
            # Add count labels on bars
            for bar, count in zip(bars, counts.values):
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                              f'{count}\n({count/len(df)*100:.1f}%)',
                              ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.suptitle('Class Distribution After Preprocessing & Splitting', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save_plot(save_path)
        plt.show()
        
        print("\n✓ Class balance visualization complete!")
        
    def _save_plot(self, path):
        """Helper to save plot"""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {path}")
        
    def generate_all_visualizations(self):
        """Generate all honest visualizations"""
        print("\n" + "="*70)
        print("GENERATING PREPROCESSING VISUALIZATIONS")
        print("="*70 + "\n")
        
        print("1. Normalization Effect (Main preprocessing step)...")
        self.plot_normalization_effect()
        
        print("\n2. Preprocessing Summary...")
        self.plot_preprocessing_summary()
        
        print("\n3. Class Balance...")
        self.plot_class_balance()
        
        print("\n" + "="*70)
        print("ALL VISUALIZATIONS COMPLETE!")
        print("="*70)
        print("\nGenerated 3 plots showing:")
        print("  1. normalization_effect.png - How normalization changed feature scales")
        print("  2. preprocessing_summary.png - Overall preprocessing metrics")
        print("  3. class_balance.png - Class distribution in train/val/test sets")
        print("\nCheck the 'plots/' directory!")


# Example usage:
if __name__ == "__main__":
    # Initialize visualizer
    file_path = "TrafficLabelling/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    
    visualizer = SimpleVisualizer(
        file_path=file_path,
        preprocessed_dir='preprocessed_data'
    )
    
    # Generate all visualizations
    visualizer.generate_all_visualizations()
    
    # Or generate individual plots:
    # visualizer.plot_normalization_effect()
    # visualizer.plot_preprocessing_summary()
    # visualizer.plot_class_balance()
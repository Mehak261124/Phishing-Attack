"""
Support Vector Machine (SVM) Classifier for Binary Classification
BENIGN (0) vs ATTACK (1)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc)
import time
import pickle
import warnings
warnings.filterwarnings('ignore')

class SVMTrainer:
    def __init__(self, data_dir='preprocessed_data'):
        """Initialize SVM Trainer"""
        self.data_dir = data_dir
        self.model_name = 'Support Vector Machine (SVM)'
        self.load_data()
        
    def load_data(self):
        """Load preprocessed data"""
        print("="*70)
        print(f"TRAINING {self.model_name.upper()} FOR BINARY CLASSIFICATION")
        print("="*70)
        print("\nLoading preprocessed data...")
        
        # Load datasets
        train_df = pd.read_csv(f'{self.data_dir}/train_data.csv')
        val_df = pd.read_csv(f'{self.data_dir}/val_data.csv')
        test_df = pd.read_csv(f'{self.data_dir}/test_data.csv')
        
        # Separate features and labels
        self.X_train = train_df.drop(columns=['Binary_Label'])
        self.y_train = train_df['Binary_Label']
        
        self.X_val = val_df.drop(columns=['Binary_Label'])
        self.y_val = val_df['Binary_Label']
        
        self.X_test = test_df.drop(columns=['Binary_Label'])
        self.y_test = test_df['Binary_Label']
        
        print(f"✓ Training set: {self.X_train.shape}")
        print(f"✓ Validation set: {self.X_val.shape}")
        print(f"✓ Test set: {self.X_test.shape}")
        print(f"✓ Features: {self.X_train.shape[1]}")
        
    def train(self):
        """Train SVM model"""
        print("\n" + "="*70)
        print("TRAINING MODEL")
        print("="*70)
        
        # Initialize model
        self.model = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42,
            cache_size=1000,
            verbose=False
        )
        
        # Train
        print(f"\nTraining {self.model_name} with RBF kernel...")
        print("(This may take a few minutes...)")
        start_time = time.time()
        self.model.fit(self.X_train, self.y_train)
        self.train_time = time.time() - start_time
        
        print(f"✓ Training completed in {self.train_time:.2f}s")
        
        # Validate
        y_val_pred = self.model.predict(self.X_val)
        val_accuracy = accuracy_score(self.y_val, y_val_pred)
        val_f1 = f1_score(self.y_val, y_val_pred)
        
        print(f"\nValidation Results:")
        print(f"  Accuracy: {val_accuracy:.4f}")
        print(f"  F1-Score: {val_f1:.4f}")
        
    def evaluate(self):
        """Evaluate on test set"""
        print("\n" + "="*70)
        print("EVALUATING ON TEST SET")
        print("="*70)
        
        # Predict
        start_time = time.time()
        self.y_pred = self.model.predict(self.X_test)
        self.test_time = time.time() - start_time
        
        self.y_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        self.precision = precision_score(self.y_test, self.y_pred)
        self.recall = recall_score(self.y_test, self.y_pred)
        self.f1 = f1_score(self.y_test, self.y_pred)
        self.cm = confusion_matrix(self.y_test, self.y_pred)
        
        # Print results
        print(f"\nTest Set Performance:")
        print(f"  Accuracy:  {self.accuracy:.4f}")
        print(f"  Precision: {self.precision:.4f}")
        print(f"  Recall:    {self.recall:.4f}")
        print(f"  F1-Score:  {self.f1:.4f}")
        print(f"  Test Time: {self.test_time:.4f}s")
        
        print(f"\nConfusion Matrix:")
        print(f"  {self.cm}")
        
        print(f"\nDetailed Classification Report:")
        print(classification_report(self.y_test, self.y_pred, 
                                   target_names=['BENIGN', 'ATTACK']))
        
    def save_model(self, output_dir='trained_models'):
        """Save trained model"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = f"{output_dir}/svm.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"\n✓ Model saved: {filepath}")
        
    def plot_results(self):
        """Generate all visualizations"""
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)

        fig = plt.figure(figsize=(16, 14))

        # Grid: 4 rows × 3 columns
        gs = fig.add_gridspec(4, 3, hspace=0.5, wspace=0.3)

        # -------- 1. Confusion Matrix --------
        ax1 = fig.add_subplot(gs[0, 0])
        sns.heatmap(self.cm, annot=True, fmt='d', cmap='Oranges',
                    xticklabels=['BENIGN', 'ATTACK'],
                    yticklabels=['BENIGN', 'ATTACK'],
                    ax=ax1, cbar=False)
        ax1.set_title(f'Confusion Matrix\nAccuracy: {self.accuracy:.4f}', fontsize=11)
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')

        # -------- 2. Performance Metrics --------
        ax2 = fig.add_subplot(gs[0, 1])
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [self.accuracy, self.precision, self.recall, self.f1]
        colors = ['#e67e22', '#d35400', '#ca6f1e', '#ba4a00']
        bars = ax2.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_ylim(0, 1.1)
        ax2.set_title('Performance Metrics', fontsize=11)
        ax2.set_ylabel('Score')
        ax2.grid(axis='y', alpha=0.3)

        for bar, val in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                    f'{val:.4f}', ha='center', fontsize=9)

        # -------- 3. ROC Curve --------
        ax3 = fig.add_subplot(gs[0, 2])
        fpr, tpr, _ = roc_curve(self.y_test, self.y_proba)
        roc_auc = auc(fpr, tpr)
        ax3.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.4f}')
        ax3.plot([0, 1], [0, 1], linestyle='--')
        ax3.set_title('ROC Curve', fontsize=11)
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        ax3.grid(alpha=0.3)
        ax3.legend()

        # -------- 4. Support Vector Information (Row 2 Full Width) --------
        ax4 = fig.add_subplot(gs[1, :])
        ax4.axis('off')

        n_support = self.model.n_support_
        total_support = sum(n_support)

        info_text = f"""
    SUPPORT VECTOR MACHINE - MODEL INFORMATION
    ============================================================
    KERNEL:
        • Kernel: {self.model.kernel.upper()}
        • Gamma: {self.model.gamma}
        • C: {self.model.C}

    SUPPORT VECTORS:
        • BENIGN: {n_support[0]}
        • ATTACK: {n_support[1]}
        • Total: {total_support} ({total_support/len(self.X_train)*100:.2f}%)

    NOTES:
        • Only support vectors define the decision boundary
        • RBF kernel enables nonlinear separation
    """
        ax4.text(0.02, 0.98, info_text, va='top',
                fontfamily='monospace', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))

        # -------- 5. Training Summary (Row 3 Full Width) --------
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')

        summary_text = f"""
    DATASET SUMMARY
    ============================================================
    Train: {len(self.X_train)} | Val: {len(self.X_val)} | Test: {len(self.X_test)}
    Features: {self.X_train.shape[1]}

    PERFORMANCE:
        • Accuracy:  {self.accuracy:.4f}
        • Precision: {self.precision:.4f}
        • Recall:    {self.recall:.4f}
        • F1-Score:  {self.f1:.4f}
        • AUC:       {roc_auc:.4f}

    CONFUSION MATRIX:
        TN={self.cm[0,0]} | FP={self.cm[0,1]}
        FN={self.cm[1,0]} | TP={self.cm[1,1]}
    """
        ax5.text(0.02, 0.98, summary_text, va='top',
                fontfamily='monospace', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.6))

        # -------- 6. Footer Row (Row 4) --------
        ax6 = fig.add_subplot(gs[3, :])
        ax6.axis('off')
        ax6.text(0.5, 0.5,
                f"{self.model_name} — Plots Generated Successfully",
                ha='center', fontsize=12, fontweight='bold')

        # Save
        import os
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/svm_results_clean.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: plots/svm_results_clean.png")

        plt.show()
        
    def run_complete_pipeline(self):
        """Run complete training pipeline"""
        self.train()
        self.evaluate()
        self.save_model()
        self.plot_results()
        
        print("\n" + "="*70)
        print("SVM TRAINING COMPLETE!")
        print("="*70)
        print("\n✅ Model trained and evaluated successfully!")
        print(f"✅ Best metric - F1-Score: {self.f1:.4f}")


if __name__ == "__main__":
    # Train SVM
    trainer = SVMTrainer(data_dir='preprocessed_data')
    trainer.run_complete_pipeline()
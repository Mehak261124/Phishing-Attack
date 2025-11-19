"""
Random Forest Classifier for Binary Classification
BENIGN (0) vs ATTACK (1)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc)
import time
import pickle
import warnings
warnings.filterwarnings('ignore')

class RandomForestTrainer:
    def __init__(self, data_dir='preprocessed_data'):
        """Initialize Random Forest Trainer"""
        self.data_dir = data_dir
        self.model_name = 'Random Forest'
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
        """Train Random Forest model"""
        print("\n" + "="*70)
        print("TRAINING MODEL")
        print("="*70)
        
        # Initialize model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        # Train
        print(f"\nTraining {self.model_name} with 100 trees...")
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
        
        filepath = f"{output_dir}/random_forest.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"\n✓ Model saved: {filepath}")
        
    def plot_results(self):
        """Generate all visualizations"""
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Confusion Matrix
        ax1 = fig.add_subplot(gs[0, 0])
        sns.heatmap(self.cm, annot=True, fmt='d', cmap='Greens', 
                   xticklabels=['BENIGN', 'ATTACK'],
                   yticklabels=['BENIGN', 'ATTACK'],
                   ax=ax1, cbar=False)
        ax1.set_title(f'Confusion Matrix\nAccuracy: {self.accuracy:.4f}', 
                     fontweight='bold', fontsize=11)
        ax1.set_ylabel('True Label', fontweight='bold')
        ax1.set_xlabel('Predicted Label', fontweight='bold')
        
        # 2. Metrics Bar Chart
        ax2 = fig.add_subplot(gs[0, 1])
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [self.accuracy, self.precision, self.recall, self.f1]
        colors = ['#2ecc71', '#27ae60', '#16a085', '#1abc9c']
        bars = ax2.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_ylim(0, 1.1)
        ax2.set_title('Performance Metrics', fontweight='bold', fontsize=11)
        ax2.set_ylabel('Score', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, val + 0.02, 
                    f'{val:.4f}', ha='center', fontweight='bold', fontsize=9)
        
        # 3. ROC Curve
        ax3 = fig.add_subplot(gs[0, 2])
        fpr, tpr, _ = roc_curve(self.y_test, self.y_proba)
        roc_auc = auc(fpr, tpr)
        ax3.plot(fpr, tpr, color='darkgreen', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        ax3.set_xlim([0.0, 1.0])
        ax3.set_ylim([0.0, 1.05])
        ax3.set_xlabel('False Positive Rate', fontweight='bold')
        ax3.set_ylabel('True Positive Rate', fontweight='bold')
        ax3.set_title('ROC Curve', fontweight='bold', fontsize=11)
        ax3.legend(loc="lower right", fontsize=9)
        ax3.grid(alpha=0.3)
        
        # 4. Feature Importance (Top 15)
        ax4 = fig.add_subplot(gs[1, :])
        feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(15)
        
        colors_grad = plt.cm.Greens(np.linspace(0.4, 0.9, len(feature_importance)))
        bars = ax4.barh(range(len(feature_importance)), feature_importance['importance'], 
                       color=colors_grad, alpha=0.8, edgecolor='black')
        ax4.set_yticks(range(len(feature_importance)))
        ax4.set_yticklabels(feature_importance['feature'], fontsize=9)
        ax4.set_xlabel('Importance', fontweight='bold')
        ax4.set_title('Top 15 Most Important Features', fontweight='bold', fontsize=11)
        ax4.grid(axis='x', alpha=0.3)
        ax4.invert_yaxis()
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, feature_importance['importance'])):
            ax4.text(val + 0.001, i, f'{val:.4f}', 
                    va='center', fontsize=8)
        
        # 5. Training Summary Text
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        summary_text = f"""
MODEL: {self.model_name}
{'='*60}

DATASET:
  • Training samples: {len(self.X_train)} ({sum(self.y_train==0)} BENIGN, {sum(self.y_train==1)} ATTACK)
  • Validation samples: {len(self.X_val)} ({sum(self.y_val==0)} BENIGN, {sum(self.y_val==1)} ATTACK)
  • Test samples: {len(self.X_test)} ({sum(self.y_test==0)} BENIGN, {sum(self.y_test==1)} ATTACK)
  • Features: {self.X_train.shape[1]}

MODEL PARAMETERS:
  • Number of Trees: {self.model.n_estimators}
  • Max Depth: {self.model.max_depth}
  • Min Samples Split: {self.model.min_samples_split}
  • Min Samples Leaf: {self.model.min_samples_leaf}

PERFORMANCE:
  • Accuracy:  {self.accuracy:.4f} ({self.accuracy*100:.2f}%)
  • Precision: {self.precision:.4f} (How many predicted attacks were correct)
  • Recall:    {self.recall:.4f} (How many actual attacks were detected)
  • F1-Score:  {self.f1:.4f} (Harmonic mean of Precision & Recall)
  • ROC-AUC:   {roc_auc:.4f}

TIMING:
  • Training Time: {self.train_time:.2f} seconds
  • Test Time: {self.test_time:.4f} seconds
  • Prediction Speed: {len(self.X_test)/self.test_time:.0f} samples/sec

CONFUSION MATRIX:
  True Negatives (TN):  {self.cm[0,0]} (Correctly identified BENIGN)
  False Positives (FP): {self.cm[0,1]} (BENIGN misclassified as ATTACK)
  False Negatives (FN): {self.cm[1,0]} (ATTACK misclassified as BENIGN)
  True Positives (TP):  {self.cm[1,1]} (Correctly identified ATTACK)

INTERPRETATION:
  • Model correctly classified {self.accuracy*100:.2f}% of test samples
  • Out of {self.cm[0,1]+self.cm[1,1]} predicted attacks, {self.cm[1,1]} were correct ({self.precision*100:.2f}%)
  • Out of {self.cm[1,0]+self.cm[1,1]} actual attacks, {self.cm[1,1]} were detected ({self.recall*100:.2f}%)
  
ADVANTAGES OF RANDOM FOREST:
  • Ensemble of {self.model.n_estimators} decision trees reduces overfitting
  • More robust and generalizable than single decision tree
  • Handles high-dimensional data well
  • Provides reliable feature importance scores
        """
        
        ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        
        plt.suptitle(f'{self.model_name} - Binary Classification Results', 
                    fontsize=14, fontweight='bold')
        
        # Save plot
        import os
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/random_forest_results.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: plots/random_forest_results.png")
        plt.show()
        
    def run_complete_pipeline(self):
        """Run complete training pipeline"""
        self.train()
        self.evaluate()
        self.save_model()
        self.plot_results()
        
        print("\n" + "="*70)
        print("RANDOM FOREST TRAINING COMPLETE!")
        print("="*70)
        print("\n✅ Model trained and evaluated successfully!")
        print(f"✅ Best metric - F1-Score: {self.f1:.4f}")


if __name__ == "__main__":
    # Train Random Forest
    trainer = RandomForestTrainer(data_dir='preprocessed_data')
    trainer.run_complete_pipeline()
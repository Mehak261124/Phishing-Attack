import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc, roc_auc_score, precision_recall_curve)
from sklearn.feature_selection import SelectFromModel
import pickle
import time
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

class RandomForestDDoSDetector:
    """
    Enhanced Random Forest model for DDoS detection in encrypted traffic
    """
    
    def __init__(self, data_path='processed_data.csv', label_column='Label'):
        """
        Initialize the Random Forest detector
        """
        self.data_path = data_path
        self.label_column = label_column
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.selected_features = None
        self.results = {}
        
    def load_and_prepare_data(self, test_size=0.2, random_state=42, balance_data=False):
        """
        Load and prepare data for training with enhanced preprocessing
        """
        print("\n" + "="*70)
        print("LOADING AND PREPARING DATA")
        print("="*70)
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file '{self.data_path}' not found!")
        
        print(f"Loading data from: {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        print(f"Dataset shape: {df.shape}")
        
        # Find label column
        if self.label_column in df.columns:
            label_col = self.label_column
        else:
            print(f"Warning: '{self.label_column}' not found. Using last column.")
            label_col = df.columns[-1]
        
        print(f"Using label column: '{label_col}'")
        
        # Separate features and target
        X = df.drop(columns=[label_col])
        y = df[label_col]
        
        # Remove non-numeric columns from features
        non_numeric = X.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric) > 0:
            print(f"Removing {len(non_numeric)} non-numeric columns: {list(non_numeric)}")
            X = X.select_dtypes(include=[np.number])
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        print(f"Number of features: {len(self.feature_names)}")
        
        # Handle infinite and missing values
        print("Cleaning data...")
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Fill NaN with median
        for col in X.columns:
            if X[col].isnull().sum() > 0:
                X[col].fillna(X[col].median(), inplace=True)
        
        # Encode labels if they're strings
        if y.dtype == 'object':
            print(f"\nEncoding string labels...")
            unique_labels = y.unique()
            print(f"Unique classes: {unique_labels}")
            y_encoded = self.label_encoder.fit_transform(y)
            print(f"\nLabel mapping:")
            for orig, encoded in zip(self.label_encoder.classes_, 
                                    self.label_encoder.transform(self.label_encoder.classes_)):
                print(f"  {orig:20s} -> {encoded}")
            y = y_encoded
        
        # Convert to binary classification (BENIGN vs ATTACK)
        unique_classes = np.unique(y)
        if len(unique_classes) > 2:
            print(f"\nMulti-class detected ({len(unique_classes)} classes)")
            print("Converting to binary: 0=BENIGN, 1=ATTACK")
            # Assume first class (usually BENIGN) is 0, rest are attacks
            y = (y != 0).astype(int)
        
        # Display class distribution
        print(f"\nClass distribution:")
        unique, counts = np.unique(y, return_counts=True)
        for cls, count in zip(unique, counts):
            label_name = "BENIGN" if cls == 0 else "ATTACK"
            print(f"  {label_name} ({cls}): {count:,} ({count/len(y)*100:.2f}%)")
        
        # Balance data if requested
        if balance_data:
            X, y = self._balance_dataset(X, y)
        
        # Split data with stratification
        print(f"\nSplitting data (test_size={test_size})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Train set: {X_train.shape[0]:,} samples")
        print(f"  - BENIGN: {(y_train == 0).sum():,}")
        print(f"  - ATTACK: {(y_train == 1).sum():,}")
        print(f"Test set: {X_test.shape[0]:,} samples")
        print(f"  - BENIGN: {(y_test == 0).sum():,}")
        print(f"  - ATTACK: {(y_test == 1).sum():,}")
        
        # Scale features
        print(f"\nScaling features using StandardScaler...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame for feature selection
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.feature_names)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.feature_names)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def _balance_dataset(self, X, y):
        """
        Balance dataset using undersampling
        """
        print("\nBalancing dataset...")
        unique, counts = np.unique(y, return_counts=True)
        min_count = counts.min()
        
        balanced_indices = []
        for cls in unique:
            cls_indices = np.where(y == cls)[0]
            selected = np.random.choice(cls_indices, min_count, replace=False)
            balanced_indices.extend(selected)
        
        balanced_indices = np.array(balanced_indices)
        np.random.shuffle(balanced_indices)
        
        X_balanced = X.iloc[balanced_indices].reset_index(drop=True)
        y_balanced = y[balanced_indices]
        
        print(f"Balanced dataset: {len(y_balanced):,} samples per class")
        return X_balanced, y_balanced
    
    def feature_selection(self, X_train, y_train, threshold='median'):
        """
        Perform feature selection using Random Forest feature importance
        """
        print("\n" + "="*70)
        print("FEATURE SELECTION")
        print("="*70)
        
        print("Training preliminary Random Forest for feature importance...")
        rf_temp = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        rf_temp.fit(X_train, y_train)
        
        # Get feature importance
        importances = rf_temp.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print(f"\nTop 20 Most Important Features:")
        print(f"{'Rank':<6} {'Feature':<40} {'Importance':<12}")
        print("-" * 60)
        for i in range(min(20, len(indices))):
            idx = indices[i]
            print(f"{i+1:<6} {self.feature_names[idx]:<40} {importances[idx]:<12.6f}")
        
        # Select features
        selector = SelectFromModel(rf_temp, threshold=threshold, prefit=True)
        X_train_selected = selector.transform(X_train)
        
        # Get selected feature names
        selected_mask = selector.get_support()
        self.selected_features = [f for f, s in zip(self.feature_names, selected_mask) if s]
        
        print(f"\n✓ Selected {len(self.selected_features)} features out of {len(self.feature_names)}")
        print(f"  Reduction: {(1 - len(self.selected_features)/len(self.feature_names))*100:.1f}%")
        
        # Plot feature importance
        self._plot_feature_importance(importances, indices)
        
        return selector
    
    def train_random_forest(self, X_train, y_train, X_test, y_test, 
                           hyperparameter_tuning=False, n_estimators=100, max_depth=20):
        """
        Train Random Forest model with optional hyperparameter tuning
        """
        print("\n" + "="*70)
        print("TRAINING RANDOM FOREST MODEL")
        print("="*70)
        
        if hyperparameter_tuning:
            print("\nPerforming hyperparameter tuning (this may take a while)...")
            self.model = self._tune_hyperparameters(X_train, y_train)
        else:
            print(f"\nTraining with default parameters:")
            print(f"  - n_estimators: {n_estimators}")
            print(f"  - max_depth: {max_depth}")
            print(f"  - min_samples_split: 5")
            print(f"  - min_samples_leaf: 2")
            
            start_time = time.time()
            
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1,
                class_weight='balanced',  # Handle class imbalance
                verbose=0
            )
            
            print("\nTraining model...")
            self.model.fit(X_train, y_train)
            training_time = time.time() - start_time
            print(f"✓ Training completed in {training_time:.2f} seconds")
        
        # Evaluate model
        self.evaluate_model(X_test, y_test)
        
        return self.model
    
    def _tune_hyperparameters(self, X_train, y_train):
        """
        Perform hyperparameter tuning using GridSearchCV
        """
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
        
        grid_search = GridSearchCV(
            rf, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=2
        )
        
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        tuning_time = time.time() - start_time
        
        print(f"\n✓ Hyperparameter tuning completed in {tuning_time:.2f} seconds")
        print(f"\nBest parameters:")
        for param, value in grid_search.best_params_.items():
            print(f"  {param}: {value}")
        print(f"\nBest cross-validation F1-score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def evaluate_model(self, X_test, y_test):
        """
        Comprehensive model evaluation
        """
        print("\n" + "="*70)
        print("MODEL EVALUATION")
        print("="*70)
        
        print("\nMaking predictions...")
        start_time = time.time()
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        inference_time = time.time() - start_time
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Store results
        self.results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'inference_time': inference_time,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        # Print metrics
        print(f"\n{'='*50}")
        print("PERFORMANCE METRICS")
        print(f"{'='*50}")
        print(f"Accuracy:      {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision:     {precision:.4f}")
        print(f"Recall:        {recall:.4f}")
        print(f"F1-Score:      {f1:.4f}")
        print(f"AUC-ROC:       {roc_auc:.4f}")
        print(f"Inference Time: {inference_time:.4f} seconds")
        print(f"Avg per sample: {inference_time/len(y_test)*1000:.2f} ms")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        print(f"\n{'='*50}")
        print("CONFUSION MATRIX")
        print(f"{'='*50}")
        print(f"                    Predicted")
        print(f"                BENIGN    ATTACK")
        print(f"Actual BENIGN   {tn:6d}    {fp:6d}")
        print(f"       ATTACK   {fn:6d}    {tp:6d}")
        
        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        print(f"\n{'='*50}")
        print("DETAILED METRICS")
        print(f"{'='*50}")
        print(f"True Positives:  {tp:6d}  (Correctly detected attacks)")
        print(f"True Negatives:  {tn:6d}  (Correctly identified benign)")
        print(f"False Positives: {fp:6d}  (Benign classified as attack)")
        print(f"False Negatives: {fn:6d}  (Attacks missed)")
        print(f"\nSpecificity:     {specificity:.4f}")
        print(f"False Positive Rate: {fpr:.4f}")
        print(f"False Negative Rate: {fnr:.4f}")
        
        # Classification report
        print(f"\n{'='*50}")
        print("CLASSIFICATION REPORT")
        print(f"{'='*50}")
        print(classification_report(y_test, y_pred, 
                                   target_names=['BENIGN', 'ATTACK'],
                                   digits=4))
        
        # Generate visualizations
        self._plot_confusion_matrix(cm)
        self._plot_roc_curve()
        self._plot_precision_recall_curve()
        
        return self.results
    
    def _plot_confusion_matrix(self, cm):
        """
        Plot and save confusion matrix with percentages
        """
        plt.figure(figsize=(10, 8))
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create annotations with both count and percentage
        annotations = np.array([[f'{count:,}\n({pct:.1f}%)' 
                               for count, pct in zip(row_count, row_pct)]
                              for row_count, row_pct in zip(cm, cm_percent)])
        
        sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', 
                    cbar_kws={'label': 'Count'},
                    xticklabels=['BENIGN', 'ATTACK'],
                    yticklabels=['BENIGN', 'ATTACK'],
                    linewidths=2, linecolor='black')
        
        plt.title('Random Forest - Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=13, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=13, fontweight='bold')
        plt.tight_layout()
        
        plt.savefig('RF_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("\n✓ Confusion matrix saved: RF_confusion_matrix.png")
    
    def _plot_roc_curve(self):
        """
        Plot ROC curve
        """
        fpr, tpr, thresholds = roc_curve(self.results['y_test'], 
                                         self.results['y_pred_proba'])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='#2ecc71', lw=3, 
                label=f'Random Forest (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.5)')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=13, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=13, fontweight='bold')
        plt.title('Random Forest - ROC Curve', fontsize=16, fontweight='bold', pad=20)
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        plt.savefig('RF_roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ ROC curve saved: RF_roc_curve.png")
    
    def _plot_precision_recall_curve(self):
        """
        Plot Precision-Recall curve
        """
        precision, recall, thresholds = precision_recall_curve(
            self.results['y_test'], self.results['y_pred_proba']
        )
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='#e74c3c', lw=3, 
                label=f'Random Forest (F1 = {self.results["f1_score"]:.4f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=13, fontweight='bold')
        plt.ylabel('Precision', fontsize=13, fontweight='bold')
        plt.title('Random Forest - Precision-Recall Curve', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.legend(loc="lower left", fontsize=12)
        plt.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        plt.savefig('RF_precision_recall_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Precision-Recall curve saved: RF_precision_recall_curve.png")
    
    def _plot_feature_importance(self, importances, indices, top_n=25):
        """
        Plot feature importance
        """
        plt.figure(figsize=(12, 10))
        
        top_indices = indices[:top_n]
        top_importances = importances[top_indices]
        top_features = [self.feature_names[i] for i in top_indices]
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
        
        plt.barh(range(top_n), top_importances, color=colors, edgecolor='black', linewidth=1.2)
        plt.yticks(range(top_n), top_features, fontsize=10)
        plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
        plt.title(f'Top {top_n} Feature Importances - Random Forest', 
                 fontsize=14, fontweight='bold', pad=15)
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        plt.savefig('RF_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("\n✓ Feature importance plot saved: RF_feature_importance.png")
    
    def save_model(self, model_dir='models'):
        """
        Save trained model and preprocessing objects
        """
        print("\n" + "="*70)
        print("SAVING MODEL")
        print("="*70)
        
        os.makedirs(model_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = os.path.join(model_dir, 'random_forest_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✓ Model saved: {model_path}")
        
        # Save scaler
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✓ Scaler saved: {scaler_path}")
        
        # Save label encoder
        if len(self.label_encoder.classes_) > 0:
            encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.label_encoder, f)
            print(f"✓ Label encoder saved: {encoder_path}")
        
        # Save feature names
        features_path = os.path.join(model_dir, 'feature_names.pkl')
        with open(features_path, 'wb') as f:
            pickle.dump(self.feature_names, f)
        print(f"✓ Feature names saved: {features_path}")
        
        # Save results
        results_path = os.path.join(model_dir, 'model_results.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(self.results, f)
        print(f"✓ Results saved: {results_path}")
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'model_type': 'RandomForestClassifier',
            'n_features': len(self.feature_names),
            'metrics': {
                'accuracy': self.results['accuracy'],
                'precision': self.results['precision'],
                'recall': self.results['recall'],
                'f1_score': self.results['f1_score'],
                'roc_auc': self.results['roc_auc']
            }
        }
        
        metadata_path = os.path.join(model_dir, 'model_metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"✓ Metadata saved: {metadata_path}")
    
    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation
        """
        print("\n" + "="*70)
        print("CROSS-VALIDATION")
        print("="*70)
        
        print(f"Performing {cv}-fold cross-validation...")
        
        scores = cross_val_score(self.model, X, y, cv=cv, 
                                scoring='f1', n_jobs=-1, verbose=1)
        
        print(f"\n{'='*50}")
        print("CROSS-VALIDATION RESULTS")
        print(f"{'='*50}")
        print(f"F1-Scores per fold: {scores}")
        print(f"Mean F1-Score: {scores.mean():.4f}")
        print(f"Std Dev:       {scores.std():.4f}")
        print(f"Min F1-Score:  {scores.min():.4f}")
        print(f"Max F1-Score:  {scores.max():.4f}")
        
        return scores

def main():
    """
    Main training pipeline
    """
    print("="*70)
    print("DDOS DETECTION USING RANDOM FOREST")
    print("CIC-IDS2017 Dataset")
    print("="*70)
    
    # Configuration
    config = {
        'data_path': 'processed_data.csv',
        'test_size': 0.2,
        'n_estimators': 200,
        'max_depth': 30,
        'balance_data': False,  # Set to True if severe class imbalance
        'feature_selection': True,
        'hyperparameter_tuning': False,  # Set to True for GridSearchCV (slow)
        'cross_validation': False  # Set to True for CV
    }
    
    try:
        # Initialize detector
        detector = RandomForestDDoSDetector(data_path=config['data_path'])
        
        # Load and prepare data
        X_train, X_test, y_train, y_test = detector.load_and_prepare_data(
            test_size=config['test_size'],
            balance_data=config['balance_data']
        )
        
        # Feature selection (optional but recommended)
        if config['feature_selection']:
            selector = detector.feature_selection(X_train, y_train, threshold='median')
            X_train = selector.transform(X_train)
            X_test = selector.transform(X_test)
            print(f"\nUsing {X_train.shape[1]} selected features for training")
        
        # Train model
        detector.train_random_forest(
            X_train, y_train, X_test, y_test,
            hyperparameter_tuning=config['hyperparameter_tuning'],
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth']
        )
        
        # Cross-validation (optional)
        if config['cross_validation']:
            X_full = np.vstack([X_train, X_test])
            y_full = np.hstack([y_train, y_test])
            detector.cross_validate(X_full, y_full, cv=5)
        
        # Save model
        detector.save_model()
        
        print("\n" + "="*70)
        print("✓ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\n📊 Generated Visualizations:")
        print("   ├─ RF_confusion_matrix.png")
        print("   ├─ RF_roc_curve.png")
        print("   ├─ RF_precision_recall_curve.png")
        print("   └─ RF_feature_importance.png")
        print("\n🤖 Saved Models (in 'models/' directory):")
        print("   ├─ random_forest_model.pkl")
        print("   ├─ scaler.pkl")
        print("   ├─ label_encoder.pkl")
        print("   ├─ feature_names.pkl")
        print("   ├─ model_results.pkl")
        print("   └─ model_metadata.pkl")
        print("\n🎯 Model Performance Summary:")
        print(f"   ├─ Accuracy:  {detector.results['accuracy']:.4f}")
        print(f"   ├─ Precision: {detector.results['precision']:.4f}")
        print(f"   ├─ Recall:    {detector.results['recall']:.4f}")
        print(f"   ├─ F1-Score:  {detector.results['f1_score']:.4f}")
        print(f"   └─ AUC-ROC:   {detector.results['roc_auc']:.4f}")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\n💡 Solution: Run 'data_explore.py' first to generate 'processed_data.csv'")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n💡 Check the error message above and verify your data format")

if __name__ == "__main__":
    main()
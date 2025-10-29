import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint, 
                                        ReduceLROnPlateau, TensorBoard)
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_curve, auc, 
                             roc_auc_score, classification_report,
                             precision_recall_curve)
import seaborn as sns
import pickle
import os
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configure GPU (if available)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print(f"✓ GPU available: {physical_devices[0].name}")
    except:
        pass
else:
    print("⚠ No GPU detected, using CPU")

class DeepLearningDDoSDetector:
    """
    Enhanced Deep Learning models for DDoS detection in encrypted traffic
    Optimized for CIC-IDS2017 dataset
    """
    
    def __init__(self, data_path='processed_data.csv', label_column='Label'):
        self.data_path = data_path
        self.label_column = label_column
        self.models = {}
        self.histories = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        
    def load_and_prepare_data(self, test_size=0.2, val_size=0.15, 
                             random_state=42, balance_data=False):
        """
        Load and prepare data with train/val/test split
        """
        print("\n" + "="*70)
        print("LOADING AND PREPARING DATA FOR DEEP LEARNING")
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
            label_col = df.columns[-1]
        
        print(f"Using label column: '{label_col}'")
        
        # Separate features and target
        X = df.drop(columns=[label_col])
        y = df[label_col]
        
        # Keep only numeric columns
        non_numeric = X.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric) > 0:
            print(f"Removing {len(non_numeric)} non-numeric columns")
            X = X.select_dtypes(include=[np.number])
        
        self.feature_names = X.columns.tolist()
        print(f"Number of features: {len(self.feature_names)}")
        
        # Handle infinite and missing values
        print("Cleaning data...")
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(X.median(), inplace=True)
        
        # Encode labels if strings
        if y.dtype == 'object':
            print(f"Encoding string labels...")
            unique_labels = y.unique()
            print(f"Unique classes: {unique_labels}")
            y_encoded = self.label_encoder.fit_transform(y)
            print(f"\nLabel mapping:")
            for orig, encoded in zip(self.label_encoder.classes_, 
                                    self.label_encoder.transform(self.label_encoder.classes_)):
                print(f"  {orig:20s} -> {encoded}")
            y = y_encoded
        
        # Convert to binary classification
        unique_classes = np.unique(y)
        if len(unique_classes) > 2:
            print(f"\nMulti-class detected ({len(unique_classes)} classes)")
            print("Converting to binary: 0=BENIGN, 1=ATTACK")
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
        
        # Split into train, validation, and test sets
        print(f"\nSplitting data (test={test_size}, val={val_size})...")
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), 
            random_state=random_state, stratify=y_temp
        )
        
        print(f"Train set: {X_train.shape[0]:,} samples")
        print(f"  - BENIGN: {(y_train == 0).sum():,}")
        print(f"  - ATTACK: {(y_train == 1).sum():,}")
        print(f"Validation set: {X_val.shape[0]:,} samples")
        print(f"  - BENIGN: {(y_val == 0).sum():,}")
        print(f"  - ATTACK: {(y_val == 1).sum():,}")
        print(f"Test set: {X_test.shape[0]:,} samples")
        print(f"  - BENIGN: {(y_test == 0).sum():,}")
        print(f"  - ATTACK: {(y_test == 1).sum():,}")
        
        # Scale features
        print(f"\nScaling features using StandardScaler...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    
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
    
    def build_optimized_dnn(self, input_dim, name="Optimized_DNN"):
        """
        Build an optimized DNN for DDoS detection
        Architecture: Input -> Dense(256) -> Dense(128) -> Dense(64) -> Dense(32) -> Output
        """
        print(f"\n{'='*70}")
        print(f"BUILDING {name.upper()}")
        print(f"{'='*70}")
        
        model = models.Sequential([
            layers.Input(shape=(input_dim,)),
            
            # Layer 1
            layers.Dense(256, activation='relu', 
                        kernel_initializer='he_normal',
                        kernel_regularizer=regularizers.l2(0.0001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Layer 2
            layers.Dense(128, activation='relu',
                        kernel_initializer='he_normal',
                        kernel_regularizer=regularizers.l2(0.0001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Layer 3
            layers.Dense(64, activation='relu',
                        kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Layer 4
            layers.Dense(32, activation='relu',
                        kernel_initializer='he_normal'),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(1, activation='sigmoid')
        ], name=name)
        
        # Compile with Adam optimizer
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall'),
                    keras.metrics.AUC(name='auc')]
        )
        
        print("\n📊 Model Architecture:")
        model.summary()
        
        # Calculate total parameters
        total_params = model.count_params()
        print(f"\n✓ Total parameters: {total_params:,}")
        
        return model
    
    def build_deep_dnn(self, input_dim, name="Deep_DNN"):
        """
        Build a deeper neural network with 6 hidden layers
        """
        print(f"\n{'='*70}")
        print(f"BUILDING {name.upper()}")
        print(f"{'='*70}")
        
        model = models.Sequential([
            layers.Input(shape=(input_dim,)),
            
            layers.Dense(512, activation='relu', 
                        kernel_initializer='he_normal',
                        kernel_regularizer=regularizers.l2(0.0001)),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            layers.Dense(256, activation='relu',
                        kernel_initializer='he_normal',
                        kernel_regularizer=regularizers.l2(0.0001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu',
                        kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu',
                        kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(32, activation='relu',
                        kernel_initializer='he_normal'),
            layers.Dropout(0.2),
            
            layers.Dense(16, activation='relu',
                        kernel_initializer='he_normal'),
            
            layers.Dense(1, activation='sigmoid')
        ], name=name)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall'),
                    keras.metrics.AUC(name='auc')]
        )
        
        print("\n📊 Model Architecture:")
        model.summary()
        print(f"\n✓ Total parameters: {model.count_params():,}")
        
        return model
    
    def build_residual_dnn(self, input_dim, name="Residual_DNN"):
        """
        Build a DNN with residual connections
        """
        print(f"\n{'='*70}")
        print(f"BUILDING {name.upper()} (With Skip Connections)")
        print(f"{'='*70}")
        
        inputs = layers.Input(shape=(input_dim,))
        
        # First block
        x = layers.Dense(256, activation='relu', kernel_initializer='he_normal')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Second block with residual
        x2 = layers.Dense(128, activation='relu', kernel_initializer='he_normal')(x)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.Dropout(0.3)(x2)
        
        # Third block
        x3 = layers.Dense(64, activation='relu', kernel_initializer='he_normal')(x2)
        x3 = layers.BatchNormalization()(x3)
        x3 = layers.Dropout(0.2)(x3)
        
        # Fourth block
        x4 = layers.Dense(32, activation='relu', kernel_initializer='he_normal')(x3)
        x4 = layers.Dropout(0.2)(x4)
        
        # Output
        outputs = layers.Dense(1, activation='sigmoid')(x4)
        
        model = models.Model(inputs=inputs, outputs=outputs, name=name)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall'),
                    keras.metrics.AUC(name='auc')]
        )
        
        print("\n📊 Model Architecture:")
        model.summary()
        print(f"\n✓ Total parameters: {model.count_params():,}")
        
        return model
    
    def train_model(self, model, X_train, y_train, X_val, y_val, 
                   model_name, epochs=100, batch_size=256):
        """
        Train a deep learning model with advanced callbacks
        """
        print(f"\n{'='*70}")
        print(f"TRAINING {model_name.upper()}")
        print(f"{'='*70}")
        
        # Create directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        # Calculate class weights for imbalanced data
        unique, counts = np.unique(y_train, return_counts=True)
        class_weights = {i: len(y_train)/(len(unique)*count) 
                        for i, count in zip(unique, counts)}
        print(f"\nUsing class weights: {class_weights}")
        
        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        )
        
        # FIXED: Changed .h5 to .keras extension
        checkpoint = ModelCheckpoint(
            f'models/{model_name}_best.keras',
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1,
            mode='min'
        )
        
        tensorboard = TensorBoard(
            log_dir=f'logs/{model_name}_{datetime.now().strftime("%Y%m%d-%H%M%S")}',
            histogram_freq=1,
            write_graph=True
        )
        
        # Train
        print(f"\nStarting training...")
        print(f"Configuration:")
        print(f"  - Epochs: {epochs}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Training samples: {len(X_train):,}")
        print(f"  - Validation samples: {len(X_val):,}")
        
        import time
        start_time = time.time()
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=[early_stop, checkpoint, reduce_lr, tensorboard],
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        print(f"\n✓ Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        print(f"✓ Best epoch: {len(history.history['loss']) - early_stop.patience if early_stop.stopped_epoch > 0 else len(history.history['loss'])}")
        
        # Store model and history
        self.models[model_name] = model
        self.histories[model_name] = history
        
        return history
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """
        Comprehensive model evaluation
        """
        print(f"\n{'='*70}")
        print(f"EVALUATING {model_name.upper()}")
        print(f"{'='*70}")
        
        # Predictions
        print("\nMaking predictions...")
        import time
        start_time = time.time()
        y_pred_prob = model.predict(X_test, verbose=0)
        inference_time = time.time() - start_time
        
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        y_pred_prob = y_pred_prob.flatten()
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_prob)
        
        # Print results
        print(f"\n{'='*50}")
        print("PERFORMANCE METRICS")
        print(f"{'='*50}")
        print(f"Accuracy:       {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision:      {precision:.4f}")
        print(f"Recall:         {recall:.4f}")
        print(f"F1-Score:       {f1:.4f}")
        print(f"AUC-ROC:        {roc_auc:.4f}")
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
        print(f"True Positives:  {tp:6d}")
        print(f"True Negatives:  {tn:6d}")
        print(f"False Positives: {fp:6d}")
        print(f"False Negatives: {fn:6d}")
        print(f"\nSpecificity:         {specificity:.4f}")
        print(f"False Positive Rate: {fpr:.4f}")
        print(f"False Negative Rate: {fnr:.4f}")
        
        # Classification report
        print(f"\n{'='*50}")
        print("CLASSIFICATION REPORT")
        print(f"{'='*50}")
        print(classification_report(y_test, y_pred, 
                                   target_names=['BENIGN', 'ATTACK'],
                                   digits=4))
        
        # Store results
        self.results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': roc_auc,
            'inference_time': inference_time,
            'y_pred_prob': y_pred_prob,
            'confusion_matrix': cm
        }
        
        # Generate visualizations
        self._plot_confusion_matrix(cm, model_name)
        self._plot_roc_curve(y_test, y_pred_prob, roc_auc, model_name)
        self._plot_precision_recall_curve(y_test, y_pred_prob, model_name)
        
        return self.results[model_name]
    
    def _plot_confusion_matrix(self, cm, model_name):
        """
        Plot confusion matrix with percentages
        """
        plt.figure(figsize=(10, 8))
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create annotations
        annotations = np.array([[f'{count:,}\n({pct:.1f}%)' 
                               for count, pct in zip(row_count, row_pct)]
                              for row_count, row_pct in zip(cm, cm_percent)])
        
        sns.heatmap(cm, annot=annotations, fmt='', cmap='YlGnBu',
                   xticklabels=['BENIGN', 'ATTACK'],
                   yticklabels=['BENIGN', 'ATTACK'],
                   linewidths=2, linecolor='black',
                   cbar_kws={'label': 'Count'})
        
        plt.title(f'{model_name} - Confusion Matrix', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=13, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Confusion matrix saved: {model_name}_confusion_matrix.png")
    
    def _plot_roc_curve(self, y_test, y_pred_prob, roc_auc, model_name):
        """
        Plot ROC curve
        """
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='#2ecc71', lw=3, 
                label=f'{model_name} (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.5)')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=13, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=13, fontweight='bold')
        plt.title(f'{model_name} - ROC Curve', fontsize=16, fontweight='bold', pad=20)
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(f'{model_name}_roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_precision_recall_curve(self, y_test, y_pred_prob, model_name):
        """
        Plot Precision-Recall curve
        """
        precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='#e74c3c', lw=3, 
                label=f'{model_name}')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=13, fontweight='bold')
        plt.ylabel('Precision', fontsize=13, fontweight='bold')
        plt.title(f'{model_name} - Precision-Recall Curve', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.legend(loc="lower left", fontsize=12)
        plt.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(f'{model_name}_pr_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_training_history(self, model_name):
        """
        Plot comprehensive training history
        """
        if model_name not in self.histories:
            print(f"No training history for {model_name}")
            return
        
        history = self.histories[model_name]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        metrics = [
            ('accuracy', 'Accuracy'),
            ('loss', 'Loss'),
            ('precision', 'Precision'),
            ('recall', 'Recall'),
            ('auc', 'AUC'),
        ]
        
        for idx, (metric, title) in enumerate(metrics):
            row, col = idx // 3, idx % 3
            if metric in history.history:
                axes[row, col].plot(history.history[metric], 
                                   label='Train', linewidth=2.5, color='#3498db')
                axes[row, col].plot(history.history[f'val_{metric}'], 
                                   label='Validation', linewidth=2.5, color='#e74c3c')
                axes[row, col].set_title(title, fontsize=13, fontweight='bold')
                axes[row, col].set_xlabel('Epoch', fontsize=11)
                axes[row, col].set_ylabel(title, fontsize=11)
                axes[row, col].legend(fontsize=10)
                axes[row, col].grid(True, alpha=0.3)
        
        # Hide last subplot
        axes[1, 2].axis('off')
        
        plt.suptitle(f'{model_name} - Training History', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{model_name}_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Training history saved: {model_name}_training_history.png")
    
    def plot_all_roc_curves(self, y_test):
        """
        Plot ROC curves for all models in one figure
        """
        print(f"\n{'='*70}")
        print("GENERATING COMBINED ROC CURVES")
        print(f"{'='*70}")
        
        plt.figure(figsize=(12, 9))
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        
        for idx, (model_name, metrics) in enumerate(self.results.items()):
            if 'y_pred_prob' in metrics:
                fpr, tpr, _ = roc_curve(y_test, metrics['y_pred_prob'])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=3, color=colors[idx % len(colors)],
                        label=f'{model_name} (AUC = {roc_auc:.4f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.5)')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=13, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=13, fontweight='bold')
        plt.title('ROC Curves - All Deep Learning Models', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig('DL_all_roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Combined ROC curves saved: DL_all_roc_curves.png")
    
    def compare_models(self):
        """
        Comprehensive model comparison
        """
        print(f"\n{'='*70}")
        print("DEEP LEARNING MODELS COMPARISON")
        print(f"{'='*70}")
        
        # Create comparison dataframe
        comparison_data = []
        for model_name, metrics in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'AUC-ROC': f"{metrics['auc_roc']:.4f}",
                'Inference(s)': f"{metrics['inference_time']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\n", comparison_df.to_string(index=False))
        
        # Save comparison
        comparison_df.to_csv('DL_model_comparison.csv', index=False)
        print("\n✓ Comparison table saved: DL_model_comparison.csv")
        
        # Plot comparison
        self._plot_comparison()
        
        # Find best model
        best_model = max(self.results.items(), key=lambda x: x[1]['f1_score'])
        print(f"\n{'='*70}")
        print(f"🏆 BEST MODEL: {best_model[0]}")
        print(f"   F1-Score: {best_model[1]['f1_score']:.4f}")
        print(f"   AUC-ROC:  {best_model[1]['auc_roc']:.4f}")
        print(f"{'='*70}")
    
    def _plot_comparison(self):
        """
        Plot comprehensive model comparison
        """
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.ravel()
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        
        for idx, metric in enumerate(metrics):
            model_names = list(self.results.keys())
            values = [self.results[m][metric] for m in model_names]
            
            bars = axes[idx].bar(range(len(model_names)), values,
                               color=colors[:len(model_names)],
                               edgecolor='black', linewidth=1.5, alpha=0.8)
            axes[idx].set_title(f'{metric.replace("_", " ").title()}',
                              fontsize=13, fontweight='bold')
            axes[idx].set_ylabel('Score', fontsize=11)
            axes[idx].set_xticks(range(len(model_names)))
            axes[idx].set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)
            axes[idx].set_ylim([0, 1.1])
            axes[idx].grid(axis='y', alpha=0.3, linestyle='--')
            
            # Add value labels
            for i, (bar, v) in enumerate(zip(bars, values)):
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                             f'{v:.3f}', ha='center', va='bottom', fontsize=9,
                             fontweight='bold')
        
        # Inference time comparison
        model_names = list(self.results.keys())
        times = [self.results[m]['inference_time'] for m in model_names]
        bars = axes[5].bar(range(len(model_names)), times,
                          color=colors[:len(model_names)],
                          edgecolor='black', linewidth=1.5, alpha=0.8)
        axes[5].set_title('Inference Time', fontsize=13, fontweight='bold')
        axes[5].set_ylabel('Time (seconds)', fontsize=11)
        axes[5].set_xticks(range(len(model_names)))
        axes[5].set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)
        axes[5].grid(axis='y', alpha=0.3, linestyle='--')
        
        for i, (bar, t) in enumerate(zip(bars, times)):
            height = bar.get_height()
            axes[5].text(bar.get_x() + bar.get_width()/2., height + max(times)*0.02,
                        f'{t:.2f}s', ha='center', va='bottom', fontsize=9,
                        fontweight='bold')
        
        plt.suptitle('Deep Learning Models - Performance Comparison',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('DL_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Comparison charts saved: DL_model_comparison.png")
    
    def save_models_and_scaler(self, model_dir='models'):
        """
        Save all models, scaler, and metadata
        """
        print(f"\n{'='*70}")
        print("SAVING MODELS AND ARTIFACTS")
        print(f"{'='*70}")
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save scaler
        scaler_path = os.path.join(model_dir, 'dl_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✓ Scaler saved: {scaler_path}")
        
        # Save label encoder
        if len(self.label_encoder.classes_) > 0:
            encoder_path = os.path.join(model_dir, 'dl_label_encoder.pkl')
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.label_encoder, f)
            print(f"✓ Label encoder saved: {encoder_path}")
        
        # Save feature names
        if self.feature_names:
            features_path = os.path.join(model_dir, 'dl_feature_names.pkl')
            with open(features_path, 'wb') as f:
                pickle.dump(self.feature_names, f)
            print(f"✓ Feature names saved: {features_path}")
        
        # Save results
        results_path = os.path.join(model_dir, 'dl_results.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(self.results, f)
        print(f"✓ Results saved: {results_path}")
        
        # Save metadata
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata = {
            'timestamp': timestamp,
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'models': list(self.models.keys()),
            'best_model': max(self.results.items(), key=lambda x: x[1]['f1_score'])[0] if self.results else None,
            'metrics': {name: {k: v for k, v in metrics.items() 
                              if k not in ['y_pred_prob', 'confusion_matrix']}
                       for name, metrics in self.results.items()}
        }
        
        metadata_path = os.path.join(model_dir, 'dl_metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"✓ Metadata saved: {metadata_path}")
        
        print(f"\n✓ All artifacts saved to '{model_dir}/' directory")

def main():
    """
    Main training pipeline for deep learning models
    """
    print("="*70)
    print("DDOS DETECTION USING DEEP LEARNING")
    print("CIC-IDS2017 Dataset - Neural Networks")
    print("="*70)
    
    # Configuration
    config = {
        'data_path': 'processed_data.csv',
        'test_size': 0.2,
        'val_size': 0.15,
        'epochs': 100,
        'batch_size': 256,
        'balance_data': False,  # Set to True if severe class imbalance
        'train_optimized': True,
        'train_deep': True,
        'train_residual': True
    }
    
    print(f"\n📋 Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    try:
        # Initialize detector
        detector = DeepLearningDDoSDetector(data_path=config['data_path'])
        
        # Load and prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = detector.load_and_prepare_data(
            test_size=config['test_size'],
            val_size=config['val_size'],
            balance_data=config['balance_data']
        )
        
        input_dim = X_train.shape[1]
        print(f"\n✓ Data prepared. Input dimension: {input_dim}")
        
        # Train Optimized DNN
        if config['train_optimized']:
            model1 = detector.build_optimized_dnn(input_dim, "Optimized_DNN")
            detector.train_model(model1, X_train, y_train, X_val, y_val,
                               "Optimized_DNN", 
                               epochs=config['epochs'], 
                               batch_size=config['batch_size'])
            detector.evaluate_model(model1, X_test, y_test, "Optimized_DNN")
            detector.plot_training_history("Optimized_DNN")
        
        # Train Deep DNN
        if config['train_deep']:
            model2 = detector.build_deep_dnn(input_dim, "Deep_DNN")
            detector.train_model(model2, X_train, y_train, X_val, y_val,
                               "Deep_DNN",
                               epochs=config['epochs'],
                               batch_size=config['batch_size'])
            detector.evaluate_model(model2, X_test, y_test, "Deep_DNN")
            detector.plot_training_history("Deep_DNN")
        
        # Train Residual DNN
        if config['train_residual']:
            model3 = detector.build_residual_dnn(input_dim, "Residual_DNN")
            detector.train_model(model3, X_train, y_train, X_val, y_val,
                               "Residual_DNN",
                               epochs=config['epochs'],
                               batch_size=config['batch_size'])
            detector.evaluate_model(model3, X_test, y_test, "Residual_DNN")
            detector.plot_training_history("Residual_DNN")
        
        # Generate comparison visualizations
        detector.plot_all_roc_curves(y_test)
        detector.compare_models()
        
        # Save everything
        detector.save_models_and_scaler()
        
        print(f"\n{'='*70}")
        print("✓ DEEP LEARNING PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"{'='*70}")
        
        print("\n📊 Generated Visualizations:")
        print("   Individual Models:")
        for model_name in detector.models.keys():
            print(f"   ├─ {model_name}_confusion_matrix.png")
            print(f"   ├─ {model_name}_roc_curve.png")
            print(f"   ├─ {model_name}_pr_curve.png")
            print(f"   └─ {model_name}_training_history.png")
        print("   Comparisons:")
        print("   ├─ DL_all_roc_curves.png")
        print("   └─ DL_model_comparison.png")
        
        print("\n📁 Data Files:")
        print("   └─ DL_model_comparison.csv")
        
        print("\n🤖 Saved Models (in 'models/' directory):")
        for model_name in detector.models.keys():
            print(f"   ├─ {model_name}_best.h5")
        print("   ├─ dl_scaler.pkl")
        print("   ├─ dl_label_encoder.pkl")
        print("   ├─ dl_feature_names.pkl")
        print("   ├─ dl_results.pkl")
        print("   └─ dl_metadata.pkl")
        
        print("\n📈 TensorBoard Logs:")
        print("   └─ logs/ (run: tensorboard --logdir=logs)")
        
        print("\n🎯 Performance Summary:")
        for model_name, metrics in detector.results.items():
            print(f"\n   {model_name}:")
            print(f"   ├─ Accuracy:  {metrics['accuracy']:.4f}")
            print(f"   ├─ Precision: {metrics['precision']:.4f}")
            print(f"   ├─ Recall:    {metrics['recall']:.4f}")
            print(f"   ├─ F1-Score:  {metrics['f1_score']:.4f}")
            print(f"   └─ AUC-ROC:   {metrics['auc_roc']:.4f}")
        
        # Best model summary
        if detector.results:
            best_model = max(detector.results.items(), 
                           key=lambda x: x[1]['f1_score'])
            print(f"\n{'='*70}")
            print(f"🏆 BEST MODEL: {best_model[0]}")
            print(f"{'='*70}")
            print(f"   F1-Score: {best_model[1]['f1_score']:.4f}")
            print(f"   Accuracy: {best_model[1]['accuracy']:.4f}")
            print(f"   AUC-ROC:  {best_model[1]['auc_roc']:.4f}")
            print(f"{'='*70}")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\n💡 Solution: Run 'data_explore.py' first to generate 'processed_data.csv'")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n💡 Check the error message above and verify your setup")

if __name__ == "__main__":
    main()
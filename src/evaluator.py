"""
Task 5: Model Evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score
)
from sklearn.model_selection import KFold
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow.keras.models import clone_model
import os
import config

class ModelEvaluator:
    """Comprehensive model evaluation and visualization"""
    
    def __init__(self, model, visualization_dir='visualizations/'):
        """
        Initialize evaluator
        
        Args:
            model: Trained Keras model
            visualization_dir: Directory to save plots
        """
        self.model = model
        self.visualization_dir = visualization_dir
        self.evaluation_results = {}
        
        # Create visualization directory
        os.makedirs(visualization_dir, exist_ok=True)
        
    def evaluate(self, X_test, y_test, class_names=None):
        """Comprehensive model evaluation on test set"""
        print("\nEvaluating model on test set...")
        
        # Make predictions
        y_pred_proba = self.model.predict(X_test, verbose=0)
        
        # For binary classification
        if y_pred_proba.shape[1] == 1:
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
            y_pred_labels = y_pred
        # For multiclass classification
        else:
            y_pred = np.argmax(y_pred_proba, axis=1)
            y_pred_labels = y_pred
        
        # Calculate metrics
        results = self._calculate_metrics(y_test, y_pred_labels, class_names)
        
        # Generate visualizations
        self._create_evaluation_visualizations(y_test, y_pred_labels, 
                                              y_pred_proba, class_names)
        
        # Store results
        self.evaluation_results['test_set'] = results
        
        # Print summary
        self._print_evaluation_summary(results)
        
        return results
    
    def _calculate_metrics(self, y_true, y_pred, class_names=None):
        """Calculate comprehensive evaluation metrics"""
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Handle class_names properly
        if class_names is not None:
            if hasattr(class_names, 'shape'):  # It's a numpy array
                class_names_list = class_names.tolist()
            else:
                class_names_list = class_names
        else:
            class_names_list = [str(i) for i in range(len(np.unique(y_true)))]
        
        # Per-class metrics
        if class_names_list and len(class_names_list) <= 10:  # Limit for readability
            per_class_metrics = {}
            for i, class_name in enumerate(class_names_list):
                if i in y_true or i in y_pred:
                    # Create binary masks for this class
                    y_true_bin = (y_true == i).astype(int)
                    y_pred_bin = (y_pred == i).astype(int)
                    
                    # Calculate metrics for this class
                    try:
                        precision_i = precision_score(y_true_bin, y_pred_bin, zero_division=0)
                        recall_i = recall_score(y_true_bin, y_pred_bin, zero_division=0)
                        f1_i = f1_score(y_true_bin, y_pred_bin, zero_division=0)
                    except:
                        precision_i = recall_i = f1_i = 0
                    
                    per_class_metrics[class_name] = {
                        'precision': precision_i,
                        'recall': recall_i,
                        'f1_score': f1_i,
                        'support': np.sum(y_true == i)
                    }
        else:
            per_class_metrics = None
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # ROC-AUC for binary or multiclass
        if len(np.unique(y_true)) == 2:
            # Binary classification
            try:
                roc_auc = roc_auc_score(y_true, y_pred)
            except:
                roc_auc = 0.5  # Random classifier
        else:
            # Multiclass - one-vs-rest AUC
            try:
                roc_auc = roc_auc_score(y_true, y_pred, multi_class='ovr')
            except:
                roc_auc = 0.5
        
        # Classification report as dictionary
        report_dict = classification_report(
            y_true, y_pred, 
            target_names=class_names_list if class_names_list else None,
            output_dict=True,
            zero_division=0
        )
        
        results = {
            'accuracy': accuracy,
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist(),
            'per_class_metrics': per_class_metrics,
            'classification_report': report_dict,
            'predictions': {
                'true': y_true.tolist(),
                'predicted': y_pred.tolist(),
                'correct': (y_true == y_pred).tolist()
            }
        }
        
        return results
    
    def _create_evaluation_visualizations(self, y_true, y_pred, y_pred_proba, class_names):
        """Create comprehensive evaluation visualizations"""
        
        # 1. Confusion Matrix Heatmap
        self._plot_confusion_matrix(y_true, y_pred, class_names)
        
        # 2. ROC Curve (for binary or multiclass)
        if len(np.unique(y_true)) == 2:
            self._plot_roc_curve(y_true, y_pred_proba)
        elif len(np.unique(y_true)) <= 10:  # Limit for readability
            self._plot_multiclass_roc(y_true, y_pred_proba, class_names)
        
        # 3. Metrics Bar Plot
        self._plot_metrics_comparison(y_true, y_pred, class_names)
        
        # 4. Prediction Distribution
        self._plot_prediction_distribution(y_true, y_pred, class_names)
        
        # 5. Error Analysis
        self._plot_error_analysis(y_true, y_pred, class_names)
    
    def _plot_confusion_matrix(self, y_true, y_pred, class_names):
        """Plot confusion matrix heatmap"""
        cm = confusion_matrix(y_true, y_pred)
        
        # Handle class_names properly
        if class_names is not None:
            if hasattr(class_names, 'shape'):  # It's a numpy array
                class_names_list = class_names.tolist()
            else:
                class_names_list = class_names
        else:
            class_names_list = [str(i) for i in range(len(cm))]
        
        plt.figure(figsize=(10, 8))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
        
        # Create heatmap
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names_list,
                   yticklabels=class_names_list)
        
        plt.title('Normalized Confusion Matrix', fontsize=16, pad=20)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f"{self.visualization_dir}confusion_matrix.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also save raw confusion matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                   xticklabels=class_names_list,
                   yticklabels=class_names_list)
        ax.set_title('Confusion Matrix (Counts)', fontsize=16, pad=20)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{self.visualization_dir}confusion_matrix_raw.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    Confusion matrix plots saved")
    
    def _plot_roc_curve(self, y_true, y_pred_proba):
        """Plot ROC curve for binary classification"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, pad=20)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.visualization_dir}roc_curve.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ROC curve plot saved")
    
    def _plot_multiclass_roc(self, y_true, y_pred_proba, class_names):
        """Plot ROC curves for multiclass classification (one-vs-rest)"""
        # Handle class_names if it's a numpy array
        if class_names is not None:
            if hasattr(class_names, 'shape'):  # It's a numpy array
                n_classes = len(class_names)
                class_names_list = class_names.tolist()
            else:
                n_classes = len(class_names)
                class_names_list = class_names
        else:
            n_classes = y_pred_proba.shape[1]
            class_names_list = [f'Class {i}' for i in range(n_classes)]
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        # Binarize the output for one-vs-rest
        try:
            y_true_bin = label_binarize(y_true, classes=range(n_classes))
            
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            # Plot all ROC curves
            plt.figure(figsize=(10, 8))
            colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))
            
            for i, color in zip(range(n_classes), colors):
                class_label = class_names_list[i] if i < len(class_names_list) else f'Class {i}'
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                        label=f'{class_label} (AUC = {roc_auc[i]:.2f})')
            
            plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title('Multiclass ROC Curves (One-vs-Rest)', fontsize=16, pad=20)
            plt.legend(loc="lower right", fontsize=10)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{self.visualization_dir}multiclass_roc.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"    Multiclass ROC curves plot saved")
        except Exception as e:
            print(f"    Could not create multiclass ROC plot: {e}")
    
    def _plot_metrics_comparison(self, y_true, y_pred, class_names):
        """Plot comparison of different metrics"""
        # Handle class_names if it's a numpy array
        if class_names is not None:
            if hasattr(class_names, 'shape'):  # It's a numpy array
                class_names_list = class_names.tolist()
            else:
                class_names_list = class_names
        else:
            class_names_list = [str(i) for i in range(len(np.unique(y_true)))]
        
        # Calculate per-class metrics
        precision_per_class = []
        recall_per_class = []
        f1_per_class = []
        
        for i in range(len(class_names_list)):
            # Create binary masks for this class
            y_true_bin = (y_true == i).astype(int)
            y_pred_bin = (y_pred == i).astype(int)
            
            # Calculate metrics
            try:
                precision_i = precision_score(y_true_bin, y_pred_bin, zero_division=0)
                recall_i = recall_score(y_true_bin, y_pred_bin, zero_division=0)
                f1_i = f1_score(y_true_bin, y_pred_bin, zero_division=0)
            except:
                precision_i = recall_i = f1_i = 0
            
            precision_per_class.append(precision_i)
            recall_per_class.append(recall_i)
            f1_per_class.append(f1_i)
        
        # Create bar plot
        x = np.arange(len(class_names_list))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width, precision_per_class, width, label='Precision', color='skyblue')
        bars2 = ax.bar(x, recall_per_class, width, label='Recall', color='lightgreen')
        bars3 = ax.bar(x + width, f1_per_class, width, label='F1-Score', color='salmon')
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:  # Only add label if height > 0
                    ax.annotate(f'{height:.2f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),  # 3 points vertical offset
                               textcoords="offset points",
                               ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Performance Metrics', fontsize=16, pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(class_names_list, rotation=45 if len(class_names_list) > 5 else 0)
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f"{self.visualization_dir}per_class_metrics.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    Per-class metrics plot saved")
    
    def _plot_prediction_distribution(self, y_true, y_pred, class_names):
        """Plot distribution of correct vs incorrect predictions"""
        correct = (y_true == y_pred)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Overall accuracy
        correct_count = np.sum(correct)
        incorrect_count = len(correct) - correct_count
        
        axes[0].bar(['Correct', 'Incorrect'], [correct_count, incorrect_count], 
                   color=['green', 'red'])
        axes[0].set_title('Overall Prediction Distribution', fontsize=14)
        axes[0].set_ylabel('Count')
        axes[0].set_ylim([0, len(correct) * 1.1])
        
        # Add percentage labels
        for i, (count, label) in enumerate(zip([correct_count, incorrect_count], 
                                              ['Correct', 'Incorrect'])):
            percentage = count / len(correct) * 100
            axes[0].text(i, count + max(correct_count, incorrect_count)*0.01,
                       f'{percentage:.1f}%', ha='center', fontsize=11)
        
        # Plot 2: Accuracy per class
        if class_names is not None:
            # Handle class_names if it's a numpy array
            if hasattr(class_names, 'shape'):  # It's a numpy array
                class_names_list = class_names.tolist()
            else:
                class_names_list = class_names
            
            class_accuracies = []
            for i, class_name in enumerate(class_names_list):
                class_mask = (y_true == i)
                if np.sum(class_mask) > 0:
                    class_acc = np.mean(y_pred[class_mask] == y_true[class_mask])
                    class_accuracies.append(class_acc)
                else:
                    class_accuracies.append(0)
            
            colors = ['green' if acc > 0.7 else 'orange' if acc > 0.5 else 'red' 
                     for acc in class_accuracies]
            
            bars = axes[1].bar(class_names_list, class_accuracies, color=colors)
            axes[1].set_title('Accuracy per Class', fontsize=14)
            axes[1].set_ylabel('Accuracy')
            axes[1].set_ylim([0, 1.1])
            axes[1].tick_params(axis='x', rotation=45 if len(class_names_list) > 5 else 0)
            
            # Add value labels
            for bar, acc in zip(bars, class_accuracies):
                if acc > 0:  # Only add label if accuracy > 0
                    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{acc:.2f}', ha='center', fontsize=10)
        
        plt.suptitle('Prediction Analysis', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(f"{self.visualization_dir}prediction_distribution.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    Prediction distribution plot saved")
    
    def _plot_error_analysis(self, y_true, y_pred, class_names):
        """Analyze and visualize error patterns"""
        # Handle class_names if it's a numpy array
        if class_names is not None:
            if hasattr(class_names, 'shape'):  # It's a numpy array
                class_names_list = class_names.tolist()
            else:
                class_names_list = class_names
        else:
            class_names_list = [str(i) for i in range(len(np.unique(y_true)))]
        
        # Create error matrix (what was predicted vs what it should be)
        error_matrix = confusion_matrix(y_true, y_pred)
        np.fill_diagonal(error_matrix, 0)  # Remove correct predictions
        
        # Only plot if there are errors
        if np.sum(error_matrix) > 0:
            plt.figure(figsize=(10, 8))
            
            # Normalize by row (true class)
            error_matrix_norm = error_matrix.astype('float') / error_matrix.sum(axis=1)[:, np.newaxis]
            error_matrix_norm = np.nan_to_num(error_matrix_norm)  # Handle division by zero
            
            sns.heatmap(error_matrix_norm, annot=True, fmt='.2f', cmap='Reds',
                       xticklabels=class_names_list,
                       yticklabels=class_names_list)
            
            plt.title('Error Analysis: Misclassification Patterns', fontsize=16, pad=20)
            plt.xlabel('Predicted as', fontsize=12)
            plt.ylabel('Actually is', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(f"{self.visualization_dir}error_analysis.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"    Error analysis plot saved")
    
    def _print_evaluation_summary(self, results):
        """Print comprehensive evaluation summary"""
        print("\n" + "="*70)
        print("MODEL EVALUATION SUMMARY")
        print("="*70)
        
        print(f"\nOverall Metrics:")
        print(f"  • Accuracy:  {results['accuracy']:.4f}")
        print(f"  • Precision: {results['precision_weighted']:.4f}")
        print(f"  • Recall:    {results['recall_weighted']:.4f}")
        print(f"  • F1-Score:  {results['f1_weighted']:.4f}")
        print(f"  • ROC-AUC:   {results['roc_auc']:.4f}")
        
        print(f"\nTest Set Statistics:")
        if 'per_class_metrics' in results and results['per_class_metrics']:
            print(f"\nPer-Class Performance:")
            for class_name, metrics in results['per_class_metrics'].items():
                print(f"  {class_name}:")
                print(f"    • Precision: {metrics['precision']:.3f}")
                print(f"    • Recall:    {metrics['recall']:.3f}")
                print(f"    • F1-Score:  {metrics['f1_score']:.3f}")
                print(f"    • Support:   {metrics['support']}")
        
        # Print confusion matrix summary
        cm = np.array(results['confusion_matrix'])
        print(f"\nConfusion Matrix Summary:")
        print(f"  • Total predictions: {np.sum(cm)}")
        print(f"  • Correct predictions: {np.trace(cm)}")
        print(f"  • Incorrect predictions: {np.sum(cm) - np.trace(cm)}")
        print(f"  • Overall accuracy: {np.trace(cm)/np.sum(cm):.2%}")
        
        # Identify most confused classes
        if len(cm) > 1:
            cm_no_diag = cm.copy()
            np.fill_diagonal(cm_no_diag, 0)
            
            if np.sum(cm_no_diag) > 0:
                max_error = np.unravel_index(np.argmax(cm_no_diag), cm_no_diag.shape)
                print(f"  • Most common error: Class {max_error[0]} → Class {max_error[1]} "
                      f"({cm_no_diag[max_error]} instances)")
        
        print(f"\nEvaluation visualizations saved to: {self.visualization_dir}")
    
    def cross_validate(self, X, y, k_folds=5, epochs=50, batch_size=32):
        """Perform k-fold cross-validation"""
        print(f"\nPerforming {k_folds}-fold Cross-Validation...")
        
        # FIX: Ensure data is in proper format
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        
        # Convert to proper dtypes for Keras
        X = X.astype(np.float32)
        y = y.astype(np.int32)
        
        print(f"  X shape: {X.shape}, dtype: {X.dtype}")
        print(f"  y shape: {y.shape}, dtype: {y.dtype}")
        
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=config.RANDOM_STATE)
        
        cv_scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        fold = 1
        for train_idx, val_idx in kfold.split(X, y):
            print(f"\n  Fold {fold}/{k_folds}:")
            
            # Split data - Now X and y are guaranteed to be numpy arrays
            X_train_fold = X[train_idx]
            X_val_fold = X[val_idx]
            y_train_fold = y[train_idx]
            y_val_fold = y[val_idx]
            
            # Clone and train model
            model_clone = clone_model(self.model)
            model_clone.compile(
                optimizer=self.model.optimizer.__class__(learning_rate=config.LEARNING_RATE),
                loss=self.model.loss,
                metrics=['accuracy']
            )
            
            # Train with early stopping
            early_stopping = tf.keras.callbacks.EarlyStopping(
                patience=10, restore_best_weights=True
            )
            
            model_clone.fit(
                X_train_fold, y_train_fold,
                validation_data=(X_val_fold, y_val_fold),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Evaluate
            y_pred = model_clone.predict(X_val_fold, verbose=0)
            
            if y_pred.shape[1] == 1:
                y_pred_labels = (y_pred > 0.5).astype(int).flatten()
            else:
                y_pred_labels = np.argmax(y_pred, axis=1)
            
            # Calculate metrics
            cv_scores['accuracy'].append(accuracy_score(y_val_fold, y_pred_labels))
            cv_scores['precision'].append(precision_score(y_val_fold, y_pred_labels, 
                                                         average='weighted', zero_division=0))
            cv_scores['recall'].append(recall_score(y_val_fold, y_pred_labels, 
                                                   average='weighted', zero_division=0))
            cv_scores['f1'].append(f1_score(y_val_fold, y_pred_labels, 
                                           average='weighted', zero_division=0))
            
            print(f"    Accuracy: {cv_scores['accuracy'][-1]:.4f}, "
                  f"F1-Score: {cv_scores['f1'][-1]:.4f}")
            
            fold += 1
        
        # Calculate statistics
        results = {
            'mean_accuracy': np.mean(cv_scores['accuracy']),
            'std_accuracy': np.std(cv_scores['accuracy']),
            'mean_precision': np.mean(cv_scores['precision']),
            'std_precision': np.std(cv_scores['precision']),
            'mean_recall': np.mean(cv_scores['recall']),
            'std_recall': np.std(cv_scores['recall']),
            'mean_f1': np.mean(cv_scores['f1']),
            'std_f1': np.std(cv_scores['f1']),
            'fold_scores': cv_scores
        }
        
        # Plot CV results
        self._plot_cross_validation_results(cv_scores)
        
        # Store results
        self.evaluation_results['cross_validation'] = results
        
        print(f"\nCross-Validation Results:")
        print(f"  Mean Accuracy:  {results['mean_accuracy']:.4f} (±{results['std_accuracy']:.4f})")
        print(f"  Mean F1-Score:  {results['mean_f1']:.4f} (±{results['std_f1']:.4f})")
        print(f"  Score Range:    {min(cv_scores['accuracy']):.4f} - {max(cv_scores['accuracy']):.4f}")
        
        return results
    
    def _plot_cross_validation_results(self, cv_scores):
        """Plot cross-validation results"""
        folds = list(range(1, len(cv_scores['accuracy']) + 1))
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Accuracy across folds
        axes[0, 0].plot(folds, cv_scores['accuracy'], 'o-', linewidth=2, markersize=8)
        axes[0, 0].axhline(y=np.mean(cv_scores['accuracy']), color='r', linestyle='--', 
                          label=f'Mean: {np.mean(cv_scores["accuracy"]):.3f}')
        axes[0, 0].fill_between(folds, 
                               np.mean(cv_scores['accuracy']) - np.std(cv_scores['accuracy']),
                               np.mean(cv_scores['accuracy']) + np.std(cv_scores['accuracy']),
                               alpha=0.2, color='gray')
        axes[0, 0].set_title('Accuracy across Folds', fontsize=14)
        axes[0, 0].set_xlabel('Fold')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_xticks(folds)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: F1-Score across folds
        axes[0, 1].plot(folds, cv_scores['f1'], 's-', linewidth=2, markersize=8, color='green')
        axes[0, 1].axhline(y=np.mean(cv_scores['f1']), color='r', linestyle='--',
                          label=f'Mean: {np.mean(cv_scores["f1"]):.3f}')
        axes[0, 1].fill_between(folds,
                               np.mean(cv_scores['f1']) - np.std(cv_scores['f1']),
                               np.mean(cv_scores['f1']) + np.std(cv_scores['f1']),
                               alpha=0.2, color='green')
        axes[0, 1].set_title('F1-Score across Folds', fontsize=14)
        axes[0, 1].set_xlabel('Fold')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].set_xticks(folds)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Box plot of all metrics
        metrics_data = [cv_scores['accuracy'], cv_scores['precision'], 
                       cv_scores['recall'], cv_scores['f1']]
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        bp = axes[1, 0].boxplot(metrics_data, labels=metric_names, patch_artist=True)
        
        # Color the boxes
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        axes[1, 0].set_title('Metrics Distribution across Folds', fontsize=14)
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Summary statistics
        summary_text = (f"Cross-Validation Summary ({len(folds)}-fold)\n"
                       f"==============================\n"
                       f"Accuracy:  {np.mean(cv_scores['accuracy']):.3f} ± {np.std(cv_scores['accuracy']):.3f}\n"
                       f"Precision: {np.mean(cv_scores['precision']):.3f} ± {np.std(cv_scores['precision']):.3f}\n"
                       f"Recall:    {np.mean(cv_scores['recall']):.3f} ± {np.std(cv_scores['recall']):.3f}\n"
                       f"F1-Score:  {np.mean(cv_scores['f1']):.3f} ± {np.std(cv_scores['f1']):.3f}\n"
                       f"\nStability: {'Good' if np.std(cv_scores['accuracy']) < 0.05 else 'Moderate' if np.std(cv_scores['accuracy']) < 0.1 else 'Variable'}")
        
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].axis('off')
        
        plt.suptitle(f'{len(folds)}-Fold Cross-Validation Results', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(f"{self.visualization_dir}cross_validation_results.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    Cross-validation results plot saved")
    
    def get_comprehensive_report(self):
        """Generate comprehensive evaluation report"""
        if not self.evaluation_results:
            raise ValueError("No evaluation results available. Run evaluate() first.")
        
        report = {
            'test_set_performance': self.evaluation_results.get('test_set', {}),
            'cross_validation': self.evaluation_results.get('cross_validation', {}),
            'model_performance_summary': self._generate_performance_summary()
        }
        
        return report
    
    def _generate_performance_summary(self):
        """Generate performance summary for reporting"""
        test_results = self.evaluation_results.get('test_set', {})
        cv_results = self.evaluation_results.get('cross_validation', {})
        
        summary = {
            'overall_performance': {
                'test_accuracy': test_results.get('accuracy', 0),
                'test_f1_score': test_results.get('f1_weighted', 0),
                'cv_mean_accuracy': cv_results.get('mean_accuracy', 0),
                'cv_std_accuracy': cv_results.get('std_accuracy', 0),
                'performance_consistency': 'High' if cv_results.get('std_accuracy', 1) < 0.05 
                                          else 'Medium' if cv_results.get('std_accuracy', 1) < 0.1 
                                          else 'Low'
            },
            'model_strengths': self._identify_strengths(test_results),
            'model_weaknesses': self._identify_weaknesses(test_results),
            'recommendations': self._generate_recommendations(test_results, cv_results)
        }
        
        return summary
    
    def _identify_strengths(self, results):
        """Identify model strengths based on results"""
        strengths = []
        
        if results.get('accuracy', 0) > 0.85:
            strengths.append("High overall accuracy")
        elif results.get('accuracy', 0) > 0.75:
            strengths.append("Good overall accuracy")
        
        if results.get('f1_weighted', 0) > 0.8:
            strengths.append("Balanced precision and recall (good F1-score)")
        
        if 'per_class_metrics' in results and results['per_class_metrics']:
            balanced = all(m['f1_score'] > 0.7 for m in results['per_class_metrics'].values())
            if balanced:
                strengths.append("Consistent performance across all classes")
        
        if results.get('roc_auc', 0.5) > 0.8:
            strengths.append("Excellent discriminative power (high AUC)")
        
        if not strengths:
            strengths.append("Baseline performance achieved")
        
        return strengths
    
    def _identify_weaknesses(self, results):
        """Identify model weaknesses based on results"""
        weaknesses = []
        
        if results.get('accuracy', 0) < 0.6:
            weaknesses.append("Low overall accuracy")
        
        if results.get('f1_weighted', 0) < 0.6:
            weaknesses.append("Poor balance between precision and recall")
        
        if 'per_class_metrics' in results and results['per_class_metrics']:
            imbalanced = any(m['f1_score'] < 0.5 for m in results['per_class_metrics'].values())
            if imbalanced:
                weaknesses.append("Poor performance on some classes")
            
            # Check for class imbalance in predictions
            support_values = [m['support'] for m in results['per_class_metrics'].values()]
            if max(support_values) / min(support_values) > 10:
                weaknesses.append("Class imbalance affecting performance")
        
        if results.get('roc_auc', 0.5) < 0.7:
            weaknesses.append("Limited discriminative power")
        
        return weaknesses
    
    def _generate_recommendations(self, test_results, cv_results):
        """Generate recommendations for improvement"""
        recommendations = []
        
        # Based on accuracy
        if test_results.get('accuracy', 0) < 0.7:
            recommendations.extend([
                "Consider more complex model architecture",
                "Increase training data or use data augmentation",
                "Perform more extensive feature engineering"
            ])
        
        # Based on F1-score
        if test_results.get('f1_weighted', 0) < test_results.get('accuracy', 0) * 0.9:
            recommendations.append("Address class imbalance with techniques like SMOTE or class weighting")
        
        # Based on cross-validation stability
        if cv_results.get('std_accuracy', 0) > 0.1:
            recommendations.extend([
                "Model performance is variable - consider ensemble methods",
                "Increase cross-validation folds for more reliable estimates",
                "Check for data leakage or preprocessing inconsistencies"
            ])
        
        # Based on confusion matrix analysis
        if 'confusion_matrix' in test_results:
            cm = np.array(test_results['confusion_matrix'])
            if np.sum(cm) - np.trace(cm) > np.trace(cm) * 0.5:  # More than 50% errors
                recommendations.append("High misclassification rate - consider different algorithm or features")
        
        # General recommendations
        recommendations.extend([
            "Experiment with different neural network architectures",
            "Perform hyperparameter tuning (learning rate, dropout, layers)",
            "Try different activation functions or optimizers",
            "Consider feature selection to reduce dimensionality"
        ])
        
        return recommendations
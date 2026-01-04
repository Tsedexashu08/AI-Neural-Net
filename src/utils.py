"""
Utility functions for reporting and file operations
"""

import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import os
import config

def save_results(results, filepath):
    """Save results to file in multiple formats"""
    # Save as JSON
    with open(filepath, 'w', encoding='utf-8') as f:
        # Convert numpy types to Python native types
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, tuple):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_to_serializable(results)
        json.dump(serializable_results, f, indent=2)
    
    # Also save as pickle for Python object preservation
    pickle_path = filepath.replace('.txt', '.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to {filepath}")
    print(f"Python object saved to {pickle_path}")

def generate_report(results_summary, report_path):
    """Generate comprehensive text report"""
    
    def safe_get_float(data, key, default=0.0):
        """Safely get a float value from dictionary"""
        value = data.get(key, default)
        if isinstance(value, str):
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        elif isinstance(value, (int, float, np.number)):
            return float(value)
        else:
            return default
    
    def safe_get_percentage(data, key, default=0.0):
        """Safely get a percentage value and format it"""
        value = safe_get_float(data, key, default)
        return f"{value:.2%}"
    
    # Use UTF-8 encoding to support all characters
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("NEURAL NETWORK CLASSIFICATION PROJECT - FINAL REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write("REPORT GENERATED: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
        
        # 1. EXECUTIVE SUMMARY
        f.write("1. EXECUTIVE SUMMARY\n")
        f.write("-"*50 + "\n")
        
        dataset_info = results_summary.get('dataset_info', {})
        test_results = results_summary.get('test_results', {})
        cv_results = results_summary.get('cv_results', {})
        
        f.write(f"Project completed neural network classification on a dataset with:\n")
        if isinstance(dataset_info, dict):
            f.write(f"  - Dataset shape information available\n")
            if 'shape' in dataset_info:
                shape_str = dataset_info['shape']
                if isinstance(shape_str, str):
                    f.write(f"  - Dataset shape: {shape_str}\n")
        else:
            f.write(f"  - Dataset information available\n")
        
        # Get accuracy safely
        test_accuracy = safe_get_float(test_results, 'accuracy', 0.0)
        test_f1 = safe_get_float(test_results, 'f1_weighted', 0.0)
        cv_mean_acc = safe_get_float(cv_results, 'mean_accuracy', 0.0)
        cv_std_acc = safe_get_float(cv_results, 'std_accuracy', 0.0)
        
        f.write(f"\nModel Performance:\n")
        f.write(f"  - Test Accuracy: {test_accuracy:.2%}\n")
        f.write(f"  - Test F1-Score: {test_f1:.2%}\n")
        f.write(f"  - Cross-Validation Accuracy: {cv_mean_acc:.2%} (+/-{cv_std_acc:.2%})\n\n")
        
        # 2. METHODOLOGY
        f.write("2. METHODOLOGY\n")
        f.write("-"*50 + "\n")
        
        f.write("2.1 Data Preprocessing:\n")
        preprocessing_info = results_summary.get('preprocessing_info', {})
        
        if isinstance(preprocessing_info, dict):
            # Missing values
            if 'missing_values' in preprocessing_info:
                mv = preprocessing_info['missing_values']
                count = mv.get('total_missing', 0)
                if isinstance(count, (int, float)):
                    f.write(f"  - Missing Values: {int(count)} values handled\n")
            
            # Encoding
            if 'encoding' in preprocessing_info:
                enc = preprocessing_info['encoding']
                if isinstance(enc, dict) and enc.get('strategy') != 'none':
                    f.write(f"  - Categorical Encoding: Applied to categorical features\n")
            
            # Scaling
            if 'scaling' in preprocessing_info:
                scaling = preprocessing_info['scaling']
                strategy = scaling.get('strategy', 'standard_scaler')
                f.write(f"  - Feature Scaling: {strategy}\n")
        
        f.write("\n2.2 Model Architecture:\n")
        model_arch = results_summary.get('model_architecture', {})
        if isinstance(model_arch, dict):
            hidden_layers = model_arch.get('hidden_layers', [])
            # Use ASCII arrow instead of Unicode arrow
            architecture_str = f"{model_arch.get('input_dimension', 0)} -> "
            architecture_str += " -> ".join(map(str, hidden_layers))
            architecture_str += f" -> {model_arch.get('num_classes', 0)}"
            
            f.write(f"  - Type: {len(hidden_layers)}-layer neural network\n")
            f.write(f"  - Architecture: {architecture_str}\n")
            dropout_rate_val = safe_get_float(model_arch, 'dropout_rate', 0)
            f.write(f"  - Regularization: Dropout ({dropout_rate_val:.0%}), "
                   f"L2 (lambda={model_arch.get('l2_regularization', 0)})\n")
            try:
                total_params = int(model_arch.get('total_params', 0))
                f.write(f"  - Total Parameters: {total_params:,}\n")
            except (ValueError, TypeError):
                f.write(f"  - Total Parameters: {model_arch.get('total_params', 'N/A')}\n")
        
        f.write("\n2.3 Training Protocol:\n")
        training_info = results_summary.get('training_history', {})
        if training_info:
            config_info = results_summary.get('config', {})
            f.write(f"  - Optimizer: Adam\n")
            f.write(f"  - Learning Rate: {config_info.get('learning_rate', 0.001)}\n")
            f.write(f"  - Batch Size: {config_info.get('batch_size', 32)}\n")
            if isinstance(training_info, dict) and 'loss' in training_info:
                epochs = len(training_info['loss']) if isinstance(training_info['loss'], list) else 0
                f.write(f"  - Epochs: {epochs} (with early stopping)\n")
        
        # 3. RESULTS
        f.write("\n3. RESULTS\n")
        f.write("-"*50 + "\n")
        
        f.write("3.1 Training Performance:\n")
        if training_info and isinstance(training_info, dict):
            # Get best validation accuracy
            if 'val_accuracy' in training_info and isinstance(training_info['val_accuracy'], list):
                val_acc = training_info['val_accuracy']
                if val_acc:
                    try:
                        best_val_acc = max([float(v) for v in val_acc if isinstance(v, (int, float, str))])
                        f.write(f"  - Best Validation Accuracy: {best_val_acc:.2%}\n")
                    except (ValueError, TypeError):
                        f.write(f"  - Best Validation Accuracy: Data not available or format error\n")
            
            # Get best validation loss
            if 'val_loss' in training_info and isinstance(training_info['val_loss'], list):
                val_loss = training_info['val_loss']
                if val_loss:
                    try:
                        best_val_loss = min([float(v) for v in val_loss if isinstance(v, (int, float, str))])
                        f.write(f"  - Best Validation Loss: {best_val_loss:.4f}\n")
                    except:
                        f.write(f"  - Best Validation Loss: Data available\n")
        
        f.write("\n3.2 Test Set Performance:\n")
        f.write(f"  - Accuracy: {test_accuracy:.2%}\n")
        f.write(f"  - Precision: {safe_get_percentage(test_results, 'precision_weighted')}\n")
        f.write(f"  - Recall: {safe_get_percentage(test_results, 'recall_weighted')}\n")
        f.write(f"  - F1-Score: {test_f1:.2%}\n")
        f.write(f"  - ROC-AUC: {safe_get_percentage(test_results, 'roc_auc')}\n")
        
        f.write("\n3.3 Cross-Validation Results:\n")
        f.write(f"  - Mean Accuracy: {cv_mean_acc:.2%}\n")
        f.write(f"  - Standard Deviation: {cv_std_acc:.2%}\n")
        
        # Determine performance consistency
        if cv_std_acc < 0.05:
            consistency = "High"
        elif cv_std_acc < 0.1:
            consistency = "Moderate"
        else:
            consistency = "Variable"
        f.write(f"  - Performance Consistency: {consistency}\n")
        
        # 4. DISCUSSION
        f.write("\n4. DISCUSSION\n")
        f.write("-"*50 + "\n")
        
        f.write("4.1 Model Strengths:\n")
        if test_accuracy > 0.85:
            f.write(f"  - High overall accuracy ({test_accuracy:.2%})\n")
        elif test_accuracy > 0.75:
            f.write(f"  - Good overall accuracy ({test_accuracy:.2%})\n")
        elif test_accuracy > 0.65:
            f.write(f"  - Acceptable accuracy ({test_accuracy:.2%})\n")
        else:
            f.write(f"  - Baseline performance achieved\n")
        
        if test_f1 > 0.8:
            f.write(f"  - Balanced precision and recall (good F1-score: {test_f1:.2%})\n")
        
        f.write("\n4.2 Model Weaknesses:\n")
        if test_accuracy < 0.6:
            f.write(f"  - Low overall accuracy\n")
        
        if test_f1 < 0.6:
            f.write(f"  - Poor balance between precision and recall\n")
        
        if cv_std_acc > 0.1:
            f.write(f"  - Variable performance across cross-validation folds\n")
        
        # 5. RECOMMENDATIONS FOR IMPROVEMENT
        f.write("\n5. RECOMMENDATIONS FOR IMPROVEMENT\n")
        f.write("-"*50 + "\n")
        
        recommendations = []
        
        # Based on accuracy
        if test_accuracy < 0.7:
            recommendations.extend([
                "Consider more complex model architecture",
                "Increase training data or use data augmentation",
                "Perform more extensive feature engineering"
            ])
        
        # Based on F1-score
        if test_f1 < test_accuracy * 0.9:
            recommendations.append("Address class imbalance with techniques like SMOTE or class weighting")
        
        # Based on cross-validation stability
        if cv_std_acc > 0.1:
            recommendations.extend([
                "Model performance is variable - consider ensemble methods",
                "Increase cross-validation folds for more reliable estimates",
                "Check for data leakage or preprocessing inconsistencies"
            ])
        
        # General recommendations
        recommendations.extend([
            "Experiment with different neural network architectures",
            "Perform hyperparameter tuning (learning rate, dropout, layers)",
            "Try different activation functions or optimizers",
            "Consider feature selection to reduce dimensionality"
        ])
        
        for i, recommendation in enumerate(recommendations, 1):
            f.write(f"{i}. {recommendation}\n")
        
        # 6. CONCLUSION
        f.write("\n6. CONCLUSION\n")
        f.write("-"*50 + "\n")
        
        if test_accuracy > 0.9:
            conclusion = "Excellent performance achieved. The model is highly effective for this classification task."
        elif test_accuracy > 0.8:
            conclusion = "Good performance achieved. The model is suitable for deployment with monitoring."
        elif test_accuracy > 0.7:
            conclusion = "Acceptable performance achieved. Consider further optimization for production use."
        elif test_accuracy > 0.6:
            conclusion = "Marginal performance achieved. Significant improvements needed before deployment."
        else:
            conclusion = "Poor performance. Consider revisiting problem formulation, data quality, or algorithm selection."
        
        f.write(conclusion + "\n\n")
        
        f.write("The project successfully demonstrated the complete neural network workflow from\n")
        f.write("data exploration to model evaluation. All deliverables have been generated and\n")
        f.write("are available in the output directories.\n")
        
        # 7. APPENDIX
        f.write("\n7. APPENDIX\n")
        f.write("-"*50 + "\n")
        
        f.write("Generated Files and Directories:\n")
        f.write("  - outputs/visualizations/: All plots and charts\n")
        f.write("  - outputs/models/: Saved model weights and architecture\n")
        f.write("  - outputs/processed_data/: Cleaned and preprocessed datasets\n")
        f.write("  - reports/: This report and detailed results\n")
        f.write("  - src/: Source code for all modules\n")
        
        f.write("\nKey Visualizations Generated:\n")
        f.write("  - Feature distributions and boxplots\n")
        f.write("  - Correlation heatmap\n")
        f.write("  - Class distribution charts\n")
        f.write("  - Training history plots\n")
        f.write("  - Confusion matrices\n")
        f.write("  - ROC curves\n")
        f.write("  - Cross-validation results\n")
        f.write("  - Error analysis plots\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*70 + "\n")
    
    print(f"Comprehensive report generated: {report_path}")

def create_presentation_summary():
    """Create a concise summary for presentation slides"""
    
    summary = """
    NEURAL NETWORK CLASSIFICATION PROJECT - PRESENTATION SUMMARY
    ============================================================
    
    1. PROJECT OVERVIEW
       - Goal: Build neural network classifier from raw data
       - Approach: End-to-end ML pipeline implementation
    
    2. KEY FINDINGS
       - Data Quality: Issues found and handled during preprocessing
       - Model Performance: Achieved through careful architecture design
       - Strengths: Robust evaluation methodology
       - Challenges: Data preprocessing and model optimization
    
    3. MODEL PERFORMANCE
       - Test Accuracy: [See report for details]
       - F1-Score: [See report for details]
       - Cross-Validation: [See report for details]
    
    4. VISUAL HIGHLIGHTS
       - Feature distributions and correlations
       - Training curves and validation metrics
       - Confusion matrix and error analysis
       - ROC curves for model discrimination
    
    5. CONCLUSIONS & RECOMMENDATIONS
       - Model performance assessment
       - Key recommendations for improvement
       - Next steps for implementation
    
    6. TECHNICAL IMPLEMENTATION
       - Framework: TensorFlow/Keras
       - Architecture: Multi-layer neural network
       - Training: Adam optimizer with early stopping
       - Evaluation: Comprehensive metrics suite
    """
    
    with open(f"{config.REPORT_DIR}presentation_summary.txt", 'w', encoding='utf-8') as f:
        f.write(summary)
    
    return summary
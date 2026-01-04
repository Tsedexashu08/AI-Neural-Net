"""
Main script to execute the complete neural network classification pipeline
"""

import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import json

# Add src to path
sys.path.append('src')

# Import custom modules
from src.data_loader import DataLoader
from src.data_preprocessor import DataPreprocessor
from src.model_builder import ModelBuilder
from src.trainer import ModelTrainer
from src.evaluator import ModelEvaluator
from src.utils import save_results, generate_report
import config

def convert_to_numpy(data):
    """Convert data to numpy array with proper dtype"""
    if hasattr(data, 'values'):
        # It's a pandas DataFrame/Series
        return data.values
    return data

def ensure_proper_dtype(X, y):
    """Ensure data has proper dtypes for Keras"""
    # Convert features to float32
    X_np = X.astype(np.float32)
    
    # Convert labels to int32 (for sparse categorical crossentropy)
    y_np = y.astype(np.int32)
    
    return X_np, y_np

def serialize_for_json(obj):
    """Convert non-serializable objects to JSON-serializable format"""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_json(item) for item in obj]
    elif isinstance(obj, tuple):
        return [serialize_for_json(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return str(obj)
    else:
        return str(obj)

def main():
    """Main execution function"""
    
    print("=" * 70)
    print("NEURAL NETWORK CLASSIFICATION PROJECT")
    print("=" * 70)
    
    # Create output directories
    for directory in [config.VISUALIZATION_DIR, config.MODEL_DIR, 
                     config.PROCESSED_DATA_DIR, config.REPORT_DIR]:
        os.makedirs(directory, exist_ok=True)
    
    # TASK 1: DATA UNDERSTANDING
    print("\n" + "="*70)
    print("TASK 1: DATA UNDERSTANDING (EXPLORATORY DATA ANALYSIS)")
    print("="*70)
    
    data_loader = DataLoader(config.DATA_PATH, config.TARGET_COLUMN)
    df = data_loader.load_data()
    
    # Generate EDA outputs
    eda_results = data_loader.perform_eda()
    
    print("\n✓ Task 1 Completed: EDA Results Generated")
    print(f"  - Summary statistics saved")
    print(f"  - Visualizations saved to {config.VISUALIZATION_DIR}")
    print(f"  - Data insights documented")
    
    # TASK 2: DATA PREPARATION
    print("\n" + "="*70)
    print("TASK 2: DATA PREPARATION")
    print("="*70)
    
    preprocessor = DataPreprocessor(config.TARGET_COLUMN, config.RANDOM_STATE)
    processed_data = preprocessor.prepare_data(df)
    
    # Save processed data
    processed_data['X_train'].to_csv(f"{config.PROCESSED_DATA_DIR}X_train.csv", index=False)
    processed_data['X_test'].to_csv(f"{config.PROCESSED_DATA_DIR}X_test.csv", index=False)
    pd.Series(processed_data['y_train']).to_csv(f"{config.PROCESSED_DATA_DIR}y_train.csv", index=False)
    pd.Series(processed_data['y_test']).to_csv(f"{config.PROCESSED_DATA_DIR}y_test.csv", index=False)
    
    print("\n✓ Task 2 Completed: Data Preparation Finished")
    print(f"  - Cleaned dataset saved to {config.PROCESSED_DATA_DIR}")
    print(f"  - Training samples: {len(processed_data['X_train'])}")
    print(f"  - Testing samples: {len(processed_data['X_test'])}")
    print(f"  - Features: {processed_data['X_train'].shape[1]}")
    print(f"  - Classes: {len(np.unique(processed_data['y_train']))}")
    
    # TASK 3: MODEL DESIGN
    print("\n" + "="*70)
    print("TASK 3: MODEL DESIGN")
    print("="*70)
    
    model_builder = ModelBuilder(
        input_dim=processed_data['X_train'].shape[1],
        num_classes=len(np.unique(processed_data['y_train'])),
        hidden_layers=config.HIDDEN_LAYERS,
        dropout_rate=config.DROPOUT_RATE,
        l2_reg=config.L2_REGULARIZATION
    )
    
    model = model_builder.build_model()
    model.summary(print_fn=lambda x: print(f"  {x}"))
    
    print("\n✓ Task 3 Completed: Model Architecture Defined")
    print(f"  - Input dimension: {processed_data['X_train'].shape[1]}")
    print(f"  - Hidden layers: {config.HIDDEN_LAYERS}")
    print(f"  - Output classes: {len(np.unique(processed_data['y_train']))}")
    
    # TASK 4: TRAINING PROCESS
    print("\n" + "="*70)
    print("TASK 4: TRAINING PROCESS")
    print("="*70)
    
    # Convert data to numpy arrays with proper dtypes
    print("Converting data to proper dtypes for training...")
    
    # Convert to numpy first
    X_train_np = convert_to_numpy(processed_data['X_train'])
    X_val_np = convert_to_numpy(processed_data['X_val'])
    X_test_np = convert_to_numpy(processed_data['X_test'])
    
    y_train_np = convert_to_numpy(processed_data['y_train'])
    y_val_np = convert_to_numpy(processed_data['y_val'])
    y_test_np = convert_to_numpy(processed_data['y_test'])
    
    # Ensure proper dtypes
    X_train_np, y_train_np = ensure_proper_dtype(X_train_np, y_train_np)
    X_val_np, y_val_np = ensure_proper_dtype(X_val_np, y_val_np)
    X_test_np, y_test_np = ensure_proper_dtype(X_test_np, y_test_np)
    
    print(f"  X_train shape: {X_train_np.shape}, dtype: {X_train_np.dtype}")
    print(f"  y_train shape: {y_train_np.shape}, dtype: {y_train_np.dtype}")
    
    trainer = ModelTrainer(
        model=model,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        patience=config.EARLY_STOPPING_PATIENCE,
        model_dir=config.MODEL_DIR
    )
    
    history = trainer.train(
        X_train=X_train_np,
        y_train=y_train_np,
        X_val=X_val_np,
        y_val=y_val_np
    )
    
    print("\n✓ Task 4 Completed: Model Training Finished")
    print(f"  - Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"  - Training epochs: {len(history.history['loss'])}")
    print(f"  - Model saved to {config.MODEL_DIR}")
    
    # TASK 5: EVALUATION
    print("\n" + "="*70)
    print("TASK 5: EVALUATION")
    print("="*70)
    
    evaluator = ModelEvaluator(model, config.VISUALIZATION_DIR)
    
    # Test set evaluation
    test_results = evaluator.evaluate(
        X_test=X_test_np,
        y_test=y_test_np,
        class_names=processed_data['class_names']
    )
    
    # Cross-validation (optional)
    print("\nPreparing data for cross-validation...")
    
    # Convert to numpy arrays for cross-validation
    X_cv = convert_to_numpy(processed_data['X_train_full'])
    y_cv = convert_to_numpy(processed_data['y_train_full'])
    
    # Ensure proper dtypes
    X_cv, y_cv = ensure_proper_dtype(X_cv, y_cv)
    
    print(f"  X_cv shape: {X_cv.shape}, dtype: {X_cv.dtype}")
    print(f"  y_cv shape: {y_cv.shape}, dtype: {y_cv.dtype}")
    
    print("\nPerforming Cross-Validation...")
    cv_results = evaluator.cross_validate(
        X=X_cv,
        y=y_cv,
        k_folds=config.K_FOLDS,
        epochs=50,
        batch_size=config.BATCH_SIZE
    )
    
    print("\n✓ Task 5 Completed: Model Evaluation Finished")
    print(f"  - Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"  - Test F1-Score: {test_results['f1_weighted']:.4f}")
    print(f"  - CV Mean Accuracy: {cv_results['mean_accuracy']:.4f} (±{cv_results['std_accuracy']:.4f})")
    
    # TASK 6: GENERATE FINAL REPORT
    print("\n" + "="*70)
    print("TASK 6: INTERPRETATION AND REPORTING")
    print("="*70)
    
    # Get model architecture summary
    model_arch = model_builder.get_architecture_summary()
    
    # Prepare training history for serialization
    training_history = {}
    for key, value in history.history.items():
        if isinstance(value, list):
            training_history[key] = [float(v) for v in value]
        else:
            training_history[key] = float(value)
    
    # Save all results
    results_summary = {
        'dataset_info': serialize_for_json(eda_results['dataset_info']),
        'preprocessing_info': serialize_for_json(preprocessor.get_preprocessing_summary()),
        'model_architecture': serialize_for_json(model_arch),
        'training_history': training_history,
        'test_results': serialize_for_json(test_results),
        'cv_results': serialize_for_json(cv_results),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'config': {
            'data_path': config.DATA_PATH,
            'target_column': config.TARGET_COLUMN,
            'test_size': float(config.TEST_SIZE),
            'validation_size': float(config.VALIDATION_SIZE),
            'hidden_layers': config.HIDDEN_LAYERS,
            'epochs': int(config.EPOCHS),
            'batch_size': int(config.BATCH_SIZE),
            'learning_rate': float(config.LEARNING_RATE)
        }
    }
    
    save_results(results_summary, f"{config.REPORT_DIR}results_summary.txt")
    generate_report(results_summary, f"{config.REPORT_DIR}analysis_report.txt")
    
    print("\n" + "="*70)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nGenerated Outputs:")
    print(f"  1. Visualizations: {config.VISUALIZATION_DIR}")
    print(f"  2. Processed Data: {config.PROCESSED_DATA_DIR}")
    print(f"  3. Trained Models: {config.MODEL_DIR}")
    print(f"  4. Results Summary: {config.REPORT_DIR}results_summary.txt")
    print(f"  5. Analysis Report: {config.REPORT_DIR}analysis_report.txt")
    print(f"  6. Training History: {config.VISUALIZATION_DIR}training_history.png")
    print(f"  7. Confusion Matrix: {config.VISUALIZATION_DIR}confusion_matrix.png")
    print(f"  8. All detailed metrics and plots available in output directories")

if __name__ == "__main__":
    main()
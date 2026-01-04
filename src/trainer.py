"""
Task 4: Training Process
"""

import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint,
    ReduceLROnPlateau, TensorBoard
)
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import config

class ModelTrainer:
    """Handles model training with monitoring and callbacks"""
    
    def __init__(self, model, epochs=100, batch_size=32, 
                 learning_rate=0.001, patience=15, model_dir='models/'):
        """
        Initialize trainer
        
        Args:
            model: Compiled Keras model
            epochs: Maximum training epochs
            batch_size: Batch size for training
            learning_rate: Initial learning rate
            patience: Patience for early stopping
            model_dir: Directory to save models
        """
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.model_dir = model_dir
        self.history = None
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        
    def train(self, X_train, y_train, X_val, y_val):
        """Train the model with callbacks and monitoring"""
        print("\nStarting model training...")
        
        # Define callbacks
        callbacks = self._create_callbacks()
        
        # Training parameters
        trainable_params = np.sum([tf.size(w).numpy() for w in self.model.trainable_weights])
        steps_per_epoch = len(X_train) // self.batch_size
        
        print(f"  Training parameters:")
        print(f"    • Epochs: {self.epochs}")
        print(f"    • Batch size: {self.batch_size}")
        print(f"    • Steps per epoch: {steps_per_epoch}")
        print(f"    • Trainable parameters: {trainable_params:,}")
        print(f"    • Training samples: {len(X_train)}")
        print(f"    • Validation samples: {len(X_val)}")
        
        # Train model
        print(f"\n  Training progress:")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Load best model - FIXED FILE EXTENSION
        self.model.load_weights(f"{self.model_dir}best_model.weights.h5")
        print(f"\n  Loaded best model weights")
        
        # Plot training history
        self._plot_training_history()
        
        return self.history
    
    def _create_callbacks(self):
        """Create training callbacks"""
        callbacks = []
        
        # 1. Early Stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.patience,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        )
        callbacks.append(early_stopping)
        print(f"    • Early stopping: patience={self.patience}, monitor='val_loss'")
        
        # 2. Model Checkpoint - FIXED: Use .weights.h5 extension
        checkpoint = ModelCheckpoint(
            filepath=f"{self.model_dir}best_model.weights.h5",  # CHANGED
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True,
            mode='max',
            verbose=0
        )
        callbacks.append(checkpoint)
        print(f"    • Model checkpoint: save best weights to {self.model_dir}best_model.weights.h5")
        
        # 3. Reduce Learning Rate on Plateau
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=self.patience // 2,
            min_lr=1e-6,
            verbose=1,
            mode='min'
        )
        callbacks.append(reduce_lr)
        print(f"    • Reduce LR on plateau: factor=0.5, patience={self.patience//2}")
        
        # 4. TensorBoard (optional)
        try:
            log_dir = f"{self.model_dir}logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
            callbacks.append(tensorboard)
            print(f"    • TensorBoard logging: {log_dir}")
        except:
            print(f"    • TensorBoard: Not available")
        
        return callbacks
    
    def _plot_training_history(self):
        """Plot training history curves"""
        if not self.history:
            return
        
        history = self.history.history
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Loss
        axes[0, 0].plot(history['loss'], label='Training Loss', linewidth=2)
        axes[0, 0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Training and Validation Loss', fontsize=14)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Accuracy
        axes[0, 1].plot(history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[0, 1].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Learning Rate (if available)
        if 'lr' in history:
            axes[1, 0].plot(history['lr'], linewidth=2, color='purple')
            axes[1, 0].set_title('Learning Rate Schedule', fontsize=14)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Learning Rate Constant\n(No reduction triggered)',
                          ha='center', va='center', fontsize=12)
            axes[1, 0].set_title('Learning Rate Schedule', fontsize=14)
        
        # Plot 4: Loss Difference
        loss_diff = np.array(history['loss']) - np.array(history['val_loss'])
        axes[1, 1].plot(loss_diff, linewidth=2, color='red')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].fill_between(range(len(loss_diff)), loss_diff, 0, 
                               where=loss_diff > 0, color='red', alpha=0.3, label='Train > Val')
        axes[1, 1].fill_between(range(len(loss_diff)), loss_diff, 0, 
                               where=loss_diff < 0, color='blue', alpha=0.3, label='Val > Train')
        axes[1, 1].set_title('Training vs Validation Loss Difference', fontsize=14)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss Difference')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add overall statistics
        best_epoch = np.argmin(history['val_loss'])
        best_val_loss = history['val_loss'][best_epoch]
        best_val_acc = history['val_accuracy'][best_epoch]
        
        stats_text = (f"Best Epoch: {best_epoch + 1}\n"
                     f"Best Val Loss: {best_val_loss:.4f}\n"
                     f"Best Val Accuracy: {best_val_acc:.4f}\n"
                     f"Final Train Accuracy: {history['accuracy'][-1]:.4f}\n"
                     f"Final Val Accuracy: {history['val_accuracy'][-1]:.4f}")
        
        fig.text(0.02, 0.02, stats_text, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Training History Analysis', fontsize=16, y=0.98)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f"{config.VISUALIZATION_DIR}training_history.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Training history plots saved to {config.VISUALIZATION_DIR}training_history.png")
    
    def get_training_summary(self):
        """Get summary of training results"""
        if not self.history:
            raise ValueError("Model not trained yet. Call train() first.")
        
        history = self.history.history
        
        # Find best epoch
        best_epoch_val_loss = np.argmin(history['val_loss'])
        best_epoch_val_acc = np.argmax(history['val_accuracy'])
        
        summary = {
            'total_epochs_trained': len(history['loss']),
            'best_epoch_by_val_loss': int(best_epoch_val_loss) + 1,
            'best_epoch_by_val_accuracy': int(best_epoch_val_acc) + 1,
            'final_training_loss': float(history['loss'][-1]),
            'final_training_accuracy': float(history['accuracy'][-1]),
            'final_validation_loss': float(history['val_loss'][-1]),
            'final_validation_accuracy': float(history['val_accuracy'][-1]),
            'best_validation_loss': float(history['val_loss'][best_epoch_val_loss]),
            'best_validation_accuracy': float(history['val_accuracy'][best_epoch_val_acc]),
            'overfitting_indicator': {
                'train_val_loss_gap': history['loss'][-1] - history['val_loss'][-1],
                'train_val_acc_gap': history['val_accuracy'][-1] - history['accuracy'][-1],
                'early_stopping_triggered': len(history['loss']) < self.epochs
            }
        }
        
        # Analyze learning rate changes if available
        if 'lr' in history:
            summary['learning_rate_changes'] = {
                'initial_lr': float(history['lr'][0]),
                'final_lr': float(history['lr'][-1]),
                'lr_reductions': len(np.unique(history['lr'])) > 1
            }
        
        return summary
    
    def explain_training_strategy(self):
        """Explain the training strategy and choices"""
        explanation = """
        TRAINING STRATEGY RATIONALE
        ===========================
        
        1. OPTIMIZER SELECTION:
           • Adam Optimizer: Combines benefits of AdaGrad and RMSProp
           • Adaptive learning rates per parameter
           • Good for problems with sparse gradients
           • Default β1=0.9, β2=0.999, ε=1e-7
        
        2. BATCH SIZE ({batch_size}):
           • Balance between computational efficiency and gradient estimation
           • Smaller batches provide more frequent updates
           • Larger batches provide better gradient estimates
           • Chosen based on dataset size ({train_samples} samples)
        
        3. EPOCHS & EARLY STOPPING:
           • Maximum epochs: {epochs}
           • Early stopping patience: {patience} epochs
           • Prevents overfitting by stopping when validation loss stops improving
           • Restores best weights automatically
        
        4. LEARNING RATE SCHEDULE:
           • Initial rate: {learning_rate}
           • Reduce on Plateau: Halves LR when validation loss plateaus
           • Minimum LR: 1e-6 (prevents too small updates)
           • Patience: {lr_patience} epochs before reduction
        
        5. OVERFITTING PREVENTION:
           • Dropout ({dropout_rate:.0%}): Random neuron deactivation
           • L2 Regularization (λ={l2_reg}): Weight decay
           • Early Stopping: Prevents memorization
           • Batch Normalization: Redoves internal covariate shift
        
        6. VALIDATION STRATEGY:
           • 15% of data held out for validation
           • Stratified splitting maintains class distribution
           • Monitored for both loss and accuracy
        
        7. MODEL CHECKPOINTING:
           • Saves best model weights based on validation accuracy
           • Automatic restoration after training
           • Ensures best model is used for evaluation
        """.format(
            batch_size=self.batch_size,
            train_samples=self.history.params['samples'] if self.history else 'unknown',
            epochs=self.epochs,
            patience=self.patience,
            learning_rate=self.learning_rate,
            lr_patience=self.patience // 2,
            dropout_rate=0.3,  # Default, adjust based on your model
            l2_reg=0.001
        )
        
        return explanation
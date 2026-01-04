"""
Task 3: Model Design
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization,
    Input, Activation
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import numpy as np
import config

class ModelBuilder:
    """Design and build neural network architecture"""
    
    def __init__(self, input_dim, num_classes, 
                 hidden_layers=[64, 32, 16], 
                 dropout_rate=0.3, 
                 l2_reg=0.001):
        """
        Initialize model builder
        
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes
            hidden_layers: List of neurons in each hidden layer
            dropout_rate: Dropout rate for regularization
            l2_reg: L2 regularization strength
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.model = None
        
    def build_model(self):
        """Build and compile the neural network model"""
        print("\nBuilding neural network model...")
        
        model = Sequential()
        
        # Input layer
        model.add(Input(shape=(self.input_dim,)))
        print(f"  Input layer: {self.input_dim} features")
        
        # Hidden layers
        for i, neurons in enumerate(self.hidden_layers):
            # Dense layer with L2 regularization
            model.add(Dense(
                neurons,
                kernel_regularizer=l2(self.l2_reg),
                bias_regularizer=l2(self.l2_reg),
                name=f"dense_{i+1}"
            ))
            
            # Batch normalization (except for last layer before output)
            if i < len(self.hidden_layers) - 1:
                model.add(BatchNormalization(name=f"batch_norm_{i+1}"))
            
            # Activation function (ReLU for hidden layers)
            model.add(Activation('relu', name=f"activation_{i+1}"))
            
            # Dropout for regularization
            model.add(Dropout(self.dropout_rate, name=f"dropout_{i+1}"))
            
            print(f"  Hidden layer {i+1}: {neurons} neurons, "
                  f"Dropout={self.dropout_rate}, L2={self.l2_reg}")
        
        # Output layer
        if self.num_classes == 2:
            # Binary classification
            model.add(Dense(1, activation='sigmoid', name='output'))
            loss = 'binary_crossentropy'
            print(f"  Output layer: 1 neuron, sigmoid activation (binary classification)")
        else:
            # Multiclass classification
            model.add(Dense(self.num_classes, activation='softmax', name='output'))
            loss = 'sparse_categorical_crossentropy'  # For integer labels
            print(f"  Output layer: {self.num_classes} neurons, "
                  f"softmax activation ({self.num_classes}-class classification)")
        
        # Compile model
        optimizer = Adam(learning_rate=config.LEARNING_RATE)
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def get_architecture_summary(self):
        """Get detailed summary of model architecture"""
        if not self.model:
            raise ValueError("Model not built yet. Call build_model() first.")
        
        # Count total parameters
        total_params = self.model.count_params()
        trainable_params = np.sum([tf.size(w).numpy() for w in self.model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        # Get model summary as string
        summary_list = []
        self.model.summary(print_fn=lambda x: summary_list.append(x))
        summary_str = "\n".join(summary_list)
        
        # Get layer information (simplified to avoid serialization issues)
        layer_details = []
        for i, layer in enumerate(self.model.layers):
            layer_info = {
                'index': i,
                'name': layer.name,
                'type': layer.__class__.__name__,
                'num_params': layer.count_params(),
                'built': bool(layer.built)
            }
            layer_details.append(layer_info)
        
        summary = {
            'input_dimension': self.input_dim,
            'num_classes': self.num_classes,
            'hidden_layers': self.hidden_layers,
            'total_layers': len(self.model.layers),
            'total_params': total_params,
            'trainable_params': trainable_params,
            'non_trainable_params': non_trainable_params,
            'dropout_rate': float(self.dropout_rate),
            'l2_regularization': float(self.l2_reg),
            'loss_function': str(self.model.loss),
            'optimizer': str(self.model.optimizer.__class__.__name__),
            'learning_rate': float(self.model.optimizer.learning_rate.numpy()),
            'summary_string': summary_str,
            'layer_details': layer_details
        }
        
        return summary
    
    def save_architecture_diagram(self, filepath):
        """Save model architecture diagram"""
        try:
            tf.keras.utils.plot_model(
                self.model,
                to_file=filepath,
                show_shapes=True,
                show_layer_names=True,
                rankdir='TB',
                dpi=96
            )
            print(f"Model architecture diagram saved to {filepath}")
        except ImportError:
            print("Graphviz not installed. Skipping architecture diagram.")
            
    def explain_design_choices(self):
        """Explain the rationale behind model design choices"""
        explanation = """
        MODEL DESIGN RATIONALE
        =====================
        
        1. INPUT LAYER:
           • Size: Matches the number of features after preprocessing
           • No activation: Raw input is passed to first hidden layer
        
        2. HIDDEN LAYERS:
           • Architecture: {}-layer network with {} neurons
           • Depth: Chosen to balance model capacity and overfitting risk
           • Width: Decreasing neuron count (64→32→16) helps learn hierarchical features
           • Activation: ReLU used for:
             - Sparsity (helps with feature selection)
             - Mitigates vanishing gradient problem
             - Computationally efficient
        
        3. REGULARIZATION:
           • Dropout ({:.0%}): Randomly drops neurons during training to prevent co-adaptation
           • L2 Regularization (λ={}): Penalizes large weights to prevent overfitting
           • Batch Normalization: Normalizes layer outputs to stabilize training
        
        4. OUTPUT LAYER:
           • {} neurons with {} activation
           • Chosen based on {} classification problem
        
        5. OPTIMIZATION:
           • Adam optimizer: Adaptive learning rate, good for sparse gradients
           • Learning rate: {} (balanced for stable convergence)
        
        6. LOSS FUNCTION:
           • {}: Appropriate for {} classification
        """.format(
            len(self.hidden_layers),
            '→'.join(map(str, self.hidden_layers)),
            self.dropout_rate,
            self.l2_reg,
            self.num_classes if self.num_classes > 2 else 1,
            'softmax' if self.num_classes > 2 else 'sigmoid',
            'multi-class' if self.num_classes > 2 else 'binary',
            config.LEARNING_RATE,
            self.model.loss,
            'multi-class' if self.num_classes > 2 else 'binary'
        )
        
        return explanation
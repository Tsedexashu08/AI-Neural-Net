# Data settings
DATA_PATH = "data/CrashData.xlsx"  
TARGET_COLUMN = "Accident Type"      
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15
RANDOM_STATE = 42

# Model architecture
HIDDEN_LAYERS = [64, 32, 16]  # Neurons in each hidden layer
DROPOUT_RATE = 0.3
L2_REGULARIZATION = 0.001

# Training parameters
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 15

# Evaluation
K_FOLDS = 5  # For cross-validation

# Output directories
VISUALIZATION_DIR = "outputs/visualizations/"
MODEL_DIR = "outputs/models/"
PROCESSED_DATA_DIR = "outputs/processed_data/"
REPORT_DIR = "reports/"
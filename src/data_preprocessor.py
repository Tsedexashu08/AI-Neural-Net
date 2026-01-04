"""
Task 2: Data Preparation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, LabelEncoder, 
    OneHotEncoder, OrdinalEncoder
)
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
warnings.filterwarnings('ignore')
import config

class DataPreprocessor:
    """Handles all data preparation tasks"""
    
    def __init__(self, target_column, random_state=42):
        self.target_column = target_column
        self.random_state = random_state
        
        # Initialize transformers
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='median')  # Default, can change
        self.onehot_encoder = OneHotEncoder(sparse_output=False, drop='first')
        self.ordinal_encoder = OrdinalEncoder()
        
        # Store metadata
        self.preprocessing_info = {}
        self.feature_names = None
        
    def prepare_data(self, df):
        """Main method to prepare data for modeling"""
        print("\nStarting data preparation...")
        
        # Step 1: Create a copy
        df_clean = df.copy()
        
        # Step 2: Handle missing values
        df_clean = self._handle_missing_values(df_clean)
        
        # Step 3: Handle duplicates
        df_clean = self._handle_duplicates(df_clean)
        
        # Step 4: Handle outliers (optional)
        df_clean = self._handle_outliers(df_clean)
        
        # Step 5: Separate features and target
        X = df_clean.drop(columns=[self.target_column])
        y = df_clean[self.target_column]
        
        # Step 6: Encode categorical features
        X_encoded, encoding_info = self._encode_categorical_features(X)
        self.preprocessing_info['encoding'] = encoding_info
        
        # Step 7: Scale features
        X_scaled, scaling_info = self._scale_features(X_encoded)
        self.preprocessing_info['scaling'] = scaling_info
        
        # Step 8: Encode target variable
        y_encoded, target_info = self._encode_target(y)
        self.preprocessing_info['target_encoding'] = target_info
        
        # Step 9: Split data
        splits = self._split_data(X_scaled, y_encoded)
        
        # Step 10: Store feature names
        self.feature_names = X_scaled.columns.tolist() if hasattr(X_scaled, 'columns') else None
        
        print("✓ Data preparation completed")
        
        return splits
    
    def _handle_missing_values(self, df):
        """Handle missing values with appropriate strategies"""
        print("  Handling missing values...")
        
        missing_before = df.isnull().sum().sum()
        
        if missing_before == 0:
            print("    No missing values found")
            self.preprocessing_info['missing_values'] = {
                'strategy': 'none',
                'count_before': 0,
                'count_after': 0
            }
            return df
        
        # Analyze missing patterns
        missing_cols = df.columns[df.isnull().any()].tolist()
        missing_info = {}
        
        for col in missing_cols:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            col_type = df[col].dtype
            
            missing_info[col] = {
                'count': missing_count,
                'percentage': missing_pct,
                'type': col_type
            }
        
        # Apply imputation strategies
        for col, info in missing_info.items():
            if info['percentage'] > 30:  # If more than 30% missing
                # Consider dropping column
                print(f"    Dropping column '{col}' ({info['percentage']:.1f}% missing)")
                df = df.drop(columns=[col])
            else:
                # Apply appropriate imputation
                if np.issubdtype(info['type'], np.number):
                    # Numeric column: use median
                    imputed_value = df[col].median()
                    df[col] = df[col].fillna(imputed_value)
                    print(f"    Filled '{col}' with median: {imputed_value:.4f}")
                else:
                    # Categorical column: use mode
                    imputed_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                    df[col] = df[col].fillna(imputed_value)
                    print(f"    Filled '{col}' with mode: {imputed_value}")
        
        missing_after = df.isnull().sum().sum()
        
        self.preprocessing_info['missing_values'] = {
            'strategy': 'column_specific',
            'count_before': missing_before,
            'count_after': missing_after,
            'columns_handled': missing_info
        }
        
        return df
    
    def _handle_duplicates(self, df):
        """Remove duplicate rows"""
        duplicate_count = df.duplicated().sum()
        
        if duplicate_count > 0:
            print(f"  Removing {duplicate_count} duplicate rows")
            df = df.drop_duplicates()
        
        self.preprocessing_info['duplicates'] = {
            'removed_count': duplicate_count,
            'strategy': 'drop_all'
        }
        
        return df
    
    def _handle_outliers(self, df):
        """Optional: Handle outliers using IQR method"""
        print("  Checking for outliers...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        outlier_info = {}
        outlier_count = 0
        
        for col in numeric_cols:
            if col == self.target_column:
                continue
                
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_count_col = len(outliers)
            
            if outlier_count_col > 0:
                # Cap outliers instead of removing
                df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
                df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
                
                outlier_info[col] = {
                    'count': outlier_count_col,
                    'percentage': (outlier_count_col / len(df)) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
                
                outlier_count += outlier_count_col
        
        self.preprocessing_info['outliers'] = {
            'handled_count': outlier_count,
            'strategy': 'capping_iqr',
            'columns': outlier_info
        }
        
        if outlier_count > 0:
            print(f"    Capped outliers in {len(outlier_info)} features")
        
        return df
    
    def _encode_categorical_features(self, X):
        """Encode categorical variables for neural network"""
        print("  Encoding categorical features...")
        
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        encoding_info = {}
        
        if not categorical_cols:
            print("    No categorical features to encode")
            return X, {'strategy': 'none'}
        
        X_encoded = X.copy()
        
        for col in categorical_cols:
            unique_count = X[col].nunique()
            
            if unique_count == 2:
                # Binary categorical: use label encoding
                X_encoded[col] = self.label_encoder.fit_transform(X_encoded[col])
                encoding_info[col] = {
                    'type': 'binary',
                    'encoding': 'label',
                    'mapping': dict(zip(self.label_encoder.classes_, 
                                       range(len(self.label_encoder.classes_)))),
                    'unique_count': unique_count
                }
                print(f"    Label encoded binary feature: {col}")
                
            elif unique_count <= 10:
                # Low cardinality: use one-hot encoding
                # Get dummies with proper naming
                dummies = pd.get_dummies(X_encoded[col], prefix=col, drop_first=True)
                
                # Drop original column and add dummies
                X_encoded = X_encoded.drop(columns=[col])
                X_encoded = pd.concat([X_encoded, dummies], axis=1)
                
                encoding_info[col] = {
                    'type': 'categorical',
                    'encoding': 'one_hot',
                    'unique_count': unique_count,
                    'created_columns': dummies.columns.tolist()
                }
                print(f"    One-hot encoded feature: {col} → {len(dummies.columns)} columns")
                
            else:
                # High cardinality: use ordinal encoding or frequency encoding
                # Frequency encoding: replace with frequency of each category
                freq = X_encoded[col].value_counts(normalize=True)
                X_encoded[col] = X_encoded[col].map(freq)
                
                encoding_info[col] = {
                    'type': 'high_cardinality',
                    'encoding': 'frequency',
                    'unique_count': unique_count
                }
                print(f"    Frequency encoded high-cardinality feature: {col}")
        
        return X_encoded, encoding_info
    
    def _scale_features(self, X):
        """Scale features using StandardScaler"""
        print("  Scaling features...")
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            print("    No numeric features to scale")
            return X, {'strategy': 'none'}
        
        X_scaled = X.copy()
        
        # Fit scaler on numeric columns
        self.scaler.fit(X_scaled[numeric_cols])
        
        # Transform
        X_scaled[numeric_cols] = self.scaler.transform(X_scaled[numeric_cols])
        
        scaling_info = {
            'strategy': 'standard_scaler',
            'features_scaled': numeric_cols,
            'scaler_params': {
                'mean': self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else [],
                'scale': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else []
            }
        }
        
        print(f"    Scaled {len(numeric_cols)} numeric features using StandardScaler")
        
        return X_scaled, scaling_info
    
    def _encode_target(self, y):
        """Encode target variable"""
        print("  Encoding target variable...")
        
        # Check if target is already numeric
        if np.issubdtype(y.dtype, np.number):
            print("    Target is already numeric")
            target_info = {
                'encoding': 'none',
                'type': 'numeric',
                'unique_values': len(np.unique(y))
            }
            return y.values, target_info
        
        # Encode categorical target
        y_encoded = self.label_encoder.fit_transform(y)
        
        target_info = {
            'encoding': 'label_encoding',
            'type': 'categorical',
            'unique_classes': len(self.label_encoder.classes_),
            'class_names': self.label_encoder.classes_.tolist(),
            'mapping': dict(zip(self.label_encoder.classes_, 
                              range(len(self.label_encoder.classes_))))
        }
        
        print(f"    Encoded target variable into {len(self.label_encoder.classes_)} classes")
        print(f"    Classes: {self.label_encoder.classes_.tolist()}")
        
        return y_encoded, target_info
    
    def _split_data(self, X, y):
        """Split data into train, validation, and test sets"""
        print("  Splitting data...")
        
        # First split: train + validation vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=config.TEST_SIZE,
            random_state=self.random_state,
            stratify=y if len(np.unique(y)) > 1 else None
        )
        
        # Second split: train vs validation
        val_size = config.VALIDATION_SIZE / (1 - config.TEST_SIZE)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            random_state=self.random_state,
            stratify=y_temp if len(np.unique(y_temp)) > 1 else None
        )
        
        # Calculate class distribution in each set
        train_dist = np.bincount(y_train) / len(y_train) * 100
        val_dist = np.bincount(y_val) / len(y_val) * 100
        test_dist = np.bincount(y_test) / len(y_test) * 100
        
        print(f"    Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"    Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"    Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        # Print class distribution
        if hasattr(self.label_encoder, 'classes_'):
            for i, class_name in enumerate(self.label_encoder.classes_):
                print(f"      Class '{class_name}': "
                      f"Train={train_dist[i]:.1f}%, "
                      f"Val={val_dist[i]:.1f}%, "
                      f"Test={test_dist[i]:.1f}%")
        
        self.preprocessing_info['data_split'] = {
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'train_percentage': len(X_train)/len(X)*100,
            'val_percentage': len(X_val)/len(X)*100,
            'test_percentage': len(X_test)/len(X)*100,
            'stratified': True,
            'random_state': self.random_state
        }
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'X_train_full': X_temp,  # For cross-validation
            'y_train_full': y_temp,   # For cross-validation
            'class_names': self.label_encoder.classes_ if hasattr(self.label_encoder, 'classes_') else None
        }
    
    def get_preprocessing_summary(self):
        """Return summary of preprocessing steps"""
        return self.preprocessing_info
    # In src/data_preprocessor.py, add this method to the DataPreprocessor class:

def _ensure_numeric(self, X):
    """Ensure all features are numeric"""
    print("  Ensuring all features are numeric...")
    
    X_numeric = X.copy()
    
    # Check for non-numeric columns
    non_numeric_cols = X_numeric.select_dtypes(include=['object']).columns.tolist()
    
    if non_numeric_cols:
        print(f"    Found {len(non_numeric_cols)} non-numeric columns: {non_numeric_cols}")
        
        for col in non_numeric_cols:
            # Try to convert to numeric
            try:
                X_numeric[col] = pd.to_numeric(X_numeric[col], errors='coerce')
                print(f"    Converted '{col}' to numeric")
            except:
                # If conversion fails, use label encoding
                print(f"    Could not convert '{col}' to numeric, using label encoding")
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                X_numeric[col] = le.fit_transform(X_numeric[col].fillna('missing'))
    
    # Check for any remaining object dtype
    remaining_objects = X_numeric.select_dtypes(include=['object']).columns.tolist()
    if remaining_objects:
        print(f"    Warning: {len(remaining_objects)} columns still have object dtype")
        print(f"    Columns: {remaining_objects}")
    
    # Fill any NaN values that resulted from conversion
    X_numeric = X_numeric.fillna(0)
    
    return X_numeric
"""
Task 1: Data Understanding (Exploratory Data Analysis)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import config

class DataLoader:
    """Handles data loading, exploration, and initial analysis"""
    
    def __init__(self, data_path, target_column):
        self.data_path = data_path
        self.target_column = target_column
        self.df = None
        self.numeric_cols = None
        self.categorical_cols = None
        
        # Set visualization style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Create visualization directory
        os.makedirs(config.VISUALIZATION_DIR, exist_ok=True)
        os.makedirs(config.REPORT_DIR, exist_ok=True)
        
    def load_data(self):
        """Load and return the dataset"""
        print("Loading dataset...")
        
        # Read Excel file (supports .xlsx, .xls)
        if self.data_path.endswith('.xlsx') or self.data_path.endswith('.xls'):
            self.df = pd.read_excel(self.data_path)
        else:
            # Try to load as CSV if not Excel
            try:
                self.df = pd.read_csv(self.data_path)
            except:
                raise ValueError("Please provide an Excel (.xlsx, .xls) or CSV file")
        
        print(f"Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        
        # Identify column types
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target from feature lists if present
        if self.target_column in self.numeric_cols:
            self.numeric_cols.remove(self.target_column)
        if self.target_column in self.categorical_cols:
            self.categorical_cols.remove(self.target_column)
        
        return self.df
    
    def perform_eda(self):
        """Perform comprehensive exploratory data analysis"""
        print("\nPerforming Exploratory Data Analysis...")
        
        results = {
            'dataset_info': self._get_dataset_info(),
            'missing_values': self._analyze_missing_values(),
            'duplicates': self._find_duplicates(),
            'summary_stats': self._generate_summary_statistics(),
            'class_distribution': self._analyze_class_distribution(),
            'correlation_analysis': self._analyze_correlations()
        }
        
        # Generate all visualizations
        self._create_visualizations()
        
        # Print key insights
        self._print_key_insights(results)
        
        return results
    
    def _get_dataset_info(self):
        """Get basic dataset information"""
        info = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'data_types': str(self.df.dtypes.to_dict()),
            'numeric_features': self.numeric_cols,
            'categorical_features': self.categorical_cols,
            'memory_usage_MB': self.df.memory_usage(deep=True).sum() / (1024**2)
        }
        
        # Save to file
        with open(f"{config.REPORT_DIR}dataset_info.txt", 'w') as f:
            f.write("DATASET INFORMATION\n")
            f.write("="*50 + "\n")
            f.write(f"Shape: {info['shape']}\n")
            f.write(f"Memory Usage: {info['memory_usage_MB']:.2f} MB\n")
            f.write(f"Numeric Features: {len(info['numeric_features'])}\n")
            f.write(f"Categorical Features: {len(info['categorical_features'])}\n")
            f.write(f"\nColumns and Data Types:\n")
            for col in self.df.columns:
                f.write(f"  {col}: {self.df[col].dtype}\n")
        
        return info
    
    def _analyze_missing_values(self):
        """Analyze and report missing values"""
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'missing_count': missing,
            'missing_percentage': missing_pct
        }).sort_values('missing_count', ascending=False)
        
        # Save missing values report
        missing_df[missing_df['missing_count'] > 0].to_csv(
            f"{config.REPORT_DIR}missing_values_report.csv"
        )
        
        return {
            'columns_with_missing': missing_df[missing_df['missing_count'] > 0].index.tolist(),
            'total_missing': int(missing.sum()),
            'missing_details': missing_df[missing_df['missing_count'] > 0].to_dict()
        }
    
    def _find_duplicates(self):
        """Find and report duplicate rows"""
        duplicate_count = self.df.duplicated().sum()
        duplicate_pct = (duplicate_count / len(self.df)) * 100
        
        duplicates_info = {
            'count': int(duplicate_count),
            'percentage': float(duplicate_pct)
        }
        
        # Save duplicates report
        with open(f"{config.REPORT_DIR}duplicates_report.txt", 'w') as f:
            f.write("DUPLICATE ROWS ANALYSIS\n")
            f.write("="*50 + "\n")
            f.write(f"Total Duplicate Rows: {duplicate_count}\n")
            f.write(f"Percentage: {duplicate_pct:.2f}%\n")
        
        return duplicates_info
    
    def _generate_summary_statistics(self):
        """Generate summary statistics for all features"""
        print("\nGenerating summary statistics...")
        
        # Numeric features summary
        if self.numeric_cols:
            numeric_df = self.df[self.numeric_cols]
            numeric_summary = numeric_df.describe().T
            
            # Calculate skewness and kurtosis
            try:
                numeric_summary['skewness'] = numeric_df.apply(stats.skew)
                numeric_summary['kurtosis'] = numeric_df.apply(stats.kurtosis)
                print("    Added skewness and kurtosis calculations")
            except Exception as e:
                print(f"    Note: Could not calculate skewness/kurtosis: {e}")
                numeric_summary['skewness'] = np.nan
                numeric_summary['kurtosis'] = np.nan
        else:
            numeric_summary = pd.DataFrame()
            print("    No numeric features found")
        
        # Categorical features summary
        categorical_summary = {}
        for col in self.categorical_cols:
            value_counts = self.df[col].value_counts()
            categorical_summary[col] = {
                'unique_count': self.df[col].nunique(),
                'top_category': value_counts.index[0] if len(value_counts) > 0 else None,
                'top_count': value_counts.iloc[0] if len(value_counts) > 0 else 0
            }
        
        # Target variable summary
        target_summary = {}
        if self.target_column in self.df.columns:
            if self.df[self.target_column].dtype == 'object':
                target_counts = self.df[self.target_column].value_counts()
                target_summary = {
                    'type': 'categorical',
                    'unique_count': len(target_counts),
                    'distribution': target_counts.to_dict()
                }
            else:
                target_desc = self.df[self.target_column].describe()
                target_summary = {
                    'type': 'numeric',
                    'summary': target_desc.to_dict()
                }
        
        # Save to files
        if not numeric_summary.empty:
            numeric_summary.to_csv(f"{config.REPORT_DIR}numeric_summary.csv")
            print(f"    Numeric summary saved to {config.REPORT_DIR}numeric_summary.csv")
        
        if categorical_summary:
            with open(f"{config.REPORT_DIR}categorical_summary.txt", 'w') as f:
                f.write("CATEGORICAL FEATURES SUMMARY\n")
                f.write("="*50 + "\n")
                for col, col_stats in categorical_summary.items():
                    f.write(f"\n{col}:\n")
                    f.write(f"  Unique values: {col_stats['unique_count']}\n")
                    f.write(f"  Most common: '{col_stats['top_category']}' "
                           f"(count: {col_stats['top_count']})\n")
            print(f"    Categorical summary saved to {config.REPORT_DIR}categorical_summary.txt")
        
        return {
            'numeric': numeric_summary.to_dict() if not numeric_summary.empty else {},
            'categorical': categorical_summary,
            'target': target_summary
        }
    
    def _analyze_class_distribution(self):
        """Analyze distribution of target variable"""
        if self.target_column not in self.df.columns:
            print(f"Warning: Target column '{self.target_column}' not found in dataset")
            return {}
        
        class_dist = self.df[self.target_column].value_counts()
        class_pct = (class_dist / len(self.df)) * 100
        
        distribution_df = pd.DataFrame({
            'count': class_dist,
            'percentage': class_pct
        })
        
        # Save class distribution
        distribution_df.to_csv(f"{config.REPORT_DIR}class_distribution.csv")
        
        # Return in a consistent format
        result = {}
        for idx, (class_name, row) in enumerate(distribution_df.iterrows()):
            result[str(class_name)] = {
                'count': int(row['count']),
                'percentage': float(row['percentage'])
            }
        
        return result
    
    def _analyze_correlations(self):
        """Analyze correlations between features"""
        if len(self.numeric_cols) > 1:
            correlation_matrix = self.df[self.numeric_cols].corr()
            
            # Get top correlations
            corr_pairs = correlation_matrix.unstack()
            sorted_corr = corr_pairs.sort_values(ascending=False, key=abs)
            top_correlations = sorted_corr[sorted_corr != 1].head(20)
            
            # Save correlation matrix
            correlation_matrix.to_csv(f"{config.REPORT_DIR}correlation_matrix.csv")
            
            # Save top correlations
            pd.DataFrame(top_correlations).to_csv(
                f"{config.REPORT_DIR}top_correlations.csv"
            )
            
            return {
                'has_correlations': True,
                'top_correlations_count': len(top_correlations)
            }
        return {'has_correlations': False}
    
    def _create_visualizations(self):
        """Create all required visualizations"""
        print("\nCreating visualizations...")
        
        # 1. Feature Distributions (Histograms)
        self._plot_feature_distributions()
        
        # 2. Boxplots for outlier detection
        self._plot_boxplots()
        
        # 3. Correlation Heatmap
        if len(self.numeric_cols) > 1:
            self._plot_correlation_heatmap()
        
        # 4. Class Distribution
        if self.target_column in self.df.columns:
            self._plot_class_distribution()
        
        print(f"Visualizations saved to {config.VISUALIZATION_DIR}")
    
    def _plot_feature_distributions(self):
        """Plot histograms for all numeric features"""
        if not self.numeric_cols:
            print("    No numeric features to plot")
            return
        
        n_cols = 3
        n_features = len(self.numeric_cols)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows*4))
        
        # Flatten axes if multiple rows
        if n_rows > 1:
            axes = axes.flatten()
        elif n_features == 1:
            axes = [axes]
        
        for idx, col in enumerate(self.numeric_cols):
            if idx < len(axes):
                ax = axes[idx]
                # Histogram with KDE
                sns.histplot(data=self.df, x=col, kde=True, ax=ax, bins=30)
                ax.set_title(f'Distribution of {col}', fontsize=12)
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
                
                # Add skewness and kurtosis
                try:
                    col_data = self.df[col].dropna()
                    if len(col_data) > 0:
                        skew_val = stats.skew(col_data)
                        kurt_val = stats.kurtosis(col_data)
                        ax.text(0.02, 0.95, f'Skew: {skew_val:.2f}\nKurt: {kurt_val:.2f}',
                               transform=ax.transAxes, fontsize=9,
                               verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                except:
                    pass  # Skip if stats calculation fails
        
        # Remove empty subplots
        for idx in range(len(self.numeric_cols), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.savefig(f"{config.VISUALIZATION_DIR}feature_distributions.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    Feature distributions plot saved")
    
    def _plot_boxplots(self):
        """Create boxplots for outlier detection"""
        if not self.numeric_cols:
            print("    No numeric features for boxplots")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Boxplot for all numeric features
        df_numeric = self.df[self.numeric_cols]
        sns.boxplot(data=df_numeric, ax=axes[0])
        axes[0].set_title('Boxplot of All Numeric Features', fontsize=14)
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
        axes[0].set_ylabel('Value')
        
        # Identify outliers
        Q1 = df_numeric.quantile(0.25)
        Q3 = df_numeric.quantile(0.75)
        IQR = Q3 - Q1
        
        outliers = ((df_numeric < (Q1 - 1.5 * IQR)) | 
                   (df_numeric > (Q3 + 1.5 * IQR))).sum()
        
        # Bar plot of outlier counts
        outliers_sorted = outliers.sort_values(ascending=False)
        sns.barplot(x=outliers_sorted.values, y=outliers_sorted.index, ax=axes[1])
        axes[1].set_title('Number of Outliers per Feature', fontsize=14)
        axes[1].set_xlabel('Number of Outliers')
        axes[1].set_ylabel('Feature')
        
        plt.tight_layout()
        plt.savefig(f"{config.VISUALIZATION_DIR}boxplots_outliers.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    Boxplots saved")
    
    def _plot_correlation_heatmap(self):
        """Plot correlation matrix heatmap"""
        if len(self.numeric_cols) <= 1:
            print("    Not enough numeric features for correlation heatmap")
            return
        
        correlation_matrix = self.df[self.numeric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm',
                   center=0, fmt='.2f', linewidths=0.5,
                   cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Matrix', fontsize=16, pad=20)
        
        plt.tight_layout()
        plt.savefig(f"{config.VISUALIZATION_DIR}correlation_heatmap.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    Correlation heatmap saved")
    
    def _plot_class_distribution(self):
        """Plot class distribution for target variable"""
        if self.target_column not in self.df.columns:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Count plot
        target_counts = self.df[self.target_column].value_counts()
        
        # For categorical targets
        if self.df[self.target_column].dtype == 'object':
            sns.barplot(x=target_counts.index.astype(str), y=target_counts.values, ax=axes[0])
            axes[0].set_title(f'Class Distribution of {self.target_column}', fontsize=14)
            axes[0].set_xlabel(self.target_column)
            axes[0].set_ylabel('Count')
            
            # Add count labels on bars
            for i, count in enumerate(target_counts.values):
                axes[0].text(i, count + max(target_counts.values)*0.01, 
                           str(count), ha='center', va='bottom')
            
            # Pie chart
            wedges, texts, autotexts = axes[1].pie(
                target_counts.values, 
                labels=target_counts.index.astype(str),
                autopct='%1.1f%%',
                startangle=90,
                colors=sns.color_palette("husl", len(target_counts))
            )
            
            axes[1].set_title(f'Class Proportion of {self.target_column}', fontsize=14)
            
            # Make autopct text larger
            for autotext in autotexts:
                autotext.set_fontsize(11)
                autotext.set_weight('bold')
        else:
            # For numeric targets, use histogram
            axes[0].hist(self.df[self.target_column], bins=30, edgecolor='black')
            axes[0].set_title(f'Distribution of {self.target_column}', fontsize=14)
            axes[0].set_xlabel(self.target_column)
            axes[0].set_ylabel('Frequency')
            
            # KDE plot
            sns.kdeplot(data=self.df, x=self.target_column, ax=axes[1])
            axes[1].set_title(f'Density of {self.target_column}', fontsize=14)
            axes[1].set_xlabel(self.target_column)
            axes[1].set_ylabel('Density')
        
        plt.tight_layout()
        plt.savefig(f"{config.VISUALIZATION_DIR}class_distribution.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    Class distribution plot saved")
    
    def _print_key_insights(self, results):
        """Print key insights from EDA"""
        print("\n" + "="*50)
        print("KEY INSIGHTS FROM EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Dataset overview
        print(f"\n1. DATASET OVERVIEW:")
        print(f"   • {self.df.shape[0]} samples, {self.df.shape[1]} features")
        print(f"   • {len(self.numeric_cols)} numeric features")
        print(f"   • {len(self.categorical_cols)} categorical features")
        
        # Missing values
        missing_data = results['missing_values']
        if missing_data['columns_with_missing']:
            print(f"\n2. MISSING VALUES: {missing_data['total_missing']} total missing values")
            for col in missing_data['columns_with_missing'][:5]:  # Show first 5
                print(f"   • {col}: has missing values")
            if len(missing_data['columns_with_missing']) > 5:
                print(f"   • ... and {len(missing_data['columns_with_missing']) - 5} more columns")
        else:
            print(f"\n2. MISSING VALUES: None found")
        
        # Duplicates
        duplicates = results['duplicates']
        if duplicates['count'] > 0:
            print(f"\n3. DUPLICATES: {duplicates['count']} duplicate rows ({duplicates['percentage']:.2f}%)")
        else:
            print(f"\n3. DUPLICATES: No duplicates found")
        
        # Class distribution - FIXED SECTION
        class_dist_data = results['class_distribution']
        if class_dist_data:
            print(f"\n4. CLASS DISTRIBUTION:")
            # Handle the corrected data structure
            for class_name, stats in class_dist_data.items():
                if isinstance(stats, dict):
                    count = stats.get('count', 0)
                    pct = stats.get('percentage', 0)
                    print(f"   • Class {class_name}: {count} samples ({pct:.1f}%)")
                else:
                    # Fallback for unexpected format
                    print(f"   • Class {class_name}: {stats}")
        else:
            print(f"\n4. CLASS DISTRIBUTION: Not available or target column not found")
        
        print(f"\n5. PREPROCESSING REQUIREMENTS:")
        
        # Check numeric features for scaling needs
        if self.numeric_cols:
            print(f"   • Numeric features ({len(self.numeric_cols)}): Need scaling for neural network")
            # Check ranges of first few numeric features
            for col in self.numeric_cols[:3]:  # Show first 3
                if col in self.df.columns:
                    min_val = self.df[col].min()
                    max_val = self.df[col].max()
                    range_val = max_val - min_val
                    print(f"     - {col}: range [{min_val:.2f}, {max_val:.2f}]")
                    if range_val > 100:  # Arbitrary threshold
                        print(f"       → Definitely needs scaling")
        
        # Check categorical features for encoding needs
        if self.categorical_cols:
            print(f"   • Categorical features ({len(self.categorical_cols)}): Need encoding")
            for col in self.categorical_cols[:3]:  # Show first 3
                if col in self.df.columns:
                    unique_count = self.df[col].nunique()
                    print(f"     - {col}: {unique_count} unique categories")
                    if unique_count > 10:
                        print(f"       → High cardinality - consider embedding or hashing")
                    elif unique_count == 2:
                        print(f"       → Binary - use label encoding")
                    else:
                        print(f"       → Use one-hot encoding")
        
        print(f"\n6. VISUALIZATIONS GENERATED:")
        print(f"   • Feature distributions")
        print(f"   • Boxplots and outlier detection")
        if len(self.numeric_cols) > 1:
            print(f"   • Correlation heatmap")
        if self.target_column in self.df.columns:
            print(f"   • Class distribution charts")
        
        # Additional insights
        print(f"\n7. ADDITIONAL INSIGHTS:")
        
        # Check for class imbalance
        if self.target_column in self.df.columns and self.df[self.target_column].dtype == 'object':
            class_counts = self.df[self.target_column].value_counts()
            if len(class_counts) > 0:
                max_count = class_counts.max()
                min_count = class_counts.min()
                imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
                print(f"   • Class imbalance ratio: {imbalance_ratio:.1f}")
                if imbalance_ratio > 5:
                    print(f"     ⚠️  Significant class imbalance detected!")
                    print(f"     → Consider using class weights or oversampling")
        
        # Check feature types
        print(f"   • Feature types analysis:")
        print(f"     - Total features: {len(self.df.columns)}")
        print(f"     - Numeric: {len(self.numeric_cols)}")
        print(f"     - Categorical: {len(self.categorical_cols)}")
        print(f"     - Target variable: {self.target_column}")
        
        # Next steps
        print(f"\n8. NEXT STEPS FOR PREPROCESSING:")
        print(f"   1. Handle missing values (if any)")
        print(f"   2. Encode categorical variables")
        print(f"   3. Scale numeric features")
        print(f"   4. Split data into train/validation/test sets")
        print(f"   5. Prepare data for neural network input")
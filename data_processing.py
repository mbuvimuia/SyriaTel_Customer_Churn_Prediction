import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
    
    def load_data(self):
        '''Load the data from the file_path and return it as a pandas dataframe'''
        self.data = pd.read_csv(self.file_path)
        return self.data
    
    def preprocess_data(self):
        '''Perform initial data preprocessing steps'''
        self._handle_missing_values()
        self._encode_categorical_features()
        self._scale_features()
        return self.data
    
    def _handle_missing_values(self):
        '''Handle missing values in the data'''
        self.data.fillna(method='ffill', inplace=True)

    def _encode_categorical_features(self):
        '''Encode categorical features in the data'''
        self.data = pd.get_dummies(self.data)   
    
    def _scale_features(self):
        '''Scale the features in the data'''
        scaler = StandardScaler()
        numerical_features = self.data.select_dtypes(include=['float64', 'int64']).columns
        self.data[numerical_features] = scaler.fit_transform(self.data[numerical_features])

    def split_data(self, target_column, test_size=0.2, random_state=42):
        '''Split the data into train and test sets'''
        X = self.data.drop(target_column, axis=1)
        y = self.data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

class DataAnalysis:
    def __init__(self, data):
        self.data = data
    
    def summary_statistics(self):
        '''Get summary statistics of the data'''
        return self.data.describe()
    
    def correlation_matrix(self):
        '''Get the correlation matrix of the data'''
        plt.figure(figsize=(12, 8))
        correlation = self.data.corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()
        return correlation
    
    def  plot_feature_distribution(self):
        '''Plot the distribution of features in the data'''
        self.data.hist(bins=50, figsize=(20, 15))
        plt.show()
    
    def univariate_analysis(self, column):
        '''Perform univariate analysis on all columns and produce a single output with all plots'''
        num_columns = self.data.shape[1]
        num_rows = num_columns // 4 + (num_columns % 4 > 0)
        
        fig, axes = plt.subplots(num_rows, 4, figsize=(20, 5 * num_rows))
        axes = axes.flatten()
        
        for i, column in enumerate(self.data.columns):
            if self.data[column].dtype == 'object':
                sns.countplot(x=self.data[column], ax=axes[i])
            else:
                sns.histplot(self.data[column], kde=True, ax=axes[i])
            
            axes[i].set_title(column)
        
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.show()
    
    def bivariate_analysis(self, target='target'):
        '''Perform bivariate analysis for all columns with respect to the target variable'''
        # Convert boolean target column to string for seaborn compatibility
        if self.data[target].dtype == 'bool':
            self.data[target]  = self.data[target].astype(str)
        elif self.data[target].dtype == 'category':
            self.data[target] = self.data[target].astype(str)


        
        num_cols = len(self.data.columns)
        plt.figure(figsize=(20, 4* num_cols))
        for i, column in enumerate(self.data.columns):
            if column == target:
                continue
            plt.subplot(num_cols, 1, i + 1)
            if self.data[column].dtype == 'object':
                sns.countplot(x=column, hue=target, data=self.data)
            else:
                sns.histplot(self.data, x=column, hue=target, kde=True, element='step')
            plt.title(f'{column} vs {target}')
        plt.tight_layout()
        plt.show()
    
    def check_imbalance(self, target='target'):
        '''Check for class imbalance in the target variable'''
        plt.figure(figsize=(8, 6))
        sns.countplot(x=target, data=self.data)
        plt.title('Target Variable Distribution')
        plt.show()
        return self.data[target].value_counts(normalize=True)
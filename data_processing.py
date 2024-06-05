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
        self.data.fillna(self.data.mean(), inplace=True)

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
        '''Plot and describe a single feature'''
        plt.figure(figsize=(10, 6))
        sns.histplot(self.data[column], kde=True)
        plt.title(f'Distribution of {column}')
        plt.show()
        return self.data[column].describe()
    
    def bivariate_analysis(self, feature, target='target'):
        '''Plot and describe the relationship between a feature and the target'''
        plt.figure(figsize=(10, 6))
        if self.data[feature].dtype == 'object':
            sns.countplot(x=feature, hue=target, data=self.data)
        else:
            sns.boxplot(x=target, y=feature, data=self.data)
        plt.title(f'{feature} vs {target}')
        plt.show()
        return self.data.grouby(target)[feature].describe()
    
    def check_imbalance(self, target='target'):
        '''Check for class imbalance in the target variable'''
        plt.figure(figsize=(8, 6))
        sns.countplot(x=target, data=self.data)
        plt.title('Target Variable Distribution')
        plt.show()
        return self.data[target].value_counts(normalize=True)
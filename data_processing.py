import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
import numpy as np


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
        self._encode_categorical_features()
        self._scale_features()
        return self.data
    
    def split_data(self, target_column, test_size=0.2, random_state=42):
        '''Split the data into train and test sets'''
        X = self.data.drop(target_column, axis=1)
        y = self.data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    
    # Encode categorical features in the data using OneHotEncoder
    def _encode_categorical_features(self, X_train, X_test):
        '''Encode categorical features in the data using OneHotEncoder'''
        ohe = OneHotEncoder(drop='first', sparse_output=False)
        categorical_columns = X_train.select_dtypes(include=['object']).columns

        X_train_encoded = ohe.fit_transform(X_train[categorical_columns])
        X_test_encoded = ohe.transform(X_test[categorical_columns])

        #Create DataFrames with appropriate column names
        X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=ohe.get_feature_names_out(categorical_columns))
        X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=ohe.get_feature_names_out(categorical_columns))

        #Drop the original categorical columns from the data
        X_train.drop(categorical_columns, axis=1, inplace=True)
        X_test.drop(categorical_columns, axis=1, inplace=True)

        #Concatenate the encoded columns to the data
        X_train = pd.concat([X_train.reset_index(drop=True), X_train_encoded_df], axis=1)
        X_test = pd.concat([X_test.reset_index(drop=True), X_test_encoded_df], axis=1)
        
        return X_train, X_test
    
    # Encode target variable in the data
    def _encode_target_variable(self, y_train, y_test):
        '''Encode the target variable in the data'''
        y_train = y_train.map({'True': 1, 'False': 0})
        y_test = y_test.map({'True': 1, 'False': 0})

        return y_train, y_test
    
    def imbalance_solved(self, X_train, y_train):
        '''Handle imbalance in the target variable using SMOTE'''
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        
        return X_train_res, y_train_res
    
    def _scale_features(self, X_train, X_test):
        '''Scale the features in the data using StandardScaler'''
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled

        

class DataAnalysis:
    def __init__(self, data):
        self.data = data
    
    def summary_statistics(self):
        '''Get summary statistics of the data'''
        return self.data.describe()
    
    def correlation_matrix(self):
        '''Get the correlation matrix of the data'''
         #Drop state and area code columns(not important for corelation matrix)
        self.data.drop(['state', 'area_code'], axis=1, inplace=True)

        # Transform the categorical columns to numerical columns
        categorical_columns = self.data.select_dtypes(include=['object']).columns
        self.data = pd.get_dummies(self.data, columns=categorical_columns, drop_first=True)

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

class ModelEvaluation:
    def __init__(self, model, X_test, y_test, feature_names=None):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names
    
    def evaluate_model(self):
        '''Evaluate the model using the test data'''
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        return print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}')
    
    def plot_confusion_matrix(self):
        '''Plot the confusion matrix for the model'''
        y_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.model.classes_)
        disp.plot(cmap='Blues')
        plt.show()
    
    def plot_roc_curve(self):
        '''Plot the ROC curve for the model'''
        y_pred_prob = self.model.predict_proba(self.X_test)[:, 1]
        RocCurveDisplay.from_predictions(self.y_test, y_pred_prob)
        plt.show()
    
    def plot_precision_recall_curve(self):
        '''Plot the precision-recall curve for the model'''
        y_pred_prob = self.model.predict_proba(self.X_test)[:, 1]
        PrecisionRecallDisplay.from_predictions(self.y_test, y_pred_prob)
        plt.show()
    


    
    def plot_learning_curve(self):
        '''Plot the learning curve for the model'''
        train_sizes, train_scores, test_scores = learning_curve(self.model, self.X_test, self.y_test, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), verbose=0)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure(figsize=(12, 6))
        plt.title('Learning Curve')
        plt.xlabel('Training Examples')
        plt.ylabel('Score')
        plt.ylim(0.7, 1.01)
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color='g')

        plt.legend(loc='best')
        plt.show()
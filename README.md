# 🚀 SyriaTel Customer Churn Prediction

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Machine Learning](https://img.shields.io/badge/ML-Classification-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📊 Project Overview

A comprehensive machine learning solution for predicting customer churn in the telecommunications industry. This project helps SyriaTel proactively identify customers at risk of leaving, enabling targeted retention strategies to reduce customer attrition and improve business profitability.

### 🎯 Business Impact
- **96% accuracy** in churn prediction
- **Proactive customer retention** through risk identification
- **Cost reduction** in customer acquisition
- **Revenue optimization** through targeted interventions

## 🔍 Problem Statement

Customer churn poses a significant threat to telecommunications companies, with acquisition costs being 5-10x higher than retention costs. This project addresses the critical need to:

- Predict which customers are likely to churn
- Identify key factors driving customer attrition
- Enable data-driven retention strategies
- Optimize resource allocation for maximum ROI

## 📈 Dataset Overview

**Source**: [Kaggle - Churn in Telecoms Dataset](https://www.kaggle.com/becksddf/churn-in-telecoms-dataset)

**Dataset Characteristics**:
- **Size**: 3,333 observations × 21 features
- **Target Variable**: Binary churn indicator (imbalanced dataset)
- **Feature Types**: Demographic, behavioral, and usage metrics

### Key Features:
- **Demographics**: State, area code, account length
- **Service Plans**: International plan, voicemail plan
- **Usage Metrics**: Call minutes, charges, and frequencies across different time periods
- **Service Quality**: Customer service call frequency

## 🛠️ Technical Architecture

### Data Processing Pipeline
```python
DataProcessor → DataAnalysis → ModelEvaluation
```

#### Custom Classes:
1. **`DataProcessor`**: Handles data loading, preprocessing, and feature engineering
2. **`DataAnalysis`**: Comprehensive EDA with statistical analysis and visualizations
3. **`ModelEvaluation`**: Model performance assessment and comparison

### Feature Engineering
- **Encoding**: OneHotEncoder for categorical variables
- **Scaling**: StandardScaler for numerical features
- **Balancing**: SMOTE (Synthetic Minority Over-sampling Technique) for class imbalance
- **Validation**: Train-test split with stratification

## 🤖 Machine Learning Models

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | AUC Score |
|-------|----------|-----------|--------|----------|-----------|
| **Gradient Boosting** | **96%** | **89%** | **82%** | **85%** | **94%** |
| Random Forest | 94% | 82% | 79% | 80% | 94% |
| Logistic Regression | Baseline | - | - | - | - |

### Model Performance Visualizations
- ROC Curves with AUC scores
- Precision-Recall curves
- Confusion matrices
- Feature importance analysis

## 📊 Key Insights

### Churn Predictors Identified:
1. **Customer Service Calls**: Higher frequency correlates with increased churn risk
2. **International Plan**: Customers without international plans show higher churn rates
3. **Voicemail Plan**: Voicemail usage indicates lower churn probability
4. **Geographic Variation**: Churn rates vary significantly across states

### Business Recommendations:
- **Proactive Support**: Monitor customers with frequent service calls
- **Plan Optimization**: Review international plan offerings and pricing
- **Value-Added Services**: Promote voicemail and other retention-driving features
- **Regional Strategies**: Implement state-specific retention programs

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
pip install -r requirements.txt
```

### Installation & Usage
```bash
# Clone the repository
git clone https://github.com/yourusername/SyriaTel_Customer_Churn_Prediction.git
cd SyriaTel_Customer_Churn_Prediction

# Install dependencies
pip install -r requirements.txt

# Run the analysis
jupyter notebook index.ipynb
```

### Quick Start Example
```python
from data_processing import DataProcessor, DataAnalysis, ModelEvaluation

# Initialize data processor
processor = DataProcessor('data/telecom_dataset.xls')
data = processor.load_data()

# Perform analysis
analyzer = DataAnalysis(data)
analyzer.summary_statistics()
analyzer.correlation_matrix()

# Train and evaluate model
# (See notebook for complete implementation)
```

## 📁 Project Structure
```
SyriaTel_Customer_Churn_Prediction/
├── data/
│   └── telecom_dataset.xls          # Raw dataset
├── Plots/                           # Generated visualizations
│   ├── image.png                    # ROC curves
│   ├── image-1.png                  # Precision-Recall curves
│   └── ...
├── data_processing.py               # Core ML pipeline classes
├── index.ipynb                      # Main analysis notebook
├── jupyter_notebook.pdf             # Exported notebook
├── Presentation.pdf                 # Project presentation
└── README.md                        # Project documentation
```

## 🔧 Technical Dependencies

### Core Libraries
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **imbalanced-learn**: Handling class imbalance
- **xgboost**: Gradient boosting implementation

### Visualization
- **matplotlib**: Static plotting
- **seaborn**: Statistical data visualization

### Environment
- **jupyter**: Interactive development
- **Python 3.8+**: Core runtime

## 📋 Methodology

### 1. Data Understanding & Exploration
- Statistical analysis of features
- Correlation analysis
- Univariate and bivariate analysis
- Class imbalance assessment

### 2. Data Preprocessing
- Missing value handling
- Categorical encoding (One-Hot)
- Feature scaling (StandardScaler)
- Class balancing (SMOTE)

### 3. Model Development
- Baseline model establishment
- Hyperparameter tuning
- Cross-validation
- Ensemble methods

### 4. Model Evaluation
- Comprehensive metrics (Accuracy, Precision, Recall, F1, AUC)
- ROC and Precision-Recall curves
- Feature importance analysis
- Business impact assessment

## 🎯 Results & Impact

### Model Performance
- **Production-Ready Accuracy**: 96% overall accuracy
- **High Precision**: 89% precision minimizes false positives
- **Balanced Performance**: Strong recall (82%) ensures most churners are identified
- **Robust AUC**: 94% AUC score indicates excellent model discrimination

### Business Value
- **Proactive Intervention**: Identify at-risk customers before they churn
- **Cost Optimization**: Focus retention efforts on high-risk segments
- **Revenue Protection**: Reduce customer acquisition costs through improved retention
- **Strategic Insights**: Data-driven understanding of churn drivers

## 🔮 Future Enhancements

- **Real-time Scoring**: Deploy model for real-time churn prediction
- **Feature Engineering**: Advanced feature creation from temporal patterns
- **Deep Learning**: Explore neural networks for improved performance
- **A/B Testing**: Validate retention strategies through controlled experiments
- **Automated Retraining**: Implement MLOps pipeline for model updates

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or collaboration opportunities, please reach out:
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

---

*Built with ❤️ for data-driven customer retention*



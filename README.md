# Random Forest for Student Stress Level Prediction

A comprehensive machine learning project that predicts student stress levels using Random Forest algorithm based on multifactorial bio-signals and socio-environmental factors.

## 🎯 Project Overview

This project implements a **Random Forest classifier** to predict student stress levels using a comprehensive dataset of psychological, physiological, social, environmental, and academic factors. The model analyzes 20+ key features that contribute to student stress and provides accurate predictions with proper hyperparameter tuning to handle overfitting.

### Problem Statement

**Predicting Student Stress from Multifactorial Bio-Signals**: The objective is to develop a predictive model that analyzes and quantifies the relationship between psychological, physiological, social, environmental, and academic factors and a student's overall stress level.

## 📊 Dataset Information

The **StressLevelDataset.csv** contains 1,100 student records with 21 features:

### Features Include:
- **Psychological Factors**: anxiety_level, self_esteem, mental_health_history, depression
- **Physiological Factors**: headache, blood_pressure, sleep_quality, breathing_problem
- **Environmental Factors**: noise_level, living_conditions, safety, basic_needs
- **Academic Factors**: academic_performance, study_load, teacher_student_relationship, future_career_concerns
- **Social Factors**: social_support, peer_pressure, extracurricular_activities, bullying
- **Target Variable**: stress_level (0: Low, 1: Moderate, 2: High)

Dataset source: [Kaggle - Student Stress Factors Analysis](https://www.kaggle.com/datasets/rxnach/student-stress-factors-a-comprehensive-analysis)

## 🌳 What is Random Forest?

**Random Forest** is a powerful ensemble machine learning algorithm that combines multiple decision trees to create more accurate and robust predictions. Here's how it works:

### Core Concepts:

1. **Ensemble Learning**: Instead of relying on a single decision tree, Random Forest creates a "forest" of many trees and combines their predictions
2. **Bootstrap Aggregation (Bagging)**: Each tree is trained on a different random sample of the data
3. **Random Feature Selection**: At each split, only a random subset of features is considered
4. **Majority Voting**: For classification, the final prediction is determined by majority vote from all trees

### Key Advantages:

- **High Accuracy**: Combining multiple trees typically produces better results than single models
- **Reduced Overfitting**: Random sampling and feature selection prevent overfitting
- **Handles Missing Data**: Naturally robust to missing values and outliers
- **Feature Importance**: Provides rankings of which features are most important for predictions
- **Versatility**: Works well for both classification and regression problems
- **No Assumptions**: Non-parametric method that doesn't assume data distribution

### How Random Forest Works:

1. **Create Multiple Trees**: Generate many decision trees, each trained on a random subset of data
2. **Random Feature Selection**: For each tree node split, randomly select features to consider
3. **Individual Predictions**: Each tree makes its own prediction
4. **Aggregate Results**: Combine all tree predictions through voting (classification) or averaging (regression)
5. **Final Output**: The most voted class becomes the final prediction

## 🛠️ Implementation Details

### Libraries Used:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
```

### Model Architecture:
- **Algorithm**: RandomForestClassifier
- **Initial Parameters**: n_estimators=500, max_depth=None, class_weight="balanced"
- **Preprocessing**: StandardScaler for feature normalization
- **Train-Test Split**: 80-20 split with random_state=42

## 📈 Model Performance

### Initial Model Results:
- **Training Accuracy**: 100% (indicating overfitting)
- **Test Accuracy**: 85.45%

### After Hyperparameter Tuning:
- **Training Accuracy**: 100% 
- **Test Accuracy**: 87.73%
- **Cross-Validation Score**: 89.88%

### Best Hyperparameters:
```python
{
    'class_weight': 'balanced',
    'max_depth': None,
    'min_samples_leaf': 1,
    'min_samples_split': 10,
    'n_estimators': 100
}
```

### Classification Report (Tuned Model):
```
              precision    recall  f1-score   support

         0.0       0.84      0.89      0.87        76
         1.0       0.90      0.86      0.88        73  
         2.0       0.90      0.87      0.89        71

    accuracy                           0.88       220
   macro avg       0.88      0.88      0.88       220
weighted avg       0.88      0.88      0.88       220
```

## 🔧 Key Features of Implementation

### 1. Comprehensive EDA:
- **Data Quality Check**: No missing values or duplicates found
- **Statistical Analysis**: Descriptive statistics for all 21 features
- **Correlation Analysis**: Heatmap visualization of feature relationships
- **Outlier Detection**: Box plots and IQR-based outlier removal

### 2. Data Preprocessing:
- **Outlier Treatment**: IQR-based outlier capping for all features
- **Feature Scaling**: StandardScaler normalization
- **Data Splitting**: Stratified train-test split

### 3. Model Development:
- **Baseline Model**: Initial Random Forest with default parameters
- **Hyperparameter Tuning**: GridSearchCV with 3-fold cross-validation
- **Performance Evaluation**: Comprehensive metrics and confusion matrix

### 4. Hyperparameter Tuning Details:
```python
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced']
}
```

## 🚀 Getting Started

### Prerequisites:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn kagglehub
```

### Running the Project:

1. **Clone the repository**:
```bash
git clone https://github.com/GoriyashKashyap/RandomForest_Project.git
cd RandomForest_Project
```

2. **Run the Jupyter notebook**:
```bash
jupyter notebook RandomForest.ipynb
```

3. **Or execute directly**:
The notebook automatically downloads the dataset from Kaggle using `kagglehub`.

## 📁 Project Structure

```
RandomForest_Project/
├── RandomForest.ipynb          # Main Jupyter notebook with complete analysis
├── StressLevelDataset.csv      # Dataset file
├── README.md                   # Project documentation
```

## 🎯 Key Insights

### Model Insights:
1. **Balanced Performance**: The model performs well across all stress levels (Low, Moderate, High)
2. **Feature Importance**: Random Forest provides rankings of most influential factors
3. **Overfitting Control**: Hyperparameter tuning improved generalization
4. **Robust Predictions**: Ensemble approach provides stable results

### Hyperparameter Impact:
- **min_samples_split=10**: Prevents overfitting by requiring more samples for splits
- **n_estimators=100**: Optimal balance between performance and computational efficiency
- **class_weight='balanced'**: Handles any class imbalance effectively

## 🔮 Future Enhancements

### Potential Improvements:
1. **Feature Engineering**: Create interaction terms and polynomial features
2. **Advanced Ensemble**: Implement Gradient Boosting or XGBoost for comparison
3. **Real-time Prediction**: Deploy model as web application
4. **Feature Selection**: Apply advanced feature selection techniques
5. **Cross-validation**: Implement k-fold stratified cross-validation
6. **Model Interpretability**: Add SHAP values for better feature explanation

### Applications:
- **Educational Institutions**: Early identification of stressed students
- **Mental Health Support**: Proactive intervention systems  
- **Academic Planning**: Workload optimization based on stress factors
- **Research**: Understanding key contributors to student stress

## 📚 Learning Outcomes

This project demonstrates:
- **Ensemble Learning**: Understanding Random Forest as an ensemble method
- **Hyperparameter Tuning**: Systematic optimization using GridSearchCV
- **Data Preprocessing**: Complete pipeline from raw data to model-ready format
- **Model Evaluation**: Comprehensive assessment using multiple metrics
- **Overfitting Management**: Techniques to improve model generalization

## 🤝 Contributing

Feel free to contribute to this project by:
- Reporting bugs or issues
- Suggesting new features
- Improving documentation
- Adding new analysis techniques

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**GoriyashKashyap**
- GitHub: [@GoriyashKashyap](https://github.com/GoriyashKashyap)

---

⭐ **Star this repository if you found it helpful!**

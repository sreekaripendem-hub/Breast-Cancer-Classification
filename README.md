# Breast-Cancer-Classification
This project explores feature selection, dimensionality reduction, model evaluation and classification using the Wisconsin Breast Cancer Dataset. It focuses on identifying the most relevant features and evaluating machine learning models to distinguish between benign and malignant tumours.

The project also investigates K-Fold Cross-Validation and Logistic Regression, including log-loss, parameter sensitivity and gradient behaviour.

Dataset

The project uses the Wisconsin Breast Cancer Dataset, containing:

569 observations
30 numerical features
2 diagnosis classes: Benign (B) and Malignant (M)
357 benign and 212 malignant observations

The dataset includes measurements such as radius, texture, perimeter, area, smoothness, compactness and concavity.

Objectives
Explore and preprocess the dataset.
Apply feature selection and dimensionality reduction techniques.
Compare model performance using different feature subsets.
Implement K-Fold Cross-Validation.
Determine the optimal number of neighbours for KNN.
Develop a Logistic Regression classifier.
Analyse log-loss, parameter sensitivity and analytical gradients.
Methodology
1. Data Preprocessing and Exploration
Data loading and cleaning using Pandas.
Feature standardization using StandardScaler.
Exploratory Data Analysis (EDA).
Correlation analysis and visualization.
2. Feature Selection and Dimensionality Reduction

Four techniques were implemented to reduce the original 30-dimensional feature space to 10 features:

Information Gain: Mutual Information-based feature selection.
Backward Elimination: Sequential removal of features based on model performance.
SVM-RFE: Recursive Feature Elimination using Support Vector Machines.
PCA: Principal Component Analysis.

K-Nearest Neighbours (KNN) was used to evaluate the selected feature representations.

3. Model Evaluation and Comparison

The performance of the four techniques was compared using:

Accuracy
Malignant Recall
Malignant F1-Score
Confusion Matrix
Classification Report

SVM-RFE achieved the following results:

Evaluation Metric	Score
Accuracy	97.08%
Malignant Recall	92.19%
Malignant F1-Score	95.93%

It reduced the feature space from 30 features to 10.

4. K-Fold Cross-Validation
Implemented Stratified 5-Fold Cross-Validation.
Compared single train-test split results with cross-validation results.
Evaluated different neighbourhood sizes for KNN.
Used Matthews Correlation Coefficient (MCC) to identify the optimal K value.

Selected optimal KNN neighbourhood size: k = 5

Mean Cross-Validation MCC: 0.9326

5. Logistic Regression

A Logistic Regression classifier was developed using the SVM-RFE selected features.

The analysis included:

Out-of-fold predicted probabilities using 5-Fold Cross-Validation.
Log-loss calculation.
Parameter sensitivity analysis through log-loss plots.
Analytical gradient computation.
Interpretation of gradient signs and magnitudes during gradient descent.

5-Fold Cross-Validation Log-Loss: 0.0986

Technologies and Libraries
Python
Jupyter Notebook
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn
Repository Structure

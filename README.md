# M.Tech Machine Learning Assignment 2: Binary Classification App

2026aj05470 
P. Sai Prakash Pattabhi

An end-to-end Machine Learning assignment for the BITS Pilani M.Tech AIML program. This repository contains the complete pipeline for training, evaluating, and deploying five distinct classification algorithms using a 12-feature dataset via an interactive Streamlit web application.

---

## 1. Problem Statement
The objective of this assignment is to develop, evaluate, and deploy a multi-model machine learning system that performs binary classification on a 12-feature dataset. The system provides dynamic model evaluation and interactive visualizations (Confusion Matrix Heatmaps and ROC Curves) on test data uploaded by the user without performing on-the-fly retraining.

---

## 2. Dataset Description
* **Features:** 12 numeric features (`feature_1` through `feature_12`)
* **Target:** Binary classification label (`0` or `1`)
* **Total Samples:** 1,200 instances (900 training samples, 300 test samples)
* **Test Dataset File:** `test_data.csv` (contains 300 test instances with features and the target label)

---

## 3. Repository Structure
```text
project-folder/
├── app.py                      # Main Streamlit web application
├── train_models.py             # Script for data generation, training & exporting models
├── requirements.txt            # Environment dependencies
├── README.md                   # Assignment documentation
├── test_data.csv               # 300-sample test set (features + target)
└── model/                      # Directory storing serialized joblib artifacts
    ├── scaler.joblib
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── support_vector_machine.joblib


## 4. Models Used & Performance Evaluation

### Comparison Table
The evaluation metrics below were generated using the 300 holdout samples in `test_data.csv`:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8667 | 0.9374 | 0.8686 | 0.8667 | 0.8662 | 0.7344 |
| **Decision Tree** | 0.8067 | 0.8671 | 0.8136 | 0.8067 | 0.8048 | 0.6183 |
| **KNN** | 0.9467 | 0.9852 | 0.9485 | 0.9467 | 0.9465 | 0.8949 |
| **Naive Bayes** | 0.8300 | 0.9138 | 0.8435 | 0.8300 | 0.8274 | 0.6714 |
| **Support Vector Machine** | **0.9567** | **0.9873** | **0.9576** | **0.9567** | **0.9566** | **0.9140** |

---

### Performance Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieves strong baseline performance (86.67% accuracy, 0.9374 AUC). Performs well due to linear decision boundaries, though slightly limited by non-linear feature combinations. |
| **Decision Tree** | Lowest performer among all models (80.67% accuracy, 0.6183 MCC). The depth constraint (`max_depth=5`) prevents overfitting but leads to underfitting high-dimensional interaction spaces. |
| **KNN** | Excellent performance (94.67% accuracy, 0.9852 AUC). Distance-based clustering effectively captures complex localized relationships after standard scaling. |
| **Naive Bayes** | Moderate performance (83.00% accuracy). Output suffers slightly due to the conditional independence assumption among the 12 correlated/redundant features. |
| **Support Vector Machine** | Top overall performance (95.67% accuracy, 0.9873 AUC). RBF kernel maps the 12 features into non-linear spaces where classes are linearly separable with optimal margins. |
| **Overall Winner** | **Support Vector Machine (SVM)** is the clear winner across all six metrics, attaining the highest accuracy (95.67%), AUC (0.9873), and MCC (0.9140). |

---

## 5. Live Streamlit Application
* **Deployment Platform:** Streamlit Community Cloud
* **Live App Link:** [https://2026aj05470-ml-assignment-2.streamlit.app/](https://2026aj05470-ml-assignment-2.streamlit.app/)

### Features Included
* **CSV File Upload:** Drag-and-drop support for test datasets.
* **Algorithm Selector:** Dropdown menu to evaluate models dynamically without retraining.
* **Evaluation Metrics Card Grid:** Interactive display for Accuracy, AUC, MCC, Precision, Recall, and F1 score.
* **Visual Insights:** Live confusion matrix heatmaps and ROC curve charts.

---

## 6. How to Run Locally

### Prerequisites
* Python 3.9+ installed

### Setup & Run
1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/2026aj05470/ml-assignment-2.git](https://github.com/2026aj05470/ml-assignment-2.git)
   cd ml-assignment-2
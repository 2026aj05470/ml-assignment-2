import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import make_classification

os.makedirs("model", exist_ok=True)

# =====================================================================
# CRITICAL: Replace this synthetic data block with your Kaggle/UCI data
# df = pd.read_csv("your_kaggle_dataset.csv")
# X = df.drop(columns=['target'])
# y = df['target']
# feature_names = X.columns.tolist()
# =====================================================================

# Placeholder generating exactly 12 features and 1200 instances
X, y = make_classification(n_samples=1200, n_features=12, n_informative=9, n_redundant=3, random_state=42)
feature_names = [f"feature_{i+1}" for i in range(12)]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

X_train, X_test, y_train, y_test = train_test_split(
    df[feature_names], df['target'], test_size=0.25, random_state=42
)

# Save test dataset (must contain target column for Streamlit evaluation)
test_df = pd.concat([X_test, y_test], axis=1)
test_df.to_csv("test_data.csv", index=False)

# Fit and save scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
joblib.dump(scaler, "model/scaler.joblib")

# Define 5 classification algorithms (SVM must have probability=True for AUC)
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(probability=True, random_state=42)
}

# Fit and export each model binary
for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)
    # Save filenames cleanly mapped to keys
    safe_name = model_name.replace(" ", "_").lower()
    joblib.dump(model, f"model/{safe_name}.joblib")

print("Training complete. Artifacts saved in 'model/' and test set saved to 'test_data.csv'.")
import os
import joblib
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Create output directories
os.makedirs("model", exist_ok=True)

# Generate synthetic binary dataset matching requirements (12 features, 1200 samples)
X, y = make_classification(
    n_samples=1200, 
    n_features=12, 
    n_informative=9, 
    n_redundant=3, 
    random_state=42
)
feature_names = [f"feature_{i+1}" for i in range(12)]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# Split into Train and Test sets (75% train, 25% test)
X_train, X_test, y_train, y_test = train_test_split(
    df[feature_names], df['target'], test_size=0.25, random_state=42
)

# Save test dataset for Streamlit UI upload
test_df = pd.concat([X_test, y_test], axis=1)
test_df.to_csv("test_data.csv", index=False)

# Fit and save scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
joblib.dump(scaler, "model/scaler.joblib")

# Define 5 classification algorithms
models = {
    "logistic_regression": LogisticRegression(random_state=42),
    "decision_tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "naive_bayes": GaussianNB(),
    "support_vector_machine": SVC(probability=True, random_state=42)
}

# Fit and export each model binary
for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)
    joblib.dump(model, f"model/{model_name}.joblib")

print("Training complete. Artifacts saved in 'model/' and test set saved to 'test_data.csv'.")
import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    classification_report, confusion_matrix, roc_curve
)

st.set_page_config(page_title="M.Tech ML Assignment 2", layout="wide")
st.title("M.Tech ML Assignment: 12-Feature Classification")

# Required for Streamlit v1.12.0 caching
# Use modern caching decorator supported across current Streamlit releases
@st.cache_resource
def load_artifacts():
    scaler = joblib.load(os.path.join("model", "scaler.joblib"))
    
    models = {
        "Logistic Regression": joblib.load(os.path.join("model", "logistic_regression.joblib")),
        "Decision Tree": joblib.load(os.path.join("model", "decision_tree.joblib")),
        "KNN": joblib.load(os.path.join("model", "knn.joblib")),
        "Naive Bayes": joblib.load(os.path.join("model", "naive_bayes.joblib")),
        "Support Vector Machine": joblib.load(os.path.join("model", "support_vector_machine.joblib"))
    }
    return scaler, models

# Load artifacts safely
try:
    scaler, models = load_artifacts()
    st.sidebar.success("All models & scaler loaded successfully!")
except FileNotFoundError:
    st.error("Model artifacts not found. Please run train_models.py first.")
    st.stop()

# Requirement: Model selection dropdown[cite: 5]
st.sidebar.header("Configuration")
selected_model_name = st.sidebar.selectbox("Select Classification Algorithm", list(models.keys()))
active_model = models[selected_model_name]

# Requirement: Dataset upload option (CSV)[cite: 5]
st.markdown("Upload your `test_data.csv` containing the features and the `target` column to evaluate the model.")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(input_data.head())
    
    if 'target' not in input_data.columns:
        st.error("The uploaded CSV must contain a 'target' column to calculate evaluation metrics.")
    else:
        if st.button("Evaluate Model"):
            with st.spinner(f"Evaluating using {selected_model_name}..."):
                # Split features and target
                X_test = input_data.drop(columns=['target'])
                y_true = input_data['target']
                
                # Scale features
                X_test_scaled = scaler.transform(X_test)
                
                # Predictions
                y_pred = active_model.predict(X_test_scaled)
                
                # Calculate Probabilities for AUC
                if hasattr(active_model, "predict_proba"):
                    y_proba = active_model.predict_proba(X_test_scaled)
                    if y_proba.shape[1] == 2:
                        y_proba_pos = y_proba[:, 1]
                        auc = roc_auc_score(y_true, y_proba_pos)
                    else:
                        y_proba_pos = y_proba
                        auc = roc_auc_score(y_true, y_proba_pos, multi_class='ovr')
                else:
                    auc = "N/A (Model lacks predict_proba)"

                # Requirement: Calculate Evaluation Metrics[cite: 3]
                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                mcc = matthews_corrcoef(y_true, y_pred)

                # Requirement: Display Evaluation Metrics[cite: 5]
                st.subheader(f"Evaluation Metrics: {selected_model_name}")
                col1, col2, col3 = st.columns(3)
                col1.metric("Accuracy", f"{acc:.4f}")
                col2.metric("AUC Score", f"{auc:.4f}" if isinstance(auc, float) else auc)
                col3.metric("MCC Score", f"{mcc:.4f}")
                
                col4, col5, col6 = st.columns(3)
                col4.metric("Precision", f"{prec:.4f}")
                col5.metric("Recall", f"{rec:.4f}")
                col6.metric("F1 Score", f"{f1:.4f}")

                st.markdown("---")

                # Requirement: Confusion matrix or classification report[cite: 5]
                st.subheader("Classification Report")
                report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                st.dataframe(pd.DataFrame(report).transpose())
                
                # ---------------- Graphs ---------------- #
                st.markdown("---")
                st.subheader("Visualizations")
                
                col_graph1, col_graph2 = st.columns(2)
                
                # Confusion Matrix Heatmap
                cm = confusion_matrix(y_true, y_pred)
                with col_graph1:
                    st.write("**Confusion Matrix Heatmap**")
                    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                    plt.ylabel('Actual')
                    plt.xlabel('Predicted')
                    st.pyplot(fig_cm)
                
                # ROC Curve
                with col_graph2:
                    if hasattr(active_model, "predict_proba") and isinstance(auc, float):
                        st.write("**ROC Curve**")
                        fpr, tpr, thresholds = roc_curve(y_true, y_proba_pos)
                        fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
                        ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
                        ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                        ax_roc.set_xlim([0.0, 1.0])
                        ax_roc.set_ylim([0.0, 1.05])
                        ax_roc.set_xlabel('False Positive Rate')
                        ax_roc.set_ylabel('True Positive Rate')
                        ax_roc.legend(loc="lower right")
                        st.pyplot(fig_roc)
                    else:
                        st.info("ROC Curve not available for this model (Requires predict_proba).")
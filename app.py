import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="ML Classifier Benchmark App", layout="wide")

st.title("📊 Multi-Model Machine Learning Classification Dashboard")
st.markdown("BITS Pilani M.Tech AIML — Machine Learning Assignment 2")

# Sidebar Controls
st.sidebar.header("User Control Panel")
uploaded_file = st.sidebar.file_uploader("Upload Test CSV File", type=["csv"])

model_options = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib"
}

selected_model_label = st.sidebar.selectbox("Select ML Model", list(model_options.keys()))

@st.experimental_singleton
def load_scaler():
    return joblib.load("model/scaler.joblib")

@st.experimental_singleton
def load_model(filename):
    return joblib.load(f"model/{filename}")

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Test Dataset Preview", test_df.head())

    if "target" not in test_df.columns:
        st.error("Error: CSV file must contain a 'target' column.")
    else:
        X_test = test_df.drop(columns=["target"])
        y_test = test_df["target"]

        # Scaler transformation & model inference
        scaler = load_scaler()
        X_test_scaled = scaler.transform(X_test)
        
        model_filename = model_options[selected_model_label]
        model = load_model(model_filename)
        
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        st.subheader(f"Evaluation Metrics: {selected_model_label}")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("AUC", f"{auc:.4f}")
        col3.metric("Precision", f"{prec:.4f}")
        col4.metric("Recall", f"{rec:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC", f"{mcc:.4f}")

        st.divider()

        # Render Confusion Matrix & Classification Report
        col_cm, col_cr = st.columns(2)

        with col_cm:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            st.pyplot(fig)

        with col_cr:
            st.subheader("Classification Report")
            report_dict = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report_dict).transpose().style.format("{:.4f}"))
else:
    st.info("👈 Upload your test dataset (`test_data.csv`) using the sidebar to view metrics.")
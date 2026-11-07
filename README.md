# M.Tech ML Assignment 2: Binary Classification

This repository contains a Streamlit web application for a 12-feature binary classification machine learning model, developed for the BITS Pilani M.Tech AIML program.

## Dataset
* **Source:** The model was trained using data sourced from Kaggle [Insert Kaggle link here if you have it].
* **Testing:** To test the application, users can upload the provided `test.csv` file, which contains the 12 required features for prediction.

## Files in this Repository
* `app.py`: The main Streamlit application script.
* `train_models.py`: The script used to train the models and generate `.joblib` files.
* `requirements.txt`: List of Python dependencies (including Streamlit v1.12.0).
* `test.csv`: Sample data for testing the live app.
* `*.joblib`: Pre-trained machine learning models.

## How to Run Locally
1. Clone the repository.
2. Install the required packages: `pip install -r requirements.txt`
3. Run the application: `streamlit run app.py`
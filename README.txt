Customer Personality Project
=================================

Contents:
- customer_personality_notebook.ipynb : Jupyter notebook (training, feature engineering, model development, evaluation)
- app.py : Streamlit app for upload, predict, and retrain on new dataset
- requirements.txt : Python dependencies
- model/ : saved models and scaler (created after running notebook)
- marketing_campaign.csv : (NOT included) Please download the dataset from Kaggle and place it here:
    https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis

Notes:
- The notebook reads 'marketing_campaign.csv' from the project folder. Place the Kaggle CSV there before running.
- The notebook clusters customers into 6 clusters (as requested), then trains supervised models to predict cluster labels.
- The Streamlit app allows users to upload a dataset for prediction and to retrain the pipeline on a new dataset (live retraining).
- This ZIP was generated programmatically. Open in VS Code and run the notebook using the Jupyter extension.

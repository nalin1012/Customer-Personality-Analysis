# Customer-Personality-Analysis
🧠 Customer Personality Analysis & CRM Prediction

A machine learning project that analyzes customer behavior, segments customers into meaningful groups, and provides real-time personality predictions through a Streamlit web app with live model retraining.

🚀 Features

✅ Customer segmentation using clustering
✅ Personality typing (6 customer groups)
✅ Streamlit web app for:

CSV upload

Automatic preprocessing

Prediction of customer segment

Personality label assignment

Live retraining of models

✅ CRM Dashboard for predicting new customer profiles
✅ Downloadable prediction results



📊 Dataset

Source: Kaggle – Customer Personality Analysis
Columns include:

Demographics (Age, Income, etc.)

Purchase behavior

Spending patterns

Channel preference

Customer tenure and recency

🧠 Model Workflow

1️⃣ Data Cleaning & Preprocessing
2️⃣ Feature Engineering
3️⃣ Clustering (KMeans → 6 segments)
4️⃣ Classification (Random Forest)
5️⃣ Deployment with Streamlit
6️⃣ Optional Retraining on user-uploaded data

🏷️ Personality Segments

The model classifies customers into:

Occasional Shoppers

Budget-Conscious Families

Young Enthusiasts

Established Professionals

Luxury Spenders

Traditional Buyers

🌐 Run the App Locally
pip install -r requirements.txt
streamlit run app.py

🖥️ Live Retraining

Upload a new dataset and the app will:

✅ retrain KMeans
✅ retrain RandomForest
✅ save new models
✅ update predictions automatically

📦 Requirements
pip install -r requirements.txt

🚀 Future Improvements

🔹 Deploy on Streamlit Cloud
🔹 Add visualization dashboard
🔹 Improve clustering quality
🔹 Add customer churn prediction

👨‍💻 Author

Developed as part of a Machine Learning project focusing on CRM insights and real-time model deployment.

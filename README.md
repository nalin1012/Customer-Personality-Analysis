# 🧠 Customer Personality Predictor

A modern Machine Learning & Streamlit-based dashboard that predicts customer segments and generates personalized marketing strategies using:

✅ KMeans Clustering (Customer Segmentation)
✅ Random Forest Classifier (Cluster Prediction)
✅ StandardScaler (Feature Scaling)
✅ Interactive Streamlit UI

---

## 🚀 Project Overview

This project analyzes customer purchasing behavior and classifies users into meaningful market segments. Based on the input details, the system:

* Predicts customer cluster
* Identifies generation group
* Provides financial and engagement insights
* Suggests personalized marketing strategies
* Displays model training and performance details

This helps businesses:
✅ Understand customer behavior
✅ Improve targeting & retention
✅ Increase conversion rates
✅ Optimize marketing spend

---

## 🎯 Features

✅ Real‑time customer prediction
✅ Customer segmentation (6 clusters)
✅ Personalized marketing strategies
✅ Insights dashboard
✅ Model information & training summary
✅ Visualization charts:

* Accuracy chart
* Inertia metric (KMeans)
* Spend behavior
  ✅ Modern UI with glass theme

---

## 🧠 Machine Learning Pipeline

### 1. Data Processing

* Removed missing values
* Cleaned inconsistent entries
* Feature engineering:

  * Total_Spend
  * Purchase_Frequency
  * Tenure
  * Income ratios
* Scaled numeric features using **StandardScaler**

### 2. Segmentation Model

```
KMeans(n_clusters=6, random_state=42)
```

Used for grouping customers by:
✅ Spending
✅ Engagement
✅ Purchase channels

### 3. Classification Model

```
RandomForestClassifier(n_estimators=200, random_state=42)
```

Used for predicting cluster labels for new customers

---

## 🏗️ Project Structure

```
customer_personality_project/
│
├── app.py
├── model/
│   ├── scaler.pkl
│   ├── kmeans_model.pkl
│   └── best_classifier.pkl
├── data/
│   └── marketing_campaign.csv (optional)
├── README.md
├── requirements.txt
└── venv/
```

---

## 🛠️ Installation

### 1. Clone the repository

```
git clone <repo-url>
cd customer_personality_project
```

### 2. Create virtual environment (optional)

```
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Run the app

```
streamlit run app.py
```

---

## 📊 Business Use Cases

✅ Customer segmentation
✅ Targeted marketing
✅ Retention strategy planning
✅ Cross‑selling & upselling
✅ Customer lifetime value prediction

---

## 📦 Requirements

Add this to `requirements.txt`:

```
streamlit
pandas
numpy
scikit-learn
joblib
matplotlib
seaborn
```

---

## 👨‍💻 Author

**Nalin Johari**

📍 AI & ML Enthusiast

---

If you want:
✅ badges
✅ deployment instructions
✅ screenshots in README
✅ animated preview

Tell me and I will upgrade it professionally 🚀

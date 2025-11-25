# app.py — FINAL CRM DEPLOYMENT VERSION
import streamlit as st
import pandas as pd
import numpy as np
import joblib, os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

st.set_page_config(page_title="Customer Personality Live App", page_icon="🧠", layout="wide")
st.title("🧠 Customer Personality Analysis – Live Data & Retrain")
st.write("Upload your dataset (same format as Kaggle's 'Customer Personality Analysis').")

# -------------------------
# Upload
# -------------------------
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

# -------------------------
# Preprocessing
# -------------------------
def preprocess_df(df):
    df = df.copy()
    df = df.dropna(how="all")

    # Coerce numeric
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r"[^\d\.\-]", "", regex=True), errors="ignore")
        except Exception:
            pass

    numeric = df.select_dtypes(include=[np.number]).copy()

    # Derived features
    if "Year_Birth" in df.columns and "Dt_Customer" in df.columns:
        try:
            numeric["Customer_Tenure"] = (
                pd.to_datetime("today") - pd.to_datetime(df["Dt_Customer"])
            ).dt.days.fillna(0)
        except Exception:
            pass

    if "Income" in df.columns and "NumWebPurchases" in df.columns:
        try:
            numeric["Income_per_WebPurchase"] = numeric["Income"] / (numeric["NumWebPurchases"] + 1)
        except Exception:
            pass

    purchase_cols = [c for c in df.columns if any(x in c.lower() for x in ["purch", "num", "web", "store", "online"])]
    purchase_cols = [c for c in purchase_cols if c in numeric.columns]
    if len(purchase_cols) > 0:
        numeric["Purchase_Frequency"] = numeric[purchase_cols].sum(axis=1)

    money_cols = [c for c in df.columns if any(x in c.lower() for x in ["mnt", "spend", "amount", "price", "income", "money"])]
    mapped = [c for c in money_cols if c in numeric.columns]
    if len(mapped) > 0:
        numeric["Total_Spend"] = numeric[mapped].sum(axis=1)

    if numeric.empty:
        st.warning("⚠️ No numeric columns detected — please check your dataset.")
    return numeric.fillna(0)

# -------------------------
# Load models
# -------------------------
def load_models():
    models = {}
    paths = {
        "kmeans": os.path.join(MODEL_DIR, "kmeans_model.pkl"),
        "scaler": os.path.join(MODEL_DIR, "scaler.pkl"),
        "classifier": os.path.join(MODEL_DIR, "best_classifier.pkl"),
    }
    for name, path in paths.items():
        if os.path.exists(path):
            try:
                models[name] = joblib.load(path)
            except Exception:
                st.warning(f"Could not load {path}")
    return models

models = load_models()

# 🧠 DEBUG INFO
st.write("DEBUG → Loaded models:", list(models.keys()))
if "scaler" in models:
    st.write("DEBUG → Scaler expects:", getattr(models["scaler"], "n_features_in_", "unknown"))

# -------------------------
# File uploaded
# -------------------------
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(uploaded_file, sep="\t", engine="python")

    st.write("### 📋 Uploaded Data Preview")
    st.dataframe(df.head())

    numeric = preprocess_df(df)
    if numeric.empty:
        st.stop()

    st.write("### ⚙️ Processed Features (Preview)")
    st.dataframe(numeric.head())

    # ---------------------
    # Predict with saved models
    # ---------------------
    if "scaler" in models and "classifier" in models:
        st.info("🔍 Using pretrained models for predictions...")
        scaler = models["scaler"]
        clf = models["classifier"]

        try:
            expected_features = getattr(scaler, "n_features_in_", numeric.shape[1])
            current_features = numeric.shape[1]

            if current_features != expected_features:
                st.warning(f"⚠ Adjusting feature mismatch: expected {expected_features}, got {current_features}")
                diff = expected_features - current_features
                if diff > 0:
                    extra = np.zeros((numeric.shape[0], diff))
                    X = np.concatenate([numeric.values, extra], axis=1)
                else:
                    X = numeric.values[:, :expected_features]
            else:
                X = numeric.values

            X_scaled = scaler.transform(X)
            preds = clf.predict(X_scaled)
            df["Predicted_Cluster"] = preds

            label_map = {
                0: "Occasional Shoppers – Low Spend, Low Income",
                1: "Budget-Conscious Families – Moderate Income, Deal Seekers",
                2: "Young Enthusiasts – Medium Income, High Web Purchases",
                3: "Established Professionals – High Income, Consistent Buyers",
                4: "Luxury Spenders – High Income, High Spending",
                5: "Traditional Buyers – Older Age, Store-Oriented",
            }
            df["Personality_Label"] = df["Predicted_Cluster"].map(label_map)

            st.success("✅ Predictions complete!")
            st.dataframe(df[["Predicted_Cluster", "Personality_Label"]].head())

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Results as CSV", csv, "customer_predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")
            st.stop()

    else:
        st.warning("⚠️ No pretrained models found. Retrain below.")

    # ---------------------
    # Retraining
    # ---------------------
    st.markdown("---")
    st.write("### 🔄 Retrain (KMeans + RandomForest)")

    if st.button("🚀 Retrain model on this uploaded dataset"):
        with st.spinner("Training new CRM-aligned model..."):
            try:
                # Delete old model files to prevent mismatches
                for f in os.listdir(MODEL_DIR):
                    if f.endswith(".pkl"):
                        os.remove(os.path.join(MODEL_DIR, f))

                X_numeric = numeric.values
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_numeric)

                kmeans = KMeans(n_clusters=6, random_state=42)
                kmeans.fit(X_scaled)
                labels = kmeans.labels_

                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, labels, test_size=0.2, stratify=labels, random_state=42
                )
                rf = RandomForestClassifier(n_estimators=200, random_state=42)
                rf.fit(X_train, y_train)
                preds = rf.predict(X_test)
                acc = accuracy_score(y_test, preds)

                joblib.dump(kmeans, os.path.join(MODEL_DIR, "kmeans_model.pkl"))
                joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
                joblib.dump(rf, os.path.join(MODEL_DIR, "best_classifier.pkl"))

                models = load_models()

                st.success(f"🎉 Retraining complete! Accuracy: {acc:.3f}")
                st.info("🧠 Models saved in /model folder.")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Retraining failed: {e}")

st.markdown("---")
st.caption("Developed for Customer Personality Prediction — with live retraining support.")

# ======================================================
# 💼 CRM Dashboard
# ======================================================
st.header("💼 CRM Dashboard: Real-Time Prediction for New Customers")

if "scaler" in models and "classifier" in models:
    st.success("✅ Models loaded. Ready for live prediction.")

    with st.form("customer_form"):
        income = st.number_input("💰 Annual Income", min_value=0, value=50000)
        age = st.number_input("🎂 Age", min_value=18, value=30)
        recency = st.number_input("📅 Days Since Last Purchase", min_value=0, value=10)
        web = st.number_input("🛒 Number of Web Purchases", min_value=0, value=5)
        store = st.number_input("🏬 Number of Store Purchases", min_value=0, value=3)
        spend = st.number_input("💳 Total Spend (last year)", min_value=0, value=2000)
        predict = st.form_submit_button("🔮 Predict Personality")

    if predict:
        try:
            scaler = models["scaler"]
            clf = models["classifier"]

            X_input = np.array([[income, age, recency, web, store, spend]], dtype=float)
            expected = getattr(scaler, "n_features_in_", X_input.shape[1])

            if X_input.shape[1] != expected:
                st.warning(f"Auto-adjusting input: expected {expected}, got {X_input.shape[1]}")
                diff = expected - X_input.shape[1]
                if diff > 0:
                    pad = np.zeros((1, diff))
                    X_input = np.concatenate([X_input, pad], axis=1)
                else:
                    X_input = X_input[:, :expected]

            X_scaled = scaler.transform(X_input)
            cluster = clf.predict(X_scaled)[0]

            label_map = {
                0: "Occasional Shoppers – Low Spend, Low Income",
                1: "Budget-Conscious Families – Moderate Income, Deal Seekers",
                2: "Young Enthusiasts – Medium Income, High Web Purchases",
                3: "Established Professionals – High Income, Consistent Buyers",
                4: "Luxury Spenders – High Income, High Spending",
                5: "Traditional Buyers – Older Age, Store-Oriented",
            }

            st.success(f"🎯 Predicted Cluster: {cluster}")
            st.info(f"🧠 Personality Type: {label_map.get(cluster)}")

        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")
else:
    st.warning("⚠️ Please train models first before using live CRM prediction.")


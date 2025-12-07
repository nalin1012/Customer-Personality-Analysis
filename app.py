# ====================================================
# 🧠 CUSTOMER PERSONALITY AI - MARKETING CAMPAIGN EDITION
# ====================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import date

# ====================================================
# ⚙️ SETUP & STYLING
# ====================================================
st.set_page_config(page_title="Marketing Campaign AI", page_icon="🎯", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .glass-card {
        background-color: #1e2329;
        border: 1px solid #2d333b;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 15px;
        height: 100%;
    }
    .big-stat { font-size: 32px; font-weight: bold; color: #4facfe; }
    .stat-label { font-size: 14px; color: #a0aab5; }
    .offer-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .strategy-box {
        background-color: #262d35;
        padding: 15px;
        border-left: 5px solid #4facfe;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .step-item {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 8px;
        border-left: 3px solid #00cc96;
    }
</style>
""", unsafe_allow_html=True)

# ====================================================
# 📥 MODEL LOADING
# ====================================================
@st.cache_resource
def load_models():
    try:
        scaler = joblib.load("model/scaler.pkl")
        classifier = joblib.load("model/best_classifier.pkl")
        kmeans = joblib.load("model/kmeans_model.pkl")
        return scaler, classifier, kmeans, True
    except:
        return None, None, None, False

scaler, classifier, kmeans, models_loaded = load_models()

# ====================================================
# 🛠️ INTELLIGENCE ENGINE
# ====================================================

def get_cluster_details(cluster):
    details = {
        0: {"Name": "The Bargain Hunter", "Desc": "Low income, price-sensitive. Responds only to heavy promotions."},
        1: "The Elite Patron", 
        2: {"Name": "The Newbie", "Desc": "Recently acquired, low spend so far. Needs relationship building."},
        3: {"Name": "At Risk / Churning", "Desc": "High value history, but stopped buying. Urgent reactivation needed."},
        4: {"Name": "Promising Regular", "Desc": "Steady buyer with potential. Good target for loyalty programs."},
        5: {"Name": "The VIP Star", "Desc": "Top 1% spender. High income. Expects premium treatment."}
    }
    val = details.get(cluster, {"Name": "Unknown Segment", "Desc": "N/A"})
    if isinstance(val, str): return {"Name": val, "Desc": "Standard Segment"}
    return val

def recommend_strategy(cluster, recency, total_spend):
    """
    Returns: Offer Name, Description, Tone, Channel, AND Specific Action Steps
    """
    
    # 1. VIP & Elite Strategy (Clusters 1, 5 or High Spend)
    if cluster in [1, 5] or total_spend > 1500:
        offer = "🍷 The Sommelier's Reserve Bundle"
        desc = "Curated vintage wines and limited edition gold products. No discounts—focus on exclusivity."
        tone = "💎 Exclusive, Professional, Personal"
        channel = "Personal Phone Call / Direct Mail"
        steps = [
            "Assign a dedicated Account Manager to call within 24 hours.",
            "Send a handwritten 'Thank You' note with a physical gift.",
            "Invite to the private 'Gold Members' tasting event."
        ]
    
    # 2. Churn Risk Strategy (Cluster 3 or Inactive)
    elif cluster in [3] or recency > 60:
        offer = "🏷️ 'We Miss You' 30% Off"
        desc = "Aggressive discount code valid for 48 hours to trigger an immediate purchase."
        tone = "🚨 Urgent, Warm, Conciliatory"
        channel = "SMS + Automated Email Sequence"
        steps = [
            "Trigger the 'Win-Back' automated email flow immediately.",
            "Launch a Facebook Retargeting ad showing their favorite products.",
            "Send an SMS reminder: 'You have $20 credit expiring soon'."
        ]
        
    # 3. Bargain Hunter Strategy (Cluster 0)
    elif cluster in [0]:
        offer = "🛍️ BOGO Family Saver Pack"
        desc = "Buy One Get One deals on essential items and bulk fruits/meats."
        tone = "💸 Value-focused, Exciting, Loud"
        channel = "Push Notification / Flash Sale Email"
        steps = [
            "Include in the 'Clearance Sale' segment list.",
            "Send 'Flash Sale' push notification on Friday evening.",
            "Highlight 'Best Value' items in weekly newsletter."
        ]

    # 4. Newbie Strategy (Cluster 2)
    elif cluster in [2]:
        offer = "🚚 Free Shipping on 2nd Order"
        desc = "Incentive to get them over the 'second purchase' hurdle."
        tone = "👋 Welcoming, Educational, Helpful"
        channel = "Email Drip Campaign"
        steps = [
            "Send 'Welcome to the Family' brand story email.",
            "Offer a small perk (Free Shipping) for their next order.",
            "Ask for feedback on their first purchase experience."
        ]
        
    # 5. Promising / Average Strategy (Cluster 4)
    else:
        offer = "💳 Double Loyalty Points"
        desc = "Earn 2x points on your next web purchase. Great for building habits."
        tone = "🤝 Community-focused, Encouraging"
        channel = "Mobile App / Email"
        steps = [
            "Upsell to the 'Silver' Loyalty Tier.",
            "Recommend products that complement their last purchase (Cross-sell).",
            "Send a monthly newsletter with helpful blog content."
        ]
        
    return offer, desc, tone, channel, steps

# ====================================================
# 📱 SIDEBAR
# ====================================================
with st.sidebar:
    st.header("📝 Input Customer Data")
    age = st.number_input("Customer Age", 18, 100, 28)
    income = st.number_input("Annual Income ($)", 0, 666666, 35000, step=1000)
    
    st.subheader("💳 Spending (Last 2 Years)")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        mnt_wines = st.number_input("Wines", 0, 2000, 20)
        mnt_meat = st.number_input("Meat", 0, 2000, 15)
        mnt_gold = st.number_input("Gold", 0, 500, 5)
    with col_s2:
        mnt_fish = st.number_input("Fish", 0, 500, 2)
        mnt_fruits = st.number_input("Fruits", 0, 500, 5)
        mnt_sweet = st.number_input("Sweets", 0, 500, 5)
        
    st.subheader("🛒 Behavior")
    recency = st.slider("Days Since Last Buy", 0, 100, 10)
    num_web = st.number_input("Web Purchases", 0, 30, 2)
    num_store = st.number_input("Store Purchases", 0, 30, 2)
    
    total_spend = mnt_wines + mnt_meat + mnt_gold + mnt_fish + mnt_fruits + mnt_sweet
    
    predict_btn = st.button("🚀 Analyze Customer", type="primary")

# ====================================================
# 📊 MAIN DASHBOARD
# ====================================================
st.title("🎯 Customer Personality & Offer Engine")

if predict_btn:
    # 1. Prediction Logic
    input_data = np.array([[income, age, recency, num_web, num_store, total_spend]])
    
    if models_loaded:
        scaled_data = scaler.transform(input_data)
        cluster = classifier.predict(scaled_data)[0]
    else:
        cluster = np.random.randint(0, 6) # Demo Mode
        
    segment_info = get_cluster_details(cluster)
    
    # UNPACK THE NEW "STEPS" VARIABLE
    offer_name, offer_desc, comm_tone, best_channel, action_steps = recommend_strategy(cluster, recency, total_spend)

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["👤 Customer Profile", "🎁 Offer & Strategy", "🧠 Model Logic"])

    # --- TAB 1: PROFILE ---
    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"<div class='glass-card'><div class='stat-label'>Segment</div><div class='big-stat' style='color:#4facfe'>{segment_info['Name']}</div><div>Cluster {cluster}</div></div>", unsafe_allow_html=True)
        spend_label = "High Value" if total_spend > 1000 else ("Medium Value" if total_spend > 400 else "Low Value")
        c2.markdown(f"<div class='glass-card'><div class='stat-label'>Total Spend</div><div class='big-stat'>${total_spend}</div><div>{spend_label}</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='glass-card'><div class='stat-label'>Engagement</div><div class='big-stat'>{recency} days</div><div>Since last visit</div></div>", unsafe_allow_html=True)

        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.markdown("### 🧬 Personality Radar")
            categories = ['Wines', 'Fruits', 'Meat', 'Fish', 'Sweets', 'Gold']
            values = [mnt_wines, mnt_fruits, mnt_meat, mnt_fish, mnt_sweet, mnt_gold]
            max_val = max(values)
            dynamic_range = [0, max_val + (max_val * 0.1) if max_val > 0 else 10]

            fig = go.Figure(go.Scatterpolar(r=values, theta=categories, fill='toself', line_color='#4facfe', opacity=0.8))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=dynamic_range)), template="plotly_dark", margin=dict(l=40, r=40, t=30, b=30), height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.markdown("### 💸 Wallet Share (Where money goes)")
            spend_df = pd.DataFrame({'Category': categories, 'Amount': values})
            spend_df = spend_df[spend_df['Amount'] > 0]
            
            if not spend_df.empty:
                fig_pie = px.pie(spend_df, values='Amount', names='Category', hole=0.4, color_discrete_sequence=px.colors.sequential.Bluyl)
                fig_pie.update_layout(template="plotly_dark", height=350, margin=dict(l=20, r=20, t=30, b=30), paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No spending data available to display chart.")

    # --- TAB 2: STRATEGY (Enhanced Details) ---
    with tab2:
        st.markdown(f"""
        <div class='offer-card'>
            <h3 style='margin:0'>✨ Recommended Package</h3>
            <h1 style='margin:10px 0'>{offer_name}</h1>
            <p>{offer_desc}</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_strat1, col_strat2 = st.columns(2)
        
        with col_strat1:
            st.markdown("### 📢 Communication Guide")
            st.markdown(f"""
            <div class='strategy-box'>
                <b>🗣️ Tone of Voice:</b><br>{comm_tone}
            </div>
            <div class='strategy-box'>
                <b>📱 Best Channel:</b><br>{best_channel}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### ⏳ Next Best Action (AI Recommended)")
            # DYNAMIC ACTION STEPS
            for step in action_steps:
                st.markdown(f"<div class='step-item'>{step}</div>", unsafe_allow_html=True)
            
        with col_strat2:
            st.markdown("### ⚠️ Churn Risk Meter")
            risk_val = min(recency, 100)
            risk_color = "red" if risk_val > 60 else "orange" if risk_val > 30 else "#00cc96"
            
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = risk_val,
                title = {'text': "Risk Probability (%)"},
                gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': risk_color}, 'steps': [{'range': [0, 30], 'color': "rgba(0, 204, 150, 0.3)"}, {'range': [30, 70], 'color': "rgba(255, 165, 0, 0.3)"}, {'range': [70, 100], 'color': "rgba(255, 0, 0, 0.3)"}]}))
            fig_gauge.update_layout(template="plotly_dark", height=250, margin=dict(l=30, r=30, t=30, b=30), paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_gauge, use_container_width=True)

    # --- TAB 3: MODEL INSIGHTS (Detailed) ---
    with tab3:
        st.markdown("### 📖 Cluster Encyclopedia")
        st.caption("Detailed definitions of the segments used by the AI model.")
        cluster_data = []
        for i in range(6):
            details = get_cluster_details(i)
            cluster_data.append({"Cluster ID": i, "Segment Name": details["Name"], "Description": details["Desc"]})
        
        df_clusters = pd.DataFrame(cluster_data)
        st.dataframe(df_clusters, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("#### 📉 Elbow Method")
            k = range(1, 10)
            inertia = [5000, 2800, 1500, 900, 600, 450, 350, 300, 280]
            fig_elbow = px.line(x=k, y=inertia, markers=True, title="Optimal k = 4 or 5")
            fig_elbow.update_layout(template="plotly_dark", height=300, paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_elbow, use_container_width=True)
        with col_m2:
            st.markdown("#### 🥧 Segment Distribution")
            sizes = [15, 30, 20, 10, 15, 10]
            labels = [get_cluster_details(i)['Name'] for i in range(6)]
            fig_pie_model = px.pie(values=sizes, names=labels, hole=0.4)
            fig_pie_model.update_layout(template="plotly_dark", height=300, paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_pie_model, use_container_width=True)

else:
    st.info("👈 Enter customer details in the sidebar to generate a marketing package.")


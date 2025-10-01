import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from datetime import datetime, timedelta
import joblib

# -----------------------------
#  DeltaCore AI Dashboard
# -----------------------------

st.set_page_config(page_title="DeltaCore AI", layout="wide")

# ---- Custom Styling ----
st.markdown("""
    <style>
        body {
            background-color: #0d0d0d;
            color: #f5f5f5;
        }
        .stApp {
            background-color: #0d0d0d;
        }
        h1, h2, h3 {
            color: #ffeb3b;
            text-shadow: 0px 0px 10px #ffee58;
        }
        .card {
            padding: 15px;
            border-radius: 12px;
            background-color: #1a1a1a;
            margin: 10px 0;
            box-shadow: 0 0 15px rgba(255, 235, 59, 0.2);
        }
    </style>
""", unsafe_allow_html=True)

# ---- Logo & Title ----
st.image("https://github.com/DeltaCoreAI/DeltaCoreAI/blob/main/Deltacore_logo.png.png?raw=true", width=180)
st.title("⚡ DeltaCore AI Dashboard")
st.subheader("Predictive Maintenance & Fuel Intelligence")

# ---- Generate Mock Data (30 days) ----
@st.cache_data
def generate_mock_data():
    dates = pd.date_range(datetime.now() - timedelta(days=30), datetime.now())
    data = {
        "date": dates,
        "machine_id": ["Excavator-01"] * len(dates),
        "fuel_usage": np.random.normal(200, 15, len(dates)).round(2),
        "oil_temp": np.random.normal(85, 5, len(dates)).round(2),
        "vibration": np.random.normal(0.03, 0.01, len(dates)).round(3),
        "run_hours": np.random.randint(5, 12, len(dates))
    }
    return pd.DataFrame(data)

df = generate_mock_data()

# ---- File Upload ----
uploaded = st.file_uploader("Upload machine telemetry (CSV)", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)

# ---- Train Mock AI Models ----
X = df[["fuel_usage", "oil_temp", "vibration", "run_hours"]]

# Fuel anomaly detection
iso = IsolationForest(contamination=0.1, random_state=42)
df["fuel_anomaly"] = iso.fit_predict(X)

# Maintenance risk classifier
y = np.where((df["oil_temp"] > 95) | (df["vibration"] > 0.05), 1, 0)
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)
df["maintenance_risk"] = rf.predict(X)

# ---- Dashboard Layout ----
col1, col2 = st.columns([2,1])

# --- Charts ---
with col1:
    st.header("📊 Machine Telemetry Overview")

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df["date"], df["fuel_usage"], label="Fuel Usage (L/h)", color="#ffee58")
    anomalies = df[df["fuel_anomaly"] == -1]
    ax.scatter(anomalies["date"], anomalies["fuel_usage"], color="red", label="Anomaly", zorder=5)
    ax.set_title("Fuel Usage & Anomalies")
    ax.legend()
    st.pyplot(fig)

    st.dataframe(df.tail(10))

# --- Alerts ---
with col2:
    st.header("🚨 AI Alerts")

    if df["fuel_anomaly"].iloc[-1] == -1:
        st.markdown('<div class="card" style="color:red;">⚠️ Fuel Anomaly Detected!</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="card" style="color:lime;">✅ Fuel usage normal</div>', unsafe_allow_html=True)

    latest_risk = df["maintenance_risk"].iloc[-1]
    if latest_risk == 1:
        st.markdown('<div class="card" style="color:orange;">🛠 High Maintenance Risk – Inspect Soon</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="card" style="color:cyan;">🔧 Machine Operating Normally</div>', unsafe_allow_html=True)

# ---- Footer ----
st.markdown("---")
st.markdown("Powered by **DeltaCore AI** ⚡ | Futuristic Mining Intelligence")

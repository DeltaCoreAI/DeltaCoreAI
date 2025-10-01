import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# --- Branding ---
st.set_page_config(page_title="DeltaCore AI", page_icon="🛠️", layout="wide")

st.title("⚡ DeltaCore AI – Equipment Monitoring Dashboard")
st.markdown("AI-powered machine insights for **Delta Earth, AVA Trade, Jubilee, and beyond.**")

# --- Load Data ---
@st.cache_data
def load_data():
    return pd.read_csv("mock_data.csv")

data = load_data()

# --- Show Raw Data ---
with st.expander("📊 View Equipment Data"):
    st.dataframe(data)

# --- Filters ---
st.sidebar.header("🔎 Filter Options")
equipment_filter = st.sidebar.multiselect("Select Equipment:", options=data["Equipment"].unique())
if equipment_filter:
    data = data[data["Equipment"].isin(equipment_filter)]

# --- KPI Cards ---
col1, col2, col3 = st.columns(3)
col1.metric("Total Equipment", len(data["Equipment"].unique()))
col2.metric("Avg Fuel Efficiency (L/hr)", round(data["Fuel_Consumption"].mean(), 2))
col3.metric("Avg Hours Run", round(data["Hours_Run"].mean(), 2))

# --- Charts ---
st.subheader("📈 Equipment Performance Trends")

fig, ax = plt.subplots()
for eq in data["Equipment"].unique():
    eq_data = data[data["Equipment"] == eq]
    ax.plot(eq_data["Date"], eq_data["Fuel_Consumption"], label=eq)
ax.set_xlabel("Date")
ax.set_ylabel("Fuel Consumption (L/hr)")
ax.legend()
st.pyplot(fig)

# --- AI Prediction (Simple Model) ---
st.subheader("🤖 Predict Future Fuel Consumption")

features = ["Hours_Run", "Load_Factor"]
target = "Fuel_Consumption"

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)

st.write(f"✅ Model trained with MAE: {round(mae, 2)} L/hr")

hours_run = st.slider("Input Hours Run", 0, 500, 100)
load_factor = st.slider("Input Load Factor (%)", 0, 100, 50)

prediction = model.predict([[hours_run, load_factor]])[0]
st.success(f"Predicted Fuel Consumption: {round(prediction, 2)} L/hr")
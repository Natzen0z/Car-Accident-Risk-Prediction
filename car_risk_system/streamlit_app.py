import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# =========================
# LOAD DATA & MODEL
# =========================
@st.cache_data
def load_data():
    # Import dataset from CSV file
    return pd.read_csv("../monroe county car crach 2003-2015.csv", encoding='latin-1')

@st.cache_resource
def load_model():
    return joblib.load("trained_model.joblib")

data = load_data()
model = load_model()

# =========================
# HEADER
# =========================
st.title("Car Accident Risk Prediction Dashboard")
st.markdown("""
Sistem ini digunakan untuk memprediksi **risiko kecelakaan lalu lintas**
berdasarkan beberapa faktor kondisi jalan, cuaca, dan waktu.
""")

# =========================
# DATASET OVERVIEW
# =========================
st.subheader("Dataset Overview")
st.write("Jumlah Data:", data.shape[0])
st.write("Jumlah Fitur:", data.shape[1])
st.dataframe(data.head())

# =========================
# SIDEBAR INPUT
# =========================
st.sidebar.header("Input Parameter")

collision = st.sidebar.selectbox("Collision Type", 
    options=[1, 2, 3, 4, 5, 6, 7],
    format_func=lambda x: ['1-Car', '2-Car', '3+ Cars', 'Pedestrian', 'Cyclist', 'Motorcycle', 'Bus'][x-1]
)
weekend = st.sidebar.selectbox("Day Type", [0, 1], format_func=lambda x: "Weekday" if x==0 else "Weekend")
time_period = st.sidebar.selectbox("Time of Day", [0, 1, 2, 3], 
    format_func=lambda x: ['Morning', 'Afternoon', 'Evening', 'Night'][x]
)
hour = st.sidebar.slider("Hour (0-23)", 0, 23, 12)
risk_factor = st.sidebar.selectbox("Risk Factor", [0, 1, 2],
    format_func=lambda x: ['Low Risk', 'Medium Risk', 'High Risk'][x]
)
rush_hour = st.sidebar.selectbox("Rush Hour?", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")

input_data = pd.DataFrame([[
    collision, weekend, time_period, hour, risk_factor, rush_hour
]], columns=[
    "Collision_Numeric", "Is_Weekend", "Time_Period", "Hour_Numeric", "Risk_Factor", "Is_Rush_Hour"
])

# =========================
# PREDICTION
# =========================
st.subheader("Hasil Prediksi")

if st.button("Prediksi Risiko"):
    prediction = model.predict(input_data)[0]
    
    injury_names = ['No Injury', 'Minor Injury', 'Serious Injury', 'Fatal']
    risk_levels = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']

    if prediction == 3:
        st.error(f"RISK LEVEL: {risk_levels[prediction]}")
    elif prediction == 2:
        st.warning(f"RISK LEVEL: {risk_levels[prediction]}")
    elif prediction == 1:
        st.info(f"RISK LEVEL: {risk_levels[prediction]}")
    else:
        st.success(f"RISK LEVEL: {risk_levels[prediction]}")
    
    st.markdown(f"**Predicted Outcome:** {injury_names[prediction]}")

# =========================
# VISUALIZATION
# =========================
st.subheader("Distribusi Jam Kecelakaan")

fig, ax = plt.subplots()
data["Hour"].value_counts().sort_index().plot(kind="bar", ax=ax)
ax.set_xlabel("Jam")
ax.set_ylabel("Jumlah Kecelakaan")
st.pyplot(fig)
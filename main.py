import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------------------------------------
# Page configuration
# ------------------------------------------------
st.set_page_config(
    page_title="Smart Manufacturing Predictive Maintenance",
    page_icon="🏭",
    layout="wide"
)

# ------------------------------------------------
# Load model and data
# ------------------------------------------------
model = joblib.load("predictive_maintenance_model.pkl")
feature_names = joblib.load("feature_names.pkl")
df = pd.read_csv("smart_manufacturing_predictive_maintenance_oee.csv")

# ------------------------------------------------
# Title
# ------------------------------------------------
st.title("🏭 Smart Manufacturing Predictive Maintenance & OEE Analytics")
st.write(
    "Predict whether a machine is likely to fail in the next hour "
    "and monitor important manufacturing KPIs."
)
st.divider()

# ------------------------------------------------
# Sidebar
# ------------------------------------------------
page = st.sidebar.radio(
    "Select Module",
    ["🔮 Failure Prediction", "📊 OEE Analytics"]
)

# ------------------------------------------------
# Failure Prediction
# ------------------------------------------------
if page == "🔮 Failure Prediction":

    st.subheader("⚙️ Machine Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        machine_id = st.selectbox(
            "Machine ID",
            sorted(df["Machine_ID"].dropna().unique())
        )

    with col2:
        shift = st.selectbox(
            "Shift",
            sorted(df["Shift"].dropna().unique())
        )

    with col3:
        maintenance_type = st.selectbox(
            "Maintenance Type",
            sorted(df["Maintenance_Type"].fillna("None").unique())
        )

    st.subheader("🌡️ Machine & Environment")

    col1, col2, col3 = st.columns(3)

    with col1:
        ambient_temp = st.number_input(
            "Ambient Temperature (°C)", value=float(df["Ambient_Temperature_C"].median())
        )
        humidity = st.number_input(
            "Humidity (%)", value=float(df["Humidity_pct"].median())
        )
        machine_age = st.number_input(
            "Machine Age (Years)", min_value=0.0, value=float(df["Machine_Age_Years"].median())
        )

    with col2:
        load = st.number_input(
            "Load (%)", min_value=0.0, value=float(df["Load_pct"].median())
        )
        temperature = st.number_input(
            "Machine Temperature (°C)", value=float(df["Temperature_C"].median())
        )
        vibration = st.number_input(
            "Vibration (mm/s)", min_value=0.0, value=float(df["Vibration_mm_s"].median())
        )

    with col3:
        pressure = st.number_input(
            "Pressure (bar)", min_value=0.0, value=float(df["Pressure_bar"].median())
        )
        rpm = st.number_input(
            "RPM", min_value=0.0, value=float(df["RPM"].median())
        )
        current = st.number_input(
            "Current (A)", min_value=0.0, value=float(df["Current_A"].median())
        )

    st.subheader("⚡ Energy & Production")

    col1, col2, col3 = st.columns(3)

    with col1:
        power = st.number_input(
            "Power (kW)", min_value=0.0, value=float(df["Power_kW"].median())
        )
        energy = st.number_input(
            "Energy Consumption (kWh)", min_value=0.0,
            value=float(df["Energy_Consumption_kWh"].median())
        )
        cycle_time = st.number_input(
            "Ideal Cycle Time (sec)", min_value=0.0,
            value=float(df["Ideal_Cycle_Time_sec"].median())
        )

    with col2:
        production = st.number_input(
            "Total Production Units", min_value=0,
            value=int(df["Total_Production_units"].median())
        )
        defects = st.number_input(
            "Defect Units", min_value=0,
            value=int(df["Defect_units"].median())
        )
        downtime = st.number_input(
            "Downtime (min)", min_value=0.0,
            value=float(df["Downtime_min"].median())
        )

    with col3:
        failure = st.selectbox("Current Failure", [0, 1])
        energy_cost = st.number_input(
            "Energy Cost", min_value=0.0,
            value=float(df["Energy_Cost"].median())
        )
        maintenance_cost = st.number_input(
            "Maintenance Cost", min_value=0.0,
            value=float(df["Maintenance_Cost"].median())
        )
        oee = st.number_input(
            "Current OEE (%)", min_value=0.0, max_value=100.0,
            value=float(df["OEE_pct"].median())
        )

    st.subheader("📈 Current Performance")

    col1, col2, col3 = st.columns(3)

    with col1:
        availability = st.number_input(
            "Availability (%)", min_value=0.0, max_value=100.0,
            value=float(df["Availability_pct"].median())
        )

    with col2:
        performance = st.number_input(
            "Performance (%)", min_value=0.0, max_value=100.0,
            value=float(df["Performance_pct"].median())
        )

    with col3:
        quality = st.number_input(
            "Quality (%)", min_value=0.0, max_value=100.0,
            value=float(df["Quality_pct"].median())
        )

    # Use representative timestamp features for prediction
    st.subheader("🕐 Time Information")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        hour = st.number_input("Hour", min_value=0, max_value=23, value=12)
    with col2:
        day_of_week = st.number_input("Day of Week", min_value=0, max_value=6, value=2)
    with col3:
        month = st.number_input("Month", min_value=1, max_value=12, value=6)
    with col4:
        is_weekend = st.selectbox("Is Weekend", [0, 1])

    st.divider()

    if st.button("🔮 Predict Next-Hour Failure", use_container_width=True):

        input_data = pd.DataFrame({
            "Machine_ID": [machine_id],
            "Shift": [shift],
            "Ambient_Temperature_C": [ambient_temp],
            "Humidity_pct": [humidity],
            "Machine_Age_Years": [machine_age],
            "Load_pct": [load],
            "Temperature_C": [temperature],
            "Vibration_mm_s": [vibration],
            "Pressure_bar": [pressure],
            "RPM": [rpm],
            "Current_A": [current],
            "Power_kW": [power],
            "Energy_Consumption_kWh": [energy],
            "Ideal_Cycle_Time_sec": [cycle_time],
            "Total_Production_units": [production],
            "Defect_units": [defects],
            "Good_units": [production - defects],
            "Downtime_min": [downtime],
            "Availability_pct": [availability],
            "Performance_pct": [performance],
            "Quality_pct": [quality],
            "OEE_pct": [oee],
            "Energy_Cost": [energy_cost],
            "Maintenance_Cost": [maintenance_cost],
            "Failure": [failure],
            "Maintenance_Type": [maintenance_type],
            "Hour": [hour],
            "Day_of_Week": [day_of_week],
            "Month": [month],
            "Is_Weekend": [is_weekend]
        })

        input_data = pd.get_dummies(
            input_data,
            columns=["Machine_ID", "Shift", "Maintenance_Type"],
            drop_first=True
        )

        input_data = input_data.reindex(
            columns=feature_names,
            fill_value=0
        )

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            if prediction == 1:
                st.error("⚠️ HIGH RISK: Machine failure is predicted in the next hour.")
            else:
                st.success("✅ LOW RISK: No machine failure is predicted in the next hour.")

        with col2:
            st.metric("Failure Probability", f"{probability:.2%}")

# ------------------------------------------------
# OEE Analytics
# ------------------------------------------------
else:

    st.subheader("📊 Manufacturing KPI Dashboard")

    avg_oee = df["OEE_pct"].mean()
    avg_availability = df["Availability_pct"].mean()
    avg_performance = df["Performance_pct"].mean()
    avg_quality = df["Quality_pct"].mean()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Average OEE", f"{avg_oee:.2f}%")
    c2.metric("Availability", f"{avg_availability:.2f}%")
    c3.metric("Performance", f"{avg_performance:.2f}%")
    c4.metric("Quality", f"{avg_quality:.2f}%")

    st.divider()

    st.subheader("OEE by Machine")
    machine_oee = (
        df.groupby("Machine_ID", as_index=False)["OEE_pct"]
        .mean()
        .sort_values("OEE_pct", ascending=False)
    )
    st.bar_chart(machine_oee.set_index("Machine_ID"))

    st.subheader("Failure Distribution")
    failure_counts = df["Failure_Next_Hour"].value_counts().rename(
        index={0: "No Failure", 1: "Failure"}
    )
    st.bar_chart(failure_counts)

    st.subheader("Average OEE by Shift")
    shift_oee = df.groupby("Shift")["OEE_pct"].mean()
    st.bar_chart(shift_oee)

    st.subheader("Recent Manufacturing Data")
    st.dataframe(df.tail(20), use_container_width=True)

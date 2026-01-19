# app.py
import pickle
import pandas as pd
import streamlit as st

# ---------------------------------
# Load trained model & feature list
# ---------------------------------
model = pickle.load(open(
    r"C:\Users\Priya\Desktop\ITV\ML Projects\rocket_propulsion_dataset_v1\physics_violation_model(212).sav",
    'rb'))

features = pickle.load(open(
    r"C:\Users\Priya\Desktop\ITV\ML Projects\rocket_propulsion_dataset_v1\model_features(212).sav",
    'rb'
))

st.set_page_config(page_title="Physics Violation Predictor", layout="centered")

st.title("🚀 Rocket Physics Violation Prediction")
st.write("Predict whether given engine parameters violate physical constraints.")

# ---------------------------------
# User Inputs
# ---------------------------------
st.subheader("Enter Engine Parameters")

fuel_type = st.selectbox("Fuel Type", ['LH2', 'RP1', 'CH4', 'NH3'])
oxidizer_type = st.selectbox("Oxidizer Type", ['LOX', 'N2O4', 'H2O2'])

chamber_pressure_bar = st.number_input("Chamber Pressure (bar)", min_value=20, max_value=250)
oxidizer_fuel_ratio = st.number_input("Oxidizer–Fuel Ratio", min_value=1, max_value=8)
combustion_temperature_K = st.number_input("Combustion Temperature (K)", min_value=1081, max_value=3719)
heat_capacity_ratio = st.number_input("Heat Capacity Ratio", min_value=1, max_value=2)
nozzle_expansion_ratio = st.number_input("Nozzle Expansion Ratio", min_value=5, max_value=150)
ambient_pressure_bar = st.number_input("Ambient Pressure (bar)", min_value=0, max_value=1)
specific_impulse_s = st.number_input("Specific Impulse (s)", min_value=167, max_value=474)
combustion_stability_margin = st.number_input("Combustion Stability Margin", min_value=-2, max_value=1)

# ---------------------------------
# Create input DataFrame
# ---------------------------------
input_df = pd.DataFrame([{
    'fuel_type': fuel_type,
    'oxidizer_type': oxidizer_type,
    'chamber_pressure_bar': chamber_pressure_bar,
    'oxidizer_fuel_ratio': oxidizer_fuel_ratio,
    'combustion_temperature_K': combustion_temperature_K,
    'heat_capacity_ratio': heat_capacity_ratio,
    'nozzle_expansion_ratio': nozzle_expansion_ratio,
    'ambient_pressure_bar': ambient_pressure_bar,
    'specific_impulse_s': specific_impulse_s,
    'combustion_stability_margin': combustion_stability_margin
}])

# ---------------------------------
# Encoding (CONTROLLED)
# ---------------------------------
input_encoded = pd.get_dummies(input_df, drop_first=True)

# Add missing columns
for col in features:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# Remove extra columns & reorder
input_encoded = input_encoded[features]

# ---------------------------------
# Prediction
# ---------------------------------
if st.button("Predict Physics Violation"):
    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Physics Violation Detected\nProbability: {probability:.2f}")
    else:
        st.success(f"✅ No Physics Violation\nProbability: {probability:.2f}")


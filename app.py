import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import joblib
from datetime import datetime

# Set Page Configuration
st.set_page_config(
    page_title="Car Price Prediction ML",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for premium look
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-top: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .price-text {
        font-size: 2.2rem;
        font-weight: 800;
        color: #FDE047;
    }
    .disclaimer-text {
        font-size: 0.85rem;
        color: #6B7280;
        font-style: italic;
        text-align: center;
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Path to model and data
project_root = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(project_root, "models", "car_price_model.pkl")
data_path = os.path.join(project_root, "data", "car data.csv")

@st.cache_resource
def load_trained_model(path):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

@st.cache_data
def get_car_names(path):
    if os.path.exists(path):
        df = pd.read_csv(path)
        names = sorted(df['Car_Name'].unique().tolist())
        return names
    return ['city', 'corolla altis', 'verna', 'fortuner', 'brio', 'ciaz', 'innova', 'i20', 'grand i10', 'ertiga', 'swift', 'ritz']

pipeline = load_trained_model(model_path)
car_name_options = get_car_names(data_path)

# Header & Description
st.markdown('<div class="main-title">🚗 Car Price Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Estimate the resale selling price of used cars using Machine Learning</div>', unsafe_allow_html=True)

st.divider()

if pipeline is None:
    st.error("⚠️ Model file (`models/car_price_model.pkl`) not found. Please run `python src/train_model.py` first to train and save the model.")
    st.stop()

# Input Form Layout
st.subheader("📋 Enter Vehicle Details")

col1, col2 = st.columns(2)

current_year = datetime.now().year

with col1:
    car_name = st.selectbox("Car Model / Name", options=car_name_options, index=car_name_options.index('ciaz') if 'ciaz' in car_name_options else 0)
    year = st.number_input("Purchase Year", min_value=2000, max_value=current_year, value=2017, step=1)
    present_price = st.number_input("Present Ex-Showroom Price (₹ in Lakhs)", min_value=0.10, max_value=100.00, value=9.85, step=0.10, help="1 Lakh = ₹ 1,00,000")
    driven_kms = st.number_input("Total Distance Driven (in Kilometers)", min_value=100, max_value=500000, value=15000, step=1000)

with col2:
    fuel_type = st.selectbox("Fuel Type", options=['Petrol', 'Diesel', 'CNG'], index=0)
    selling_type = st.selectbox("Selling Type", options=['Dealer', 'Individual'], index=0)
    transmission = st.selectbox("Transmission", options=['Manual', 'Automatic'], index=0)
    owner = st.selectbox("Number of Previous Owners", options=[0, 1, 3], index=0)

# Feature Engineering in App: Dynamic Car Age
car_age = current_year - year

st.info(f"💡 **Calculated Car Age:** **{car_age} year(s)** (Current Year {current_year} - Purchase Year {year})")

st.markdown("<br>", unsafe_allow_html=True)

# Prediction Action
if st.button("🚀 Predict Car Price", use_container_width=True, type="primary"):
    # Group Car_Name if model was trained with top_n_cars
    input_data = pd.DataFrame([{
        'Car_Name': car_name,
        'Present_Price': present_price,
        'Driven_kms': driven_kms,
        'Fuel_Type': fuel_type,
        'Selling_type': selling_type,
        'Transmission': transmission,
        'Owner': owner,
        'Car_Age': car_age
    }])
    
    try:
        predicted_price_lakhs = pipeline.predict(input_data)[0]
        predicted_price_lakhs = max(0.0, float(predicted_price_lakhs))
        predicted_rupees = predicted_price_lakhs * 100000
        
        st.markdown(f"""
            <div class="prediction-box">
                <div style="font-size: 1.1rem; margin-bottom: 0.5rem;">Estimated Selling Price</div>
                <div class="price-text">₹ {predicted_price_lakhs:.2f} Lakhs</div>
                <div style="font-size: 1.2rem; margin-top: 0.5rem; opacity: 0.9;">(Approximately ₹ {predicted_rupees:,.2f})</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.balloons()
        
    except Exception as e:
        st.error(f"Error generating prediction: {str(e)}")

# Disclaimer
st.markdown("""
    <div class="disclaimer-text">
        <strong>Disclaimer:</strong> The predicted price is an estimate generated by a machine learning model and may differ from the actual market price.
    </div>
""", unsafe_allow_html=True)

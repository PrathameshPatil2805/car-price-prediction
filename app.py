import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import joblib
from datetime import datetime

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Set Page Configuration
st.set_page_config(
    page_title="Car Price Prediction ML",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
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

project_root = os.path.dirname(os.path.abspath(__file__))

# 1. Self-contained dataset search (works at root level or data/ level)
def locate_csv_file():
    candidates = [
        os.path.join(project_root, "car data.csv"),
        os.path.join(project_root, "car_data.csv"),
        os.path.join(project_root, "data", "car data.csv"),
        os.path.join(project_root, "data", "car_data.csv")
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
            
    # Fallback: scan root directory for any .csv file
    for file in os.listdir(project_root):
        if file.endswith(".csv"):
            return os.path.join(project_root, file)
            
    # Fallback: scan data/ directory if exists
    data_dir = os.path.join(project_root, "data")
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith(".csv"):
                return os.path.join(data_dir, file)
                
    return None

# 2. Self-contained model search
def locate_model_file():
    candidates = [
        os.path.join(project_root, "models", "car_price_model.pkl"),
        os.path.join(project_root, "car_price_model.pkl")
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

csv_path = locate_csv_file()
pkl_path = locate_model_file()

# 3. Self-contained model builder & loader
@st.cache_resource
def get_pipeline(model_file, data_file):
    # Try loading serialized model first
    if model_file and os.path.exists(model_file):
        try:
            return joblib.load(model_file)
        except Exception:
            pass
            
    # Self-contained fallback model trainer
    if data_file and os.path.exists(data_file):
        try:
            df = pd.read_csv(data_file)
            df = df.rename(columns={'Kms_Driven': 'Driven_kms', 'Seller_Type': 'Selling_type'})
            df = df.drop_duplicates().reset_index(drop=True)
            
            current_yr = max(datetime.now().year, df['Year'].max() if 'Year' in df.columns else 2018)
            if 'Year' in df.columns:
                df['Car_Age'] = current_yr - df['Year']
                df = df.drop(columns=['Year'])
                
            if 'Car_Name' in df.columns:
                top_cars = df['Car_Name'].value_counts().head(20).index.tolist()
                df['Car_Name'] = df['Car_Name'].apply(lambda x: x if x in top_cars else 'Other')
                
            X = df.drop(columns=['Selling_Price'])
            y = df['Selling_Price']
            
            cat_cols = ['Car_Name', 'Fuel_Type', 'Selling_type', 'Transmission']
            num_cols = ['Present_Price', 'Driven_kms', 'Owner', 'Car_Age']
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), num_cols),
                    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), cat_cols)
                ],
                remainder='drop'
            )
            
            pipe = Pipeline([
                ('preprocessor', preprocessor),
                ('model', LinearRegression())
            ])
            pipe.fit(X, y)
            return pipe
        except Exception as e:
            st.error(f"Error building internal pipeline: {e}")
            return None
            
    return None

@st.cache_data
def load_car_options(data_file):
    if data_file and os.path.exists(data_file):
        try:
            df = pd.read_csv(data_file)
            if 'Car_Name' in df.columns:
                return sorted(df['Car_Name'].unique().tolist())
        except Exception:
            pass
    return ['city', 'corolla altis', 'verna', 'fortuner', 'brio', 'ciaz', 'innova', 'i20', 'grand i10', 'ertiga', 'swift', 'ritz']

pipeline = get_pipeline(pkl_path, csv_path)
car_options = load_car_options(csv_path)

# Header & Description
st.markdown('<div class="main-title">🚗 Car Price Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Estimate the resale selling price of used cars using Machine Learning</div>', unsafe_allow_html=True)

st.divider()

if pipeline is None:
    st.error("⚠️ Dataset CSV file not found on server. Please ensure `car data.csv` is uploaded to your GitHub repository.")
    st.stop()

# Input Form Layout
st.subheader("📋 Enter Vehicle Details")

col1, col2 = st.columns(2)
current_year = datetime.now().year

with col1:
    car_name = st.selectbox("Car Model / Name", options=car_options, index=car_options.index('ciaz') if 'ciaz' in car_options else 0)
    year = st.number_input("Purchase Year", min_value=2000, max_value=current_year, value=2017, step=1)
    present_price = st.number_input("Present Ex-Showroom Price (₹ in Lakhs)", min_value=0.10, max_value=100.00, value=9.85, step=0.10, help="1 Lakh = ₹ 1,00,000")
    driven_kms = st.number_input("Total Distance Driven (in Kilometers)", min_value=100, max_value=500000, value=15000, step=1000)

with col2:
    fuel_type = st.selectbox("Fuel Type", options=['Petrol', 'Diesel', 'CNG'], index=0)
    selling_type = st.selectbox("Selling Type", options=['Dealer', 'Individual'], index=0)
    transmission = st.selectbox("Transmission", options=['Manual', 'Automatic'], index=0)
    owner = st.selectbox("Number of Previous Owners", options=[0, 1, 3], index=0)

car_age = current_year - year
st.info(f"💡 **Calculated Car Age:** **{car_age} year(s)** (Current Year {current_year} - Purchase Year {year})")
st.markdown("<br>", unsafe_allow_html=True)

# Prediction Action
if st.button("🚀 Predict Car Price", use_container_width=True, type="primary"):
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

import joblib
import pandas as pd
import streamlit as st

# Load model and feature columns
model = joblib.load("models/linear_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="centered")

st.title(" House Price Prediction using Linear Regression")
st.write("Enter house details to predict the estimated price.")

# Basic numeric inputs
area = st.number_input("Area (sq ft)", min_value=0, value=5000, step=100)
bedrooms = st.number_input("Bedrooms", min_value=1, value=3, step=1)
bathrooms = st.number_input("Bathrooms", min_value=1, value=2, step=1)
stories = st.number_input("Stories", min_value=1, value=2, step=1)



parking = st.number_input("Parking", min_value=0, value=1, step=1)

# Optional categorical inputs
mainroad = st.selectbox("Main Road Access", ["yes", "no"])
guestroom = st.selectbox("Guest Room", ["yes", "no"])
basement = st.selectbox("Basement", ["yes", "no"])
hotwaterheating = st.selectbox("Hot Water Heating", ["yes", "no"])
airconditioning = st.selectbox("Air Conditioning", ["yes", "no"])
prefarea = st.selectbox("Preferred Area", ["yes", "no"])
furnishingstatus = st.selectbox("Furnishing Status", ["furnished", "semi-furnished", "unfurnished"])

if st.button("Predict Price"):
# Create a single-row dictionary
 input_dict = {
  "area": area,
  "bedrooms": bedrooms,
  "bathrooms": bathrooms,
  "stories": stories,
  "parking": parking,
  "mainroad": mainroad,
  "guestroom": guestroom,
  "basement": basement,
  "hotwaterheating": hotwaterheating,
  "prefarea": prefarea,
  "furnishingstatus": furnishingstatus
}

input_df = pd.DataFrame([input_dict])

# Convert categorical values using one-hot encoding
input_encoded = pd.get_dummies(input_df, drop_first=True)

# Add missing columns



for col in feature_columns:
 if col not in input_encoded.columns:
   input_encoded[col] = 0

# Keep only training columns and same order
input_encoded = input_encoded[feature_columns]

# Predict
prediction = model.predict(input_encoded)[0]

st.success(f"Estimated House Price: ₹ {prediction:,.2f}")

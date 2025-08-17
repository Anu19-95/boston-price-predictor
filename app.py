
import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("boston_linear_model.pkl")

st.set_page_config(page_title="Boston House Price Predictor")
st.title("🏠 Boston House Price Predictor")
st.write("Enter the details below to predict the price of a house in Boston.")

# Input fields for all 13 features
CRIM = st.number_input("Per capita crime rate", 0.0, 100.0, step=0.1)
ZN = st.number_input("Proportion of residential land zoned for lots", 0.0, 100.0)
INDUS = st.number_input("Proportion of non-retail business acres", 0.0, 30.0)
CHAS = st.selectbox("Tract bounds Charles River?", [0, 1])
NOX = st.number_input("Nitric oxides concentration (parts per 10 million)", 0.0, 1.0, step=0.01)
RM = st.number_input("Average number of rooms per dwelling", 0.0, 10.0)
AGE = st.number_input("Proportion of owner-occupied units built before 1940", 0.0, 100.0)
DIS = st.number_input("Weighted distances to employment centers", 0.0, 15.0)
RAD = st.number_input("Index of accessibility to radial highways", 1, 24)
TAX = st.number_input("Full-value property-tax rate", 100, 800)
PTRATIO = st.number_input("Pupil-teacher ratio", 10.0, 30.0)
B = st.number_input("1000(Bk - 0.63)^2", 0.0, 400.0)
LSTAT = st.number_input("Percentage lower status of population", 0.0, 40.0)

if st.button("Predict Price"):
    features = np.array([[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]])
    prediction = model.predict(features)
    st.success(f"🏡 Estimated House Price: ${prediction[0]*1000:,.2f}")

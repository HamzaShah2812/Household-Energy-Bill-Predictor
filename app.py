import streamlit as st

import joblib
import numpy as np

# Load the saved model
model = joblib.load("models/svm_model.pkl")

st.set_page_config(page_title="Electricity Bill Predictor", page_icon="💡")

st.title("Electricity Bill Predictor 💡")
st.markdown("Enter your household details below to estimate your monthly electricity bill.")

# User input form
num_rooms = st.number_input("Number of Rooms", min_value=1)
num_people = st.number_input("Number of People", min_value=1)
housearea = st.number_input("House Area (sq. ft.)", min_value=100)
is_ac = st.selectbox("Has AC?", [0, 1])  # 0 = No, 1 = Yes
is_tv = st.selectbox("Has TV?", [0, 1])
is_flat = st.selectbox("Is a Flat?", [0, 1])
ave_monthly_income = st.number_input("Average Monthly Income (Rs)", min_value=1000)
num_children = st.number_input("Number of Children", min_value=0)
is_urban = st.selectbox("Is Urban Area?", [0, 1])

if st.button("Predict Bill"):
    input_data = np.array([[num_rooms, num_people, housearea, is_ac, is_tv, is_flat,
                            ave_monthly_income, num_children, is_urban]])
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Electricity Bill: Rs {prediction:.2f}")

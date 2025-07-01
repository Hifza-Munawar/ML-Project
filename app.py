import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# Load model and scaler
model = tf.keras.models.load_model("diabetes_nn_model.h5")
scaler = joblib.load("scaler.save")

st.title("🩺 Diabetes Prediction App")
st.write("Enter your health information below:")

# Input fields
preg = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose", 0, 200)
bp = st.number_input("Blood Pressure", 0, 140)
skin = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
age = st.number_input("Age", 10, 100)

if st.button("Predict"):
    user_input = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    scaled_input = scaler.transform(user_input)
    prediction = model.predict(scaled_input)

    if prediction[0][0] > 0.5:
        st.error("⚠️ Prediction: Diabetic")
    else:
        st.success("✅ Prediction: Not Diabetic")

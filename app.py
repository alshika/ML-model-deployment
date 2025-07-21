import streamlit as st
import pandas as pd
import joblib
import pickle5 as pickle

with open('model.pkl', 'rb') as f:
    data = f.read()

data = data.replace(b'numpy._core', b'numpy.core')

with open('model.pkl', 'wb') as f:
    f.write(data)

# Load trained model
model = data

st.title("🚢 Titanic Survival Prediction")

# Sidebar input
st.sidebar.header("Enter Passenger Info:")
pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
age = st.sidebar.slider("Age", 1, 80, 25)
sibsp = st.sidebar.slider("Siblings/Spouses Aboard", 0, 5, 0)
fare = st.sidebar.slider("Fare Paid", 0, 500, 50)

# Predict
if st.button("Predict Survival"):
    input_data = pd.DataFrame([[pclass, age, sibsp, fare]], columns=["Pclass", "Age", "SibSp", "Fare"])
    prediction = model.predict(input_data)
    result = "✅ Survived" if prediction[0] == 1 else "❌ Did Not Survive"
    st.success(f"Prediction: {result}")

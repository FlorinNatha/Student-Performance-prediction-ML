import streamlit as st
import requests

st.title("🎓 Student Performance Predictor")

studytime = st.slider("Study Time", 1, 4)
failures = st.slider("Failures", 0, 3)
absences = st.slider("Absences", 0, 50)

if st.button("Predict"):
    data = {
        "studytime": studytime,
        "failures": failures,
        "absences": absences
    }

    res = requests.post("http://localhost:8000/predict", json=data)
    result = res.json()
    prediction = result["prediction"]
    if prediction == 1:
        st.success("The student is predicted to PASS!")
    else:
        st.error("The student is predicted to FAIL!")
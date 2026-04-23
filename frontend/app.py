import streamlit as st
import requests

st.title("🎓 Student Performance Predictor")

st.header("Predict Final Grade")

# Numerical inputs
col1, col2 = st.columns(2)

with col1:
    studytime = st.slider("Weekly Study Time", 1, 4, 2, help="1 - <2h, 2 - 2-5h, 3 - 5-10h, 4 - >10h")
    failures = st.slider("Number of Past Failures", 0, 3, 0)
    absences = st.slider("Number of School Absences", 0, 93, 0)
    Medu = st.slider("Mother's Education", 0, 4, 2, help="0 - None, 1 - Primary, 2 - 5th-9th, 3 - Secondary, 4 - Higher")
    Fedu = st.slider("Father's Education", 0, 4, 2, help="Same as above")

with col2:
    famrel = st.slider("Family Relationships Quality", 1, 5, 4, help="1 - Very Bad, 5 - Excellent")
    goout = st.slider("Going Out with Friends", 1, 5, 3, help="1 - Very Low, 5 - Very High")
    Dalc = st.slider("Workday Alcohol Consumption", 1, 5, 1, help="1 - Very Low, 5 - Very High")
    Walc = st.slider("Weekend Alcohol Consumption", 1, 5, 1, help="1 - Very Low, 5 - Very High")
    health = st.slider("Current Health Status", 1, 5, 3, help="1 - Very Bad, 5 - Very Good")

if st.button("Predict"):
    data = {
        "studytime": studytime,
        "failures": failures,
        "absences": absences,
        "Medu": Medu,
        "Fedu": Fedu,
        "famrel": famrel,
        "goout": goout,
        "Dalc": Dalc,
        "Walc": Walc,
        "health": health
    }

    res = requests.post("http://localhost:8000/predict", json=data)
    result = res.json()
    prediction = result["prediction"]
    st.success(f"Predicted Final Score: {prediction}/20")
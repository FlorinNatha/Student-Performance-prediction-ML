import streamlit as st
import requests
import pickle
import pandas as pd

st.title("🎓 Student Performance Predictor")

# Load metrics and feature importances
try:
    metrics = pickle.load(open("../backend/metrics.pkl", "rb"))
    feature_importances = pickle.load(open("../backend/feature_importances.pkl", "rb"))
    columns = pickle.load(open("../backend/columns.pkl", "rb"))
except:
    metrics = {}
    feature_importances = None
    columns = []

tab1, tab2 = st.tabs(["Predict Final Grade", "Model Comparison"])

with tab1:
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

with tab2:
    st.header("Model Comparison Dashboard")

    if metrics:
        st.subheader("Model Performance Metrics")
        df_metrics = pd.DataFrame(metrics).T
        st.dataframe(df_metrics)

        st.subheader("MAE Comparison")
        st.bar_chart(df_metrics["MAE"])

        st.subheader("R² Comparison")
        st.bar_chart(df_metrics["R2"])
    else:
        st.write("Metrics not available.")

    if feature_importances is not None and columns:
        st.subheader("Feature Importances (Random Forest)")
        importances_df = pd.DataFrame({"Feature": columns, "Importance": feature_importances})
        importances_df = importances_df.sort_values("Importance", ascending=False)
        st.bar_chart(importances_df.set_index("Feature"))
    else:
        st.write("Feature importances not available.")
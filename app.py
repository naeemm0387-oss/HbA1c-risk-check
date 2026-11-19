from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------- 1. Load the saved model ----------
MODEL_PATH = Path(__file__).parent / "models" / "hba1c_risk_model.joblib"
THRESHOLD = 0.2   # chosen in notebook 07 before the final test

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ---------- 2. Page title ----------
st.title("HbA1c risk check")
st.write(
    "This tool estimates the chance that your HbA1c is 5.7% or higher "
    "(prediabetes range or above). It is for adults without a known diabetes diagnosis."
)

# ---------- 3. Questions ----------
st.header("About you")
age = st.number_input("Age (years)", min_value=18, max_value=80, value=55)
sex = st.radio("Sex", ["Female", "Male"], horizontal=True)

st.header("Body measurements")
height = st.number_input("Height (cm)", min_value=120.0, max_value=220.0, value=165.0)
weight = st.number_input("Weight (kg)", min_value=30.0, max_value=250.0, value=80.0)
waist = st.number_input("Waist (cm)", min_value=50.0, max_value=200.0, value=98.0)
hip = st.number_input("Hip (cm)", min_value=60.0, max_value=200.0, value=108.0)

st.header("Background")
race = st.selectbox("Race / ethnicity", [
    "Prefer not to say", "Mexican American", "Other Hispanic",
    "White", "Black", "Asian", "Other or Multiracial"])
education = st.selectbox("Highest education", [
    "Prefer not to say", "Less than 9th grade", "9th to 11th grade",
    "High school", "Some college", "College graduate"], index=3)

st.header("Health questions")
choices = ["Yes", "No", "Don't know"]
bp_told = st.radio("Has a doctor ever told you that you have high blood pressure?", choices, horizontal=True)
bp_med = st.radio("Do you take medicine for high blood pressure?", choices, horizontal=True)
chol_told = st.radio("Has a doctor ever told you that you have high cholesterol?", choices, index=1, horizontal=True)
chol_med = st.radio("Do you take medicine for high cholesterol?", choices, index=1, horizontal=True)

# ---------- 4. Helper: turn answers into numbers ----------
def yes_no(answer):
    if answer == "Yes":
        return 1
    if answer == "No":
        return 0
    return np.nan          # "Don't know" -> empty, the model fills it

def or_empty(answer):
    return np.nan if answer == "Prefer not to say" else answer

# ---------- 5. Calculate ----------
if st.button("Calculate my risk"):
    bmi = weight / (height / 100) ** 2

    person = pd.DataFrame([{
        "age_years": age,
        "is_female": 1 if sex == "Female" else 0,
        "bmi": bmi,
        "waist_cm": waist,
        "hip_cm": hip,
        "waist_to_height": waist / height,
        "waist_to_hip": waist / hip,
        "race_ethnicity_code": or_empty(race),
        "education_code": or_empty(education),
        "bp_told": yes_no(bp_told),
        "chol_told": yes_no(chol_told),
        "chol_med": yes_no(chol_med),
        "bp_med": 0 if bp_told == "No" else yes_no(bp_med),   # same rule as in notebook 07
    }])
    person = person.astype({"race_ethnicity_code": object, "education_code": object})

    risk = model.predict_proba(person)[0, 1]

    st.metric("Estimated risk", f"{risk:.0%}")
    st.write(f"Your BMI: {bmi:.1f}")
    if risk >= 0.4:
        st.error("High risk: please ask your doctor for an HbA1c blood test soon.")
    elif risk >= THRESHOLD:
        st.warning("Moderate risk: consider an HbA1c blood test at your next check-up.")
    else:
        st.success("Low risk: no strong signal. Keep a healthy lifestyle and regular check-ups.")
    st.write("For comparison: among adults without a diabetes diagnosis, about 3 in 10 had elevated HbA1c.")
    st.caption(
        "This is not a diagnosis. Model trained on NHANES 2021–2023 (US adults). "
        "'Prefer not to say' and 'Don't know' are filled with the most common answer."
    )

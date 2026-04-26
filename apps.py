import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------
# Load Trained Classification Model
# ---------------------------------------
rf = joblib.load("classification_model.pkl")

st.set_page_config(page_title="EduInsight", layout="centered")

st.title("🎓 EduInsight")
st.subheader("AI Powered Student Performance & Personalized Tutoring System")

st.write("Enter student details below:")

# ---------------------------------------
# USER INPUTS
# ---------------------------------------

student_id = st.number_input("Student ID", min_value=1)

age = st.number_input("Age", min_value=5, max_value=25)

gender = st.selectbox("Gender", ["Male", "Female"])

student_class = st.selectbox("Class", [8, 9, 10, 11, 12])

study_hours = st.number_input("Study Hours Per Day", min_value=0.0, max_value=15.0)

attendance = st.number_input("Attendance Percentage", min_value=0.0, max_value=100.0)

parental_education = st.selectbox(
    "Parental Education",
    ["High School", "Bachelor", "Master"]
)

internet_access = st.selectbox("Internet Access", ["Yes", "No"])

extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])

math_score = st.number_input("Math Score", min_value=0.0, max_value=100.0)

science_score = st.number_input("Science Score", min_value=0.0, max_value=100.0)

english_score = st.number_input("English Score", min_value=0.0, max_value=100.0)

previous_year_score = st.number_input(
    "Previous Year Score",
    min_value=0.0,
    max_value=100.0
)

# ---------------------------------------
# ENCODING (MATCH TRAINING FORMAT)
# ---------------------------------------

gender_encoded = 1 if gender == "Male" else 0
internet_encoded = 1 if internet_access == "Yes" else 0
extra_encoded = 1 if extracurricular == "Yes" else 0

education_map = {
    "High School": 0,
    "Bachelor": 1,
    "Master": 2
}
parent_edu_encoded = education_map[parental_education]

# ---------------------------------------
# CREATE INPUT DATAFRAME
# ---------------------------------------

input_data = pd.DataFrame({
    "Student_ID": [student_id],
    "Age": [age],
    "Gender": [gender_encoded],
    "Class": [student_class],
    "Study_Hours_Per_Day": [study_hours],
    "Attendance_Percentage": [attendance],
    "Parental_Education": [parent_edu_encoded],
    "Internet_Access": [internet_encoded],
    "Extracurricular_Activities": [extra_encoded],
    "Math_Score": [math_score],
    "Science_Score": [science_score],
    "English_Score": [english_score],
    "Previous_Year_Score": [previous_year_score]
})

input_data = input_data[rf.feature_names_in_]

# ---------------------------------------
# PREDICTION + LLM STYLE TUTORING
# ---------------------------------------

if st.button("Analyze Student Performance"):

    prediction = rf.predict(input_data)[0]

    # Convert numeric prediction to readable label
    if prediction == 1:
        result = "Pass"
        message = "The student is likely to PASS."
    else:
        result = "Fail"
        message = "The student is at risk of FAILING."

    st.success(f"🎯 Prediction: {result}")
    st.write(message)

    # Confidence Score
    if hasattr(rf, "predict_proba"):
        probability = rf.predict_proba(input_data)[0]
        confidence = max(probability) * 100
        st.info(f"Model Confidence: {confidence:.2f}%")

    # ---------------------------------------
    # Personalized Tutoring (LLM-Style Logic)
    # ---------------------------------------

    st.subheader("📘 Personalized Tutoring Advice")

    advice = ""

    if study_hours < 2:
        advice += "• Increase daily study time to at least 3–4 hours.\n"

    if attendance < 75:
        advice += "• Improve attendance. Regular classes strengthen conceptual clarity.\n"

    if previous_year_score < 50:
        advice += "• Revise fundamental concepts from previous year topics.\n"

    if math_score < 40:
        advice += "• Focus more on Math practice with problem-solving exercises.\n"

    if science_score < 40:
        advice += "• Strengthen Science understanding through diagrams and experiments.\n"

    if english_score < 40:
        advice += "• Improve English by daily reading and vocabulary practice.\n"

    if internet_access == "No":
        advice += "• Try accessing digital learning resources through school or library.\n"

    if extracurricular == "Yes":
        advice += "• Balance extracurricular activities with structured study time.\n"

    if advice == "":
        advice = "• The student is performing well. Maintain consistency and aim for excellence."

    st.write(advice)

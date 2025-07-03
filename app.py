# import streamlit for web app
import streamlit as st
# import pandas for data handling
import pandas as pd
# import joblib to load saved model
import joblib
# import label encoder
from sklearn.preprocessing import LabelEncoder

# load the trained model from file
model = joblib.load('model/heart_disease_model.pkl')

# set the app title
st.title("Heart Disease Risk Predictor")

# get age input from user
age = st.number_input("Age", min_value=28, max_value=77, value=53)

# get sex input from user
sex = st.selectbox("Sex", ["M", "F"])

# get chest pain type input
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])

# get resting blood pressure
resting_bp = st.number_input("Resting Blood Pressure", min_value=0, max_value=200, value=132)

# get cholesterol level
cholesterol = st.number_input("Cholesterol", min_value=0, max_value=600, value=200)

# get fasting blood sugar status
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])

# get resting ECG results
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])

# get maximum heart rate achieved
max_hr = st.number_input("Max Heart Rate", min_value=60, max_value=202, value=150)

# get exercise-induced angina status
exercise_angina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])

# get oldpeak value (ST depression)
oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=6.2, value=1.0)

# get ST slope type
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# function to convert categorical inputs to numbers
def encode_input(val, column):
    # mapping dictionaries for encoding
    mapping = {
        'Sex': {"M": 1, "F": 0},
        'ChestPainType': {"ATA": 0, "NAP": 1, "TA": 2, "ASY": 3},
        'RestingECG': {"Normal": 0, "ST": 1, "LVH": 2},
        'ExerciseAngina': {"Yes": 1, "No": 0},
        'ST_Slope': {"Up": 2, "Flat": 1, "Down": 0}
    }
    # return encoded value
    return mapping[column][val]

# create a dataframe with user inputs for prediction
input_data = pd.DataFrame({
    'Age': [age],  # wrap in list to match dataframe shape
    'Sex': [encode_input(sex, 'Sex')],
    'ChestPainType': [encode_input(chest_pain, 'ChestPainType')],
    'RestingBP': [resting_bp],
    'Cholesterol': [cholesterol],
    'FastingBS': [1 if fasting_bs == "Yes" else 0],  # binary encoding
    'RestingECG': [encode_input(resting_ecg, 'RestingECG')],
    'MaxHR': [max_hr],
    'ExerciseAngina': [encode_input(exercise_angina, 'ExerciseAngina')],
    'Oldpeak': [oldpeak],
    'ST_Slope': [encode_input(st_slope, 'ST_Slope')]
})

# show prediction result when user clicks the button
if st.button("Predict Heart Disease Risk"):
    # make prediction using the model
    prediction = model.predict(input_data)[0]
    # interpret prediction result
    risk = "High Risk" if prediction == 1 else "Low Risk"
    # display the result on screen
    st.write(f"### Prediction: {risk}")

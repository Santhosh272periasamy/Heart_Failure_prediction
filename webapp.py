import streamlit as st 
import pandas as pd
import joblib


st.title('Check Your Heart Health')

df = pd.read_csv('heart.csv')

# Age                   40
# Sex                    M
# ChestPainType        ATA
# RestingBP            140
# Cholesterol          289
# FastingBS              0
# RestingECG        Normal
# MaxHR                172
# ExerciseAngina         0
# Oldpeak              0.0
# ST_Slope              Up
# HeartDisease           0


Age = st.number_input("Age")
Sex = st.selectbox("Sex",pd.unique(df['Sex']))
ChestPainType = st.selectbox("ChestPainType",pd.unique(df['ChestPainType']))
RestingBP = st.number_input("RestingBP")
Cholesterol = st.number_input("Cholesterol")
FastingBS = st.selectbox("FastingBS",pd.unique(df['FastingBS']))
RestingECG =st.selectbox("RestingECG",pd.unique(df['RestingECG']))
MaxHR = st.number_input("MaxHR")
ExerciseAngina = st.selectbox("ExerciseAngina",pd.unique(df['ExerciseAngina']))
Oldpeak = st.number_input("Oldpeak")
ST_Slope = st.selectbox("ST_Slope",pd.unique(df['ST_Slope']))

inputs = {
    'Age' : Age , 
    'Sex' :Sex,
    'ChestPainType':ChestPainType,
    'RestingBP' :RestingBP,
    'Cholesterol' :Cholesterol,
    'FastingBS' :FastingBS,
    'RestingECG' :RestingECG,
    'MaxHR' :MaxHR,
    'ExerciseAngina' :ExerciseAngina,
    'Oldpeak': Oldpeak,
    'ST_Slope' : ST_Slope
}

if st.button('Predict'):
    model = joblib.load('Final_predication_PCA_LR.pkl')

    x_input = pd.DataFrame(inputs , index=[0])
    prediction  = model.predict(x_input)
    st.write(prediction)
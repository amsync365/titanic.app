import streamlit as st
from utils import PREPROCESSOR, columns

import numpy as np
import pandas as pd
import joblib

model = joblib.load('pipe.titanic')
st.title('Will you survive if you were among Titanic passengers or not :ship:')
# PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
passengerid = st.text_input("Input Passenger ID", '8585')
pclass = st.selectbox("Choose Class", [1,2,3])
name  = st.text_input("Input Passenger Name:", 'Am')
sex = st.selectbox('Choose Sex:',['male', 'female'])
age = st.slider('Choose Age:',0,100)
sibsp = st.slider("Choose siblings",0,10)
parch = st.slider("Choose parch",0,10)
ticket = st.text_input("Input Ticket Number", "8585") 
fare = st.number_input("Input Fare Price", 0,1000)
cabin = st.text_input("Input Cabin", "C52") 
embarked = st.select_slider("Did they Embark?", ['S','C','Q'])


def predict():
    row=np.array([passengerid, pclass, name, sex, float(age), sibsp, parch, ticket, fare, cabin, embarked])
    test = pd.DataFrame([row], columns=columns)
    prediction = model.predict(test)
    if prediction[0] == 1: 
        st.success("Passenger Survived 👍 ")
    else: 
        st.error("Passenger did not Survive 👎 ") 

trigger = st.button('Predict', on_click=predict)


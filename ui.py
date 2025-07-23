import streamlit as st 
import requests

st.title(" score Predictor")

study=st.slider("Study Time",0,10)
atd=st.slider("attendence Days",0,80)
gen=st.selectbox("Gender",["Male","Female"])

gender=1 if(gen=="Male") else 0
if(st.button("Predict the score")):
    data={
        "study_time":study,
        "attendence":atd,
        "gender_Male":gender
    }
    res=requests.post("http://10.10.1.235:8501/predict",json=data)
    result=res.json()
    st.write("The Predicted Score is ", result["Predicted_score"])
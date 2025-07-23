from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

df=joblib.load("model/score_train_model.pkl")


class stu_data(BaseException):
    study_time:float
    attendence:float
    gender_Male:int
app=FastAPI()
@app.get("/")
def root_data():
    return{"message":"HI welcome to magic show"}

@app.post("/predict")
def scr_prd(data:stu_data):
    inp_data=np.array([[data.study_time,data.attendence,data.gender_Male]])
    prd=df.predict(inp_data)
    return {"Predicted_score":int(prd[0])}

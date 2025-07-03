# import fastapi core
from fastapi import FastAPI
# for validating request data
from pydantic import BaseModel
# for loading saved ML model
import joblib
# for creating dataframe
import pandas as pd
# to enable frontend-backend communication
from fastapi.middleware.cors import CORSMiddleware

# create fastapi app
app = FastAPI()

# enable CORS so frontend can call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # allow requests from origin
    allow_credentials=True,
    allow_methods=["*"],  # allow all HTTP methods
    allow_headers=["*"],  # allow all headers
)

# load trained ML model
model = joblib.load("model.pkl")

# define expected input structure
class InputData(BaseModel):
    Age: int
    Sex: int
    ChestPainType: int
    RestingBP: int
    Cholesterol: int
    FastingBS: int
    RestingECG: int
    MaxHR: int
    ExerciseAngina: int
    Oldpeak: float
    ST_Slope: int

# define prediction route
@app.post("/predict")
def predict(data: InputData):
    # convert input to dataframe
    input_df = pd.DataFrame([data.dict()])
    # make prediction using the model
    prediction = model.predict(input_df)[0]
    # return risk level
    return {"prediction": "High Risk" if prediction == 1 else "Low Risk"}

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Load model
model_data = joblib.load("loan_svm_model.pkl")
model = model_data["model"]
columns = model_data["columns"]

# Input schema
class LoanApplication(BaseModel):
    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: str

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Loan Prediction API is running"}

@app.post("/predict")
def predict_loan(data: LoanApplication):
    # Convert input to DataFrame
    input_dict = data.dict()
    input_df = pd.DataFrame([input_dict])

    # Encode categorical features same as training
    for col in input_df.select_dtypes(include="object").columns:
        input_df[col] = input_df[col].astype("category").cat.codes

    # Reorder columns
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # Prediction
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    return {
        "prediction": "Eligible" if prediction == 1 else "Not Eligible",
        "probability": {
            "Not Eligible": (round(float(proba[0]), 3)),
            "Eligible": (round(float(proba[1]), 3))
        }
    }

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
    CIBIL_Score: float 

app = FastAPI()

def find_max_eligible_amount(model, input_df, step=5):
    original_amount = input_df["LoanAmount"].values[0]

    while original_amount > 0:
        input_df["LoanAmount"] = original_amount
        if model.predict(input_df)[0] == 1:
            return original_amount
        original_amount -= step

    return 0


@app.get("/")
def root():
    return {"message": "Loan Prediction API is running"}

@app.post("/predict")
def predict_loan(data: LoanApplication):

    input_df = pd.DataFrame([data.dict()])

    for col in input_df.select_dtypes(include="object").columns:
        input_df[col] = input_df[col].astype("category").cat.codes

    input_df = input_df.reindex(columns=columns, fill_value=0)

    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    response = {
        "prediction": "Eligible" if prediction == 1 else "Not Eligible",
        "probability": {
            "Not Eligible": round(float(proba[0]), 3),
            "Eligible": round(float(proba[1]), 3)
        }
    }

    # 🔹 APPLY ITERATIVE LOAN ADJUSTMENT
    if prediction == 0:
        eligible_amount = find_max_eligible_amount(model, input_df.copy())
        response["max_eligible_loan_amount"] = eligible_amount

    return response


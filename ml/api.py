import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from data_preprocessing import load_and_clean_data
from transaction_categorizer import TransactionCategorizer
from feature_engineering import engineer_advanced_features
from stability_report import calculate_stability_score, generate_recommendation

app = FastAPI(title="CreditBridge ML API")

# --- Data Models ---
class Transaction(BaseModel):
    date: str
    desc: str
    withdrawal: float
    deposit: float
    balance: float

class CustomerRequest(BaseModel):
    name: str
    transactions: List[Transaction]

# --- Global State ---
# Load model once on startup to avoid latency
try:
    STRESS_MODEL = joblib.load('ml/stress_model.joblib')
except:
    STRESS_MODEL = None

# Pre-defined categories for the categorizer
EXAMPLES = {
    'Housing': ['UPI RENT PAYMENT', 'HOUSE RENT', 'MORTGAGE PAYMENT', 'SOCIETY MAINTENANCE'],
    'Income': ['NEFT SALARY CREDIT', 'DIRECT DEPOSIT SALARY', 'INTEREST CREDIT', 'BONUS'],
    'Food': ['SWIGGY ORDER', 'ZOMATO DELIVERY', 'RESTAURANT', 'STARBUCKS', 'GROCERIES'],
    'Cash': ['ATM CASH WITHDRAWAL', 'CASH DEPOSIT ATM', 'CASH WITHDRAWAL'],
    'Transport': ['UBER TRIP', 'OLA CAB', 'METRO RECHARGE', 'FUEL STATION', 'PETROL'],
    'Shopping': ['AMAZON INDIA', 'FLIPKART', 'MYNTRA', 'CLOTHING STORE'],
    'Utility': ['ELECTRICITY BILL', 'WATER BILL', 'INTERNET RECHARGE', 'MOBILE BILL']
}

@app.get("/")
async def root():
    return {"message": "CreditBridge ML API is running. Use /predict for inference."}

@app.post("/predict")
async def predict(request: CustomerRequest):
    try:
        # 1. Convert request data to DataFrame
        tx_data = [tx.dict() for tx in request.transactions]
        df = pd.DataFrame(tx_data)
        df.columns = ['transaction_date', 'transaction_description', 'withdrawals', 'deposits', 'balance']
        df['account_number'] = 'api_user'
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        df = df.sort_values('transaction_date')

        # 2. Categorization
        categorizer = TransactionCategorizer()
        categorizer.fit(EXAMPLES)
        df['category'] = df['transaction_description'].astype(str).apply(categorizer.predict)

        # 3. Feature Engineering
        features_df = engineer_advanced_features(df)

        # 4. Stress Prediction
        if STRESS_MODEL:
            X = features_df.drop(['account_number'], axis=1)
            stress_prob = STRESS_MODEL.predict_proba(X)[0][1]
            stress_risk = "High" if stress_prob > 0.5 else "Low"
        else:
            stress_risk = "Unknown"

        # 5. Stability Score & Recommendation
        score_results = calculate_stability_score(features_df)
        final_score = float(score_results.loc[0, 'score'])
        pos_signals = score_results.loc[0, 'positive']
        risk_signals = score_results.loc[0, 'risk']
        recommendation = generate_recommendation(final_score, risk_signals)

        return {
            "customer_name": request.name,
            "stability_score": final_score,
            "stress_risk": stress_risk,
            "recommendation": recommendation,
            "positive_signals": pos_signals,
            "risk_signals": risk_signals,
            "categories": df[['transaction_description', 'category']].to_dict(orient='records')
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

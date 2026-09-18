import pandas as pd
import numpy as np
import joblib
import os
from data_preprocessing import load_and_clean_data, calculate_stress_target
from transaction_categorizer import TransactionCategorizer
from feature_engineering import engineer_advanced_features
from stability_report import calculate_stability_score, generate_recommendation

def run_inference(customer_transactions):
    """
    Runs a full pipeline for a single customer.
    customer_transactions: List of dicts [{'date': '...', 'desc': '...', 'withdrawal': 0, 'deposit': 0, 'balance': 0}]
    """
    print("\n--- Processing Customer Profile ---")

    # 1. Convert to DataFrame
    df = pd.DataFrame(customer_transactions)
    df.columns = ['transaction_date', 'transaction_description', 'withdrawals', 'deposits', 'balance']
    df['account_number'] = 'test_user_123'
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    df = df.sort_values('transaction_date')

    # 2. Categorize Transactions
    categorizer = TransactionCategorizer()
    # Seed labels (matching Phase 2)
    examples = {
        'Housing': ['UPI RENT PAYMENT', 'HOUSE RENT', 'MORTGAGE PAYMENT', 'SOCIETY MAINTENANCE'],
        'Income': ['NEFT SALARY CREDIT', 'DIRECT DEPOSIT SALARY', 'INTEREST CREDIT', 'BONUS'],
        'Food': ['SWIGGY ORDER', 'ZOMATO DELIVERY', 'RESTAURANT', 'STARBUCKS', 'GROCERIES'],
        'Cash': ['ATM CASH WITHDRAWAL', 'CASH DEPOSIT ATM', 'CASH WITHDRAWAL'],
        'Transport': ['UBER TRIP', 'OLA CAB', 'METRO RECHARGE', 'FUEL STATION', 'PETROL'],
        'Shopping': ['AMAZON INDIA', 'FLIPKART', 'MYNTRA', 'CLOTHING STORE'],
        'Utility': ['ELECTRICITY BILL', 'WATER BILL', 'INTERNET RECHARGE', 'MOBILE BILL']
    }
    categorizer.fit(examples)
    df['category'] = df['transaction_description'].astype(str).apply(categorizer.predict)

    # 3. Feature Engineering
    features_df = engineer_advanced_features(df)

    # 4. Predict Stress (using trained model)
    try:
        model = joblib.load('ml/stress_model.joblib')
        # Match features to model training columns
        # The model expects: ['burn_rate', 'essential_ratio', 'discretionary_ratio',
        # 'withdrawal_volatility', 'savings_buffer', 'income_consistency', 'balance_trend', 'final_balance']
        X = features_df.drop(['account_number'], axis=1)
        stress_prob = model.predict_proba(X)[0][1]
        stress_risk = "High" if stress_prob > 0.5 else "Low/Medium"
    except Exception as e:
        stress_risk = "Unknown (Model not found)"
        print(f"Model Error: {e}")

    # 5. Stability Score & Recommendation
    score_results = calculate_stability_score(features_df)
    final_score = score_results.loc[0, 'score']
    pos_signals = score_results.loc[0, 'positive']
    risk_signals = score_results.loc[0, 'risk']
    recommendation = generate_recommendation(final_score, risk_signals)

    # Print final "Customer Health Card"
    print("\n==========================================")
    print("       CREDITBRIDGE CUSTOMER PROFILE      ")
    print("==========================================")
    print(f"Overall Stability Score: {final_score}/100")
    print(f"Financial Stress Risk:   {stress_risk}")
    print(f"Recommendation:          {recommendation}")
    print("------------------------------------------")
    print(f"✅ Positive Signals: {', '.join(pos_signals) if pos_signals else 'None'}")
    print(f"⚠️ Risk Signals:      {', '.join(risk_signals) if risk_signals else 'None'}")
    print("==========================================\n")

if __name__ == "__main__":
    # TEST CASE 1: Stable User (Regular income, low spend, high buffer)
    stable_user = [
        {'date': '2023-01-01', 'desc': 'NEFT SALARY CREDIT', 'withdrawal': 0, 'deposit': 5000, 'balance': 5000},
        {'date': '2023-01-05', 'desc': 'UPI RENT PAYMENT', 'withdrawal': 1000, 'deposit': 0, 'balance': 4000},
        {'date': '2023-01-10', 'desc': 'SWIGGY ORDER', 'withdrawal': 200, 'deposit': 0, 'balance': 3800},
        {'date': '2023-02-01', 'desc': 'NEFT SALARY CREDIT', 'withdrawal': 0, 'deposit': 5000, 'balance': 8800},
        {'date': '2023-02-05', 'desc': 'UPI RENT PAYMENT', 'withdrawal': 1000, 'deposit': 0, 'balance': 7800},
    ]

    # TEST CASE 2: Stressed User (Irregular income, high burn rate, low balance)
    stressed_user = [
        {'date': '2023-01-01', 'desc': 'CASH DEPOSIT', 'withdrawal': 0, 'deposit': 1000, 'balance': 1000},
        {'date': '2023-01-02', 'desc': 'ZOMATO DELIVERY', 'withdrawal': 500, 'deposit': 0, 'balance': 500},
        {'date': '2023-01-03', 'desc': 'UBER TRIP', 'withdrawal': 600, 'deposit': 0, 'balance': -100},
        {'date': '2023-01-10', 'desc': 'SMALL PAYMENT', 'withdrawal': 0, 'deposit': 200, 'balance': 100},
        {'date': '2023-01-15', 'desc': 'HOUSE RENT', 'withdrawal': 1000, 'deposit': 0, 'balance': -900},
    ]

    print("Testing Case 1: Stable User...")
    run_inference(stable_user)

    print("\nTesting Case 2: Stressed User...")
    run_inference(stressed_user)

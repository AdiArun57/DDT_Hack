import pandas as pd
import numpy as np
import joblib
import os
from data_preprocessing import load_and_clean_data, calculate_stress_target
from transaction_categorizer import TransactionCategorizer
from feature_engineering import engineer_advanced_features
from stability_report import calculate_stability_score, generate_recommendation

def run_inference(customer_name, customer_transactions):
    """
    Runs a full pipeline for a single customer.
    """
    print(f"\n--- Processing Profile: {customer_name} ---")

    # 1. Convert to DataFrame
    df = pd.DataFrame(customer_transactions)
    df.columns = ['transaction_date', 'transaction_description', 'withdrawals', 'deposits', 'balance']
    df['account_number'] = 'test_user_123'
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    df = df.sort_values('transaction_date')

    # 2. Categorize Transactions
    categorizer = TransactionCategorizer()
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
        X = features_df.drop(['account_number'], axis=1)
        stress_prob = model.predict_proba(X)[0][1]
        # STRICT BINARY RISK: Only High or Low
        stress_risk = "High" if stress_prob > 0.5 else "Low"
    except Exception as e:
        stress_risk = "Unknown"
        print(f"Model Error: {e}")

    # 5. Stability Score & Recommendation
    score_results = calculate_stability_score(features_df)
    final_score = score_results.loc[0, 'score']
    pos_signals = score_results.loc[0, 'positive']
    risk_signals = score_results.loc[0, 'risk']
    recommendation = generate_recommendation(final_score, risk_signals)

    # Print final "Customer Health Card"
    print("\n==========================================")
    print(f"       PROFILE: {customer_name}")
    print("==========================================")
    print(f"Overall Stability Score: {final_score}/100")
    print(f"Financial Stress Risk:   {stress_risk}")
    print(f"Recommendation:          {recommendation}")
    print("------------------------------------------")
    print(f"✅ Positive Signals: {', '.join(pos_signals) if pos_signals else 'None'}")
    print(f"⚠️ Risk Signals:      {', '.join(risk_signals) if risk_signals else 'None'}")
    print("==========================================\n")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("            CREDITBRIDGE AI EVALUATION SYSTEM            ")
    print("="*60)
    print("\n📌 DEFINITIONS:")
    print("1. Financial Stress Risk (ML Model):")
    print("   A binary prediction (High/Low) of whether a customer is likely")
    print("   to reach a critically low balance within 30 days based on")
    print("   historical behavioral patterns.")
    print("\n2. Cash-Flow Stability Score (Heuristic):")
    print("   A weighted 0-100 score based on:")
    print("   - Savings Buffer (30%): Balance vs. monthly burn rate")
    print("   - Income Consistency (20%): Regularity of deposits")
    print("   - Burn Rate (20%): Spending vs. income ratio")
    print("   - Withdrawal Volatility (15%): Stability of spending")
    print("   - Balance Trend (15%): Direction of account balance")
    print("="*60 + "\n")

    test_cases = {
        "The Ideal Saver": [
            {'date': '2023-01-01', 'desc': 'NEFT SALARY CREDIT', 'withdrawal': 0, 'deposit': 5000, 'balance': 5000},
            {'date': '2023-01-05', 'desc': 'UPI RENT PAYMENT', 'withdrawal': 500, 'deposit': 0, 'balance': 4500},
            {'date': '2023-02-01', 'desc': 'NEFT SALARY CREDIT', 'withdrawal': 0, 'deposit': 5000, 'balance': 9500},
            {'date': '2023-02-05', 'desc': 'UPI RENT PAYMENT', 'withdrawal': 500, 'deposit': 0, 'balance': 9000},
        ],
        "The High-Burn Professional": [
            {'date': '2023-01-01', 'desc': 'NEFT SALARY CREDIT', 'withdrawal': 0, 'deposit': 10000, 'balance': 10000},
            {'date': '2023-01-02', 'desc': 'APPLE STORE', 'withdrawal': 8000, 'deposit': 0, 'balance': 2000},
            {'date': '2023-01-05', 'desc': 'ZOMATO DELIVERY', 'withdrawal': 1000, 'deposit': 0, 'balance': 1000},
            {'date': '2023-01-10', 'desc': 'UBER TRIP', 'withdrawal': 1500, 'deposit': 0, 'balance': -500},
        ],
        "The Irregular Freelancer": [
            {'date': '2023-01-01', 'desc': 'PROJECT PAYMENT', 'withdrawal': 0, 'deposit': 2000, 'balance': 2000},
            {'date': '2023-01-05', 'desc': 'HOUSE RENT', 'withdrawal': 1500, 'deposit': 0, 'balance': 500},
            {'date': '2023-02-15', 'desc': 'PROJECT PAYMENT', 'withdrawal': 0, 'deposit': 1000, 'balance': 1500},
            {'date': '2023-02-20', 'desc': 'ELECTRICITY BILL', 'withdrawal': 1200, 'deposit': 0, 'balance': 300},
        ],
        "The Debt Cycle": [
            {'date': '2023-01-01', 'desc': 'CASH DEPOSIT', 'withdrawal': 0, 'deposit': 500, 'balance': 500},
            {'date': '2023-01-02', 'desc': 'ATM CASH WITHDRAWAL', 'withdrawal': 600, 'deposit': 0, 'balance': -100},
            {'date': '2023-01-05', 'desc': 'CASH DEPOSIT', 'withdrawal': 0, 'deposit': 200, 'balance': 100},
            {'date': '2023-01-06', 'desc': 'ZOMATO DELIVERY', 'withdrawal': 300, 'deposit': 0, 'balance': -200},
        ],
        "The New Graduate": [
            {'date': '2023-01-01', 'desc': 'PARENTAL SUPPORT', 'withdrawal': 0, 'deposit': 1000, 'balance': 1000},
            {'date': '2023-01-05', 'desc': 'METRO RECHARGE', 'withdrawal': 200, 'deposit': 0, 'balance': 800},
            {'date': '2023-01-10', 'desc': 'STARBUCKS', 'withdrawal': 300, 'deposit': 0, 'balance': 500},
            {'date': '2023-01-20', 'desc': 'GROCERIES', 'withdrawal': 400, 'deposit': 0, 'balance': 100},
        ],
        "The Minimalist": [
            {'date': '2023-01-01', 'desc': 'INTEREST CREDIT', 'withdrawal': 0, 'deposit': 100, 'balance': 10000},
            {'date': '2023-01-10', 'desc': 'ELECTRICITY BILL', 'withdrawal': 100, 'deposit': 0, 'balance': 9900},
            {'date': '2023-02-01', 'desc': 'INTEREST CREDIT', 'withdrawal': 0, 'deposit': 100, 'balance': 10000},
            {'date': '2023-02-10', 'desc': 'ELECTRICITY BILL', 'withdrawal': 100, 'deposit': 0, 'balance': 9900},
        ],
        "The Shopping Spree": [
            {'date': '2023-01-01', 'desc': 'SALARY', 'withdrawal': 0, 'deposit': 5000, 'balance': 5000},
            {'date': '2023-01-02', 'desc': 'AMAZON INDIA', 'withdrawal': 2000, 'deposit': 0, 'balance': 3000},
            {'date': '2023-01-03', 'desc': 'FLIPKART', 'withdrawal': 2000, 'deposit': 0, 'balance': 1000},
            {'date': '2023-01-04', 'desc': 'MYNTRA', 'withdrawal': 1500, 'deposit': 0, 'balance': -500},
        ],
        "The Budget Master": [
            {'date': '2023-01-01', 'desc': 'SALARY', 'withdrawal': 0, 'deposit': 4000, 'balance': 4000},
            {'date': '2023-01-02', 'desc': 'HOUSE RENT', 'withdrawal': 1000, 'deposit': 0, 'balance': 3000},
            {'date': '2023-01-05', 'desc': 'METRO RECHARGE', 'withdrawal': 100, 'deposit': 0, 'balance': 2900},
            {'date': '2023-01-10', 'desc': 'GROCERIES', 'withdrawal': 500, 'deposit': 0, 'balance': 2400},
        ],
        "The Erratic Spender": [
            {'date': '2023-01-01', 'desc': 'SALARY', 'withdrawal': 0, 'deposit': 5000, 'balance': 5000},
            {'date': '2023-01-02', 'desc': 'CASH WITHDRAWAL', 'withdrawal': 4000, 'deposit': 0, 'balance': 1000},
            {'date': '2023-01-05', 'desc': 'ZOMATO', 'withdrawal': 100, 'deposit': 0, 'balance': 900},
            {'date': '2023-01-10', 'desc': 'CASH DEPOSIT', 'withdrawal': 0, 'deposit': 2000, 'balance': 2900},
        ],
        "The Stable Low-Income": [
            {'date': '2023-01-01', 'desc': 'PART TIME PAY', 'withdrawal': 0, 'deposit': 1500, 'balance': 1500},
            {'date': '2023-01-05', 'desc': 'BUS PASS', 'withdrawal': 100, 'deposit': 0, 'balance': 1400},
            {'date': '2023-02-01', 'desc': 'PART TIME PAY', 'withdrawal': 0, 'deposit': 1500, 'balance': 2900},
            {'date': '2023-02-05', 'desc': 'BUS PASS', 'withdrawal': 100, 'deposit': 0, 'balance': 2800},
        ],
        "The Sudden Crash": [
            {'date': '2023-01-01', 'desc': 'SALARY', 'withdrawal': 0, 'deposit': 5000, 'balance': 5000},
            {'date': '2023-01-05', 'desc': 'RENT', 'withdrawal': 1000, 'deposit': 0, 'balance': 4000},
            {'date': '2023-02-01', 'desc': 'MEDICAL EMERGENCY', 'withdrawal': 6000, 'deposit': 0, 'balance': -2000},
            {'date': '2023-02-05', 'desc': 'ZOMATO', 'withdrawal': 200, 'deposit': 0, 'balance': -2200},
        ],
        "The Side-Hustler": [
            {'date': '2023-01-01', 'desc': 'SALARY', 'withdrawal': 0, 'deposit': 3000, 'balance': 3000},
            {'date': '2023-01-15', 'desc': 'FREELANCE WORK', 'withdrawal': 0, 'deposit': 1000, 'balance': 4000},
            {'date': '2023-01-20', 'desc': 'SHOPPING', 'withdrawal': 500, 'deposit': 0, 'balance': 3500},
            {'date': '2023-02-01', 'desc': 'SALARY', 'withdrawal': 0, 'deposit': 3000, 'balance': 6500},
        ],
        "The Ghost Account": [
            {'date': '2023-01-01', 'desc': 'INITIAL DEPOSIT', 'withdrawal': 0, 'deposit': 100, 'balance': 100},
            {'date': '2023-01-02', 'desc': 'BANK FEE', 'withdrawal': 50, 'deposit': 0, 'balance': 50},
            {'date': '2023-01-10', 'desc': 'BANK FEE', 'withdrawal': 50, 'deposit': 0, 'balance': 0},
            {'date': '2023-01-20', 'desc': 'BANK FEE', 'withdrawal': 50, 'deposit': 0, 'balance': -50},
        ],
        "The Recovery Phase": [
            {'date': '2023-01-01', 'desc': 'OVERDRAFT', 'withdrawal': 0, 'deposit': 0, 'balance': -1000},
            {'date': '2023-01-10', 'desc': 'SALARY', 'withdrawal': 0, 'deposit': 3000, 'balance': 2000},
            {'date': '2023-01-15', 'desc': 'RENT', 'withdrawal': 800, 'deposit': 0, 'balance': 1200},
            {'date': '2023-02-01', 'desc': 'SALARY', 'withdrawal': 0, 'deposit': 3000, 'balance': 4200},
        ],
    }

    for name, data in test_cases.items():
        run_inference(name, data)

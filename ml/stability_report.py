import pandas as pd
import numpy as np
import joblib
import os

def calculate_stability_score(features):
    """
    Calculates a 0-100 stability score based on weighted financial features.
    """
    # Normalized weights (must sum to 1.0)
    WEIGHTS = {
        'savings_buffer': 0.30,
        'income_consistency': 0.20,
        'burn_rate': 0.20,
        'withdrawal_volatility': 0.15,
        'balance_trend': 0.15
    }

    scores = []
    for idx, row in features.iterrows():
        # Helper to handle NaN/Inf values
        def clean_val(val, default=0.0):
            return val if np.isfinite(val) else default

        sb = clean_val(row['savings_buffer'], 0.0)
        ic = clean_val(row['income_consistency'], 1.0) # High volatility by default
        br = clean_val(row['burn_rate'], 2.0)           # High burn by default
        wv = clean_val(row['withdrawal_volatility'], 1.0) # High volatility by default
        bt = clean_val(row['balance_trend'], -1000.0)   # Negative trend by default

        # 1. Savings Buffer Score (0-100)
        sb_score = np.clip((sb / 3.0) * 100, 0, 100)

        # 2. Income Consistency Score (0-100)
        ic_score = np.clip((1 - ic) * 100, 0, 100)

        # 3. Burn Rate Score (0-100)
        br_score = np.clip((2.0 - br) * 50, 0, 100)

        # 4. Withdrawal Volatility Score (0-100)
        wv_score = np.clip((1.0 - wv) * 100, 0, 100)

        # 5. Balance Trend Score (0-100)
        bt_score = np.clip((bt + 1000) / 2000 * 100, 0, 100)

        # Weighted Average
        total_score = (
            sb_score * WEIGHTS['savings_buffer'] +
            ic_score * WEIGHTS['income_consistency'] +
            br_score * WEIGHTS['burn_rate'] +
            wv_score * WEIGHTS['withdrawal_volatility'] +
            bt_score * WEIGHTS['balance_trend']
        )

        # Generate signals
        positive_signals = []
        if sb_score > 70: positive_signals.append("Strong savings buffer")
        if ic_score > 70: positive_signals.append("Consistent income stream")
        if bt_score > 70: positive_signals.append("Improving balance trend")

        risk_signals = []
        if br_score < 40: risk_signals.append("High burn rate (spending > income)")
        if wv_score < 40: risk_signals.append("Highly volatile spending patterns")
        if sb_score < 30: risk_signals.append("Critically low emergency buffer")

        scores.append({
            'score': round(total_score, 1),
            'positive': positive_signals,
            'risk': risk_signals
        })

    return pd.DataFrame(scores)

def generate_recommendation(score, risk_signals):
    """Recommends intervention based on score and risks."""
    if score > 80:
        return "Eligible for starter credit product and financial growth tools."
    elif score > 50:
        return "Recommend savings-builder product and budgeting support."
    else:
        return "Prioritize financial literacy content and emergency fund planning."

if __name__ == "__main__":
    try:
        # Load processed features
        feats = pd.read_csv('ml/final_ml_dataset.csv')
        results = calculate_stability_score(feats)

        final_report = feats[['account_number']].copy()
        final_report['stability_score'] = results['score']
        final_report['positive_signals'] = results['positive'].apply(lambda x: ", ".join(x))
        final_report['risk_signals'] = results['risk'].apply(lambda x: ", ".join(x))

        # Add Recommendation
        final_report['recommendation'] = final_report.apply(
            lambda row: generate_recommendation(row['stability_score'], results.loc[row.name, 'risk']), axis=1
        )

        final_report.to_csv('ml/stability_report.csv', index=False)
        print("✅ Stability Report generated successfully!")
        print(final_report.head())

    except Exception as e:
        print(f"Error: {e}")

import pandas as pd
import numpy as np

def engineer_advanced_features(df):
    """
    Generates advanced financial features using categorized transaction data.
    Expected input: df with columns ['account_number', 'withdrawals', 'deposits', 'balance', 'category', 'transaction_date']
    """
    # Define essential categories
    ESSENTIAL_CATS = ['Housing', 'Utility', 'Transport', 'Food']

    # Create a flag for essential spending
    df['is_essential'] = df['category'].isin(ESSENTIAL_CATS).astype(int)

    features = []

    for acc, group in df.groupby('account_number'):
        # 1. Basic Aggregates
        total_withdrawals = group['withdrawals'].sum()
        total_deposits = group['deposits'].sum()

        # 2. Spending Analysis
        essential_spend = group[group['is_essential'] == 1]['withdrawals'].sum()
        discretionary_spend = total_withdrawals - essential_spend

        essential_ratio = essential_spend / total_withdrawals if total_withdrawals > 0 else 0
        discretionary_ratio = discretionary_spend / total_withdrawals if total_withdrawals > 0 else 0

        # 3. Stability & Volatility
        # Withdrawal Volatility: Coefficient of Variation (std/mean)
        mean_withdrawal = group['withdrawals'].mean()
        withdrawal_volatility = group['withdrawals'].std() / mean_withdrawal if mean_withdrawal > 0 else 0

        # 4. Savings Buffer
        # Current balance relative to monthly burn rate
        avg_monthly_spend = total_withdrawals / group['transaction_date'].dt.to_period('M').nunique()
        final_balance = group['balance'].iloc[-1]
        savings_buffer = final_balance / avg_monthly_spend if avg_monthly_spend > 0 else 0

        # 5. Income Consistency
        # Monthly income sum -> check variance
        monthly_income = group.groupby(group['transaction_date'].dt.to_period('M'))['deposits'].sum()
        income_consistency = monthly_income.std() / monthly_income.mean() if monthly_income.mean() > 0 else 1.0
        # Note: Lower consistency value = more stable

        # 6. Balance Trend (Simple linear slope)
        # x = index of transaction, y = balance
        if len(group) > 1:
            x = np.arange(len(group))
            y = group['balance'].values
            slope = np.polyfit(x, y, 1)[0]
        else:
            slope = 0

        features.append({
            'account_number': acc,
            'burn_rate': total_withdrawals / total_deposits if total_deposits > 0 else 1.0,
            'essential_ratio': essential_ratio,
            'discretionary_ratio': discretionary_ratio,
            'withdrawal_volatility': withdrawal_volatility,
            'savings_buffer': savings_buffer,
            'income_consistency': income_consistency,
            'balance_trend': slope,
            'final_balance': final_balance
        })

    return pd.DataFrame(features)

if __name__ == "__main__":
    # Test with sample data
    try:
        df = pd.read_csv('ml/categorized_transactions.csv')
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        feats = engineer_advanced_features(df)
        print("Feature Engineering complete. Preview:\n", feats.head())
    except Exception as e:
        print(f"Error: {e}")

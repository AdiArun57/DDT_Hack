import pandas as pd
import numpy as np
from datetime import timedelta

def load_and_clean_data(filepath):
    """Loads bank transactions and performs initial cleaning."""
    df = pd.read_excel(filepath)

    # Define explicit mapping for this specific dataset
    column_mapping = {
        'Account No': 'account_number',
        'DATE': 'transaction_date',
        'TRANSACTION DETAILS': 'transaction_description',
        'WITHDRAWAL AMT': 'withdrawals',
        'DEPOSIT AMT': 'deposits',
        'BALANCE AMT': 'balance'
    }

    # Rename columns based on mapping, ignore columns not in map
    df = df.rename(columns=column_mapping)

    # Keep only the columns we need
    df = df[list(column_mapping.values())]

    # 1. Clean account numbers (remove trailing quotes/whitespace)
    df['account_number'] = df['account_number'].astype(str).str.replace("'", "").str.strip()

    # 2. Handle numeric columns (coerce to float, replace NaN with 0)
    numeric_cols = ['withdrawals', 'deposits', 'balance']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 3. Data noise filter: Remove obviously corrupted balance records (e.g., > 100M or < -100M)
    # We'll relax this to allow the model to see the actual data distribution,
    # but keep a reasonable cap to avoid infinite values.
    df = df[(df['balance'] < 1e12) & (df['balance'] > -1e12)]

    # 4. Data type conversion
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    df = df.sort_values(['account_number', 'transaction_date'])

    return df

def calculate_stress_target(df, income_percentage=0.1):
    """
    Labels financial stress based on balance vs income.
    Target = 1 if balance < (Average Monthly Income * income_percentage)
    """
    # Calculate monthly income per account
    # Assuming 'deposits' are income
    df['month'] = df['transaction_date'].dt.to_period('M')
    monthly_income = df.groupby(['account_number', 'month'])['deposits'].sum().reset_index()
    avg_income = monthly_income.groupby('account_number')['deposits'].mean().reset_index()
    avg_income.columns = ['account_number', 'avg_monthly_income']

    # Merge avg income back to main df
    df = df.merge(avg_income, on='account_number', how='left')

    # Target: 1 if balance < (avg_income * income_percentage)
    df['is_stressed'] = (df['balance'] < (df['avg_monthly_income'] * income_percentage)).astype(int)

    return df

def extract_financial_features(df):
    """Generates behavioral features for the ML model."""
    features = []

    for acc, group in df.groupby('account_number'):
        # Total spending vs Income
        total_withdrawals = group['withdrawals'].sum()
        total_deposits = group['deposits'].sum()

        # Volatility: Std dev of withdrawals
        withdrawal_volatility = group['withdrawals'].std()

        # Savings Buffer: Final Balance / Avg Monthly Spend
        avg_monthly_spend = group['withdrawals'].sum() / group['month'].nunique()
        final_balance = group['balance'].iloc[-1]
        savings_buffer = final_balance / avg_monthly_spend if avg_monthly_spend > 0 else 0

        # Income Consistency: Ratio of min monthly deposit to max monthly deposit
        monthly_deps = group.groupby('month')['deposits'].sum()
        income_consistency = monthly_deps.min() / monthly_deps.max() if monthly_deps.max() > 0 else 0

        features.append({
            'account_number': acc,
            'burn_rate': total_withdrawals / total_deposits if total_deposits > 0 else 1.0,
            'withdrawal_volatility': withdrawal_volatility,
            'savings_buffer': savings_buffer,
            'income_consistency': income_consistency,
            'final_balance': final_balance
        })

    return pd.DataFrame(features)

if __name__ == "__main__":
    # Example usage
    try:
        data_path = 'data/bank.xlsx'
        df = load_and_clean_data(data_path)
        df_stressed = calculate_stress_target(df)
        features = extract_financial_features(df_stressed)
        print("Preprocessing complete. Features head:\n", features.head())
    except Exception as e:
        print(f"Error: {e}")

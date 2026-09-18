import pandas as pd
from data_preprocessing import load_and_clean_data
from transaction_categorizer import TransactionCategorizer
import os

def main():
    print("🚀 Starting Phase 2: Transaction Categorization...")

    data_path = 'data/bank.xlsx'
    output_path = 'ml/categorized_transactions.csv'

    if not os.path.exists(data_path):
        print(f"❌ Error: Could not find data at {data_path}")
        return

    try:
        # 1. Load data
        print("🧹 Loading data...")
        df = load_and_clean_data(data_path)

        # 2. Define Seed Labels (The "Few Examples" concept)
        # In a real app, these would come from a user UI.
        examples = {
            'Housing': ['UPI RENT PAYMENT', 'HOUSE RENT', 'MORTGAGE PAYMENT', 'SOCIETY MAINTENANCE'],
            'Income': ['NEFT SALARY CREDIT', 'DIRECT DEPOSIT SALARY', 'INTEREST CREDIT', 'BONUS'],
            'Food': ['SWIGGY ORDER', 'ZOMATO DELIVERY', 'RESTAURANT', 'STARBUCKS', 'GROCERIES'],
            'Cash': ['ATM CASH WITHDRAWAL', 'CASH DEPOSIT ATM', 'CASH WITHDRAWAL'],
            'Transport': ['UBER TRIP', 'OLA CAB', 'METRO RECHARGE', 'FUEL STATION', 'PETROL'],
            'Shopping': ['AMAZON INDIA', 'FLIPKART', 'MYNTRA', 'CLOTHING STORE'],
            'Utility': ['ELECTRICITY BILL', 'WATER BILL', 'INTERNET RECHARGE', 'MOBILE BILL']
        }

        # 3. Initialize and fit the local ML categorizer
        categorizer = TransactionCategorizer()
        categorizer.fit(examples)

        # 4. Categorize all transactions
        print("🏷️ Categorizing transactions (this may take a minute)...")
        # Ensure descriptions are strings to avoid 'Unsupported input type: int' error
        df['category'] = df['transaction_description'].astype(str).apply(categorizer.predict)

        # 5. Save the results
        df.to_csv(output_path, index=False)
        print(f"✅ Phase 2 Complete! Categorized data saved to: {output_path}")

        # Print distribution
        print("\n--- Category Distribution ---")
        print(df['category'].value_counts())

    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()

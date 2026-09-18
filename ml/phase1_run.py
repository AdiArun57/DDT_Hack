import pandas as pd
from data_preprocessing import load_and_clean_data, calculate_stress_target, extract_financial_features
import os

def main():
    print("🚀 Starting Phase 1: Data Cleaning & Target Labeling...")

    # File paths
    data_path = 'data/bank.xlsx'
    output_path = 'ml/processed_financial_data.csv'

    if not os.path.exists(data_path):
        print(f"❌ Error: Could not find data at {data_path}")
        return

    try:
        # 1. Load and clean
        print("🧹 Loading and cleaning data...")
        df = load_and_clean_data(data_path)

        # 2. Create the stress target (10% of income as default)
        print("🎯 Calculating financial stress targets...")
        df_stressed = calculate_stress_target(df, income_percentage=0.1)

        # 3. Extract behavioral features
        print("🧬 Extracting behavioral features...")
        features = extract_financial_features(df_stressed)

        # Merge features back with the targets for the ML model
        # We take the last known stress status for each account
        targets = df_stressed.groupby('account_number')['is_stressed'].last().reset_index()
        final_dataset = features.merge(targets, on='account_number')

        # Save for Phase 2 and 4
        final_dataset.to_csv(output_path, index=False)
        print(f"✅ Phase 1 Complete! Processed data saved to: {output_path}")
        print("\n--- Data Preview ---")
        print(final_dataset.head())

    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()

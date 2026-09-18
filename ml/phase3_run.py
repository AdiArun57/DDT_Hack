import pandas as pd
from data_preprocessing import load_and_clean_data, calculate_stress_target
from feature_engineering import engineer_advanced_features
import os

def main():
    print("🚀 Starting Phase 3: Advanced Feature Engineering...")

    data_path = 'data/bank.xlsx'
    output_path = 'ml/final_ml_dataset.csv'

    if not os.path.exists(data_path):
        print(f"❌ Error: Could not find data at {data_path}")
        return

    try:
        # 1. Load and Clean (Phase 1)
        print("🧹 Loading and cleaning data...")
        df = load_and_clean_data(data_path)

        # 2. Add Categorization (Phase 2)
        # We load the categorized transactions if they exist, or we'd need to re-run categorizer
        if os.path.exists('ml/categorized_transactions.csv'):
            print("🏷️ Merging categorization data...")
            cat_df = pd.read_csv('ml/categorized_transactions.csv')
            # Ensure types match for merging
            cat_df['account_number'] = cat_df['account_number'].astype(str)
            df['account_number'] = df['account_number'].astype(str)

            # We only need the category mapping for the most recent records or all?
            # The categorizer was applied to the whole file, so we merge on everything
            # Note: Since the original data might have duplicate account/date, we use the index or merge cautiously
            # In Phase 2 we just added a column, so we can just re-run the a bit of the categorize logic or merge
            # For simplicity in the MVP, let's just re-read the categorized file and join by index if they are same length
            if len(df) == len(cat_df):
                df['category'] = cat_df['category']
            else:
                # Fallback: merge on account and date (risky if duplicates)
                cat_df['transaction_date'] = pd.to_datetime(cat_df['transaction_date'])
                df = df.merge(cat_df[['account_number', 'transaction_date', 'category']],
                             on=['account_number', 'transaction_date'], how='left')
        else:
            print("❌ Error: categorized_transactions.csv not found. Please run phase2_run.py first.")
            return

        # 3. Calculate Stress Target (Phase 1 Logic)
        print("🎯 Calculating stress targets...")
        df_stressed = calculate_stress_target(df)

        # 4. Advanced Feature Engineering (Phase 3)
        print("🧬 Engineering advanced behavioral features...")
        features = engineer_advanced_features(df_stressed)

        # 5. Final Merge: Features + Targets
        targets = df_stressed.groupby('account_number')['is_stressed'].last().reset_index()
        final_dataset = features.merge(targets, on='account_number')

        # Save the final golden dataset for the ML model (Phase 4)
        final_dataset.to_csv(output_path, index=False)
        print(f"✅ Phase 3 Complete! Final ML dataset saved to: {output_path}")

        print("\n--- Final Dataset Preview ---")
        print(final_dataset.head())
        print("\nDataset Shape:", final_dataset.shape)

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

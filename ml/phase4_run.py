import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import joblib
import os

def train_stress_model(dataset_path):
    """Trains a Random Forest model to predict financial stress."""
    # 1. Load data
    df = pd.read_csv(dataset_path)

    # Drop account_number for training
    X = df.drop(['account_number', 'is_stressed'], axis=1)
    y = df['is_stressed']

    # Handle NaN values (some volatility or buffer calculations might result in NaN)
    X = X.fillna(0)

    # 2. Split data
    # Handle very small datasets where stratification is impossible
    if len(df) < 10 or y.value_counts().min() < 2:
        print("⚠️ Dataset too small or imbalanced for stratified split. Using simple split.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Train Random Forest
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # 4. Evaluation
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"ROC AUC: {roc_auc:.2f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # 5. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix: Financial Stress Prediction')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('ml/confusion_matrix.png')
    plt.close()

    # 6. Feature Importance
    importances = model.feature_importances_
    feature_names = X.columns
    sorted_idx = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), importances[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.title('Feature Importance for Stress Prediction')
    plt.xlabel('Importance Score')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('ml/feature_importance.png')
    plt.close()

    # 7. Save model and metrics
    joblib.dump(model, 'ml/stress_model.joblib')

    metrics = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'feature_importances': dict(zip(feature_names[sorted_idx], importances[sorted_idx]))
    }

    return metrics

if __name__ == "__main__":
    try:
        metrics = train_stress_model('ml/final_ml_dataset.csv')
        print("\n✅ Model Training Complete!")
        print("Metrics:", metrics)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

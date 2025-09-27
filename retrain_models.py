# ===============================
# Retrain Models for Compatibility
# ===============================

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import pickle


def retrain_models():
    """Retrain both models with current versions to ensure compatibility"""

    print("Loading dataset...")
    # Load dataset
    dataset = pd.read_csv("fraud-detection.csv", encoding="utf-8-sig")

    print("Preprocessing data...")
    # Define columns exactly as in original notebook
    categorical_columns = ["Location", "MerchantCategory"]
    label_columns = ["CardHolderAge", "IsFraud"]

    # Handle missing values in categorical columns first
    dataset["Location"] = dataset["Location"].fillna("Unknown")
    dataset["MerchantCategory"] = dataset["MerchantCategory"].fillna("Unknown")

    # Encode categorical columns
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    dataset[categorical_columns] = oe.fit_transform(dataset[categorical_columns])

    # Handle missing values in numeric columns
    numeric_cols = dataset.select_dtypes(include="number").columns
    feature_cols = [col for col in numeric_cols if col not in label_columns]

    # Fill NaN in numeric features
    dataset[feature_cols] = dataset[feature_cols].fillna(dataset[feature_cols].mean())

    # Fill NaN in label columns and round
    dataset[label_columns] = (
        dataset[label_columns].fillna(dataset[label_columns].mean()).round()
    )

    # Scale numeric feature columns
    scaler = StandardScaler()
    dataset[feature_cols] = scaler.fit_transform(dataset[feature_cols])

    print("Splitting data...")
    # Split features and target - exclude CardHolderAge as it's auxiliary
    X = dataset.drop(
        columns=label_columns
    )  # This removes both CardHolderAge and IsFraud
    y = dataset["IsFraud"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    print("Training Logistic Regression...")
    # Train Logistic Regression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    lr_score = lr.score(X_test, y_test)
    print(f"Logistic Regression Accuracy: {lr_score:.3f}")

    print("Training XGBoost...")
    # Train XGBoost with current version
    xg = XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
    )
    xg.fit(X_train, y_train)
    xg_score = xg.score(X_test, y_test)
    print(f"XGBoost Accuracy: {xg_score:.3f}")

    print("Saving models...")
    # Save models
    with open("logistic.pkl", "wb") as f:
        pickle.dump(lr, f)

    with open("xgboost.pkl", "wb") as f:
        pickle.dump(xg, f)

    # Also save preprocessing objects for consistency
    with open("ordinal_encoder.pkl", "wb") as f:
        pickle.dump(oe, f)

    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Save feature names for reference
    feature_names = list(X.columns)
    with open("feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)

    print("\nModel Training Complete!")
    print(f"Feature columns used: {feature_names}")
    print(f"Logistic Regression saved to: logistic.pkl")
    print(f"XGBoost saved to: xgboost.pkl")
    print(f"Preprocessing objects saved")

    return lr, xg, oe, scaler, feature_names


if __name__ == "__main__":
    print("Retraining models for compatibility...")
    try:
        lr, xg, oe, scaler, features = retrain_models()
        print("\n✅ All models retrained successfully!")
        print("\nYou can now run the Streamlit app with:")
        print("streamlit run fraud_detection_app.py")
    except Exception as e:
        print(f"\n❌ Error during retraining: {str(e)}")
        print("Please check that fraud-detection.csv exists in the current directory")

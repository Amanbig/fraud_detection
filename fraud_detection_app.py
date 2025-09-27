import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .fraud-detected {
        background-color: #ffebee;
        border: 2px solid #f44336;
        color: #c62828;
    }
    .no-fraud {
        background-color: #e8f5e8;
        border: 2px solid #4caf50;
        color: #2e7d32;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Load models and preprocessing objects
@st.cache_resource
def load_models_and_preprocessors():
    """Load the trained models and preprocessing objects"""
    try:
        with open("logistic.pkl", "rb") as f:
            logistic_model = pickle.load(f)
        with open("xgboost.pkl", "rb") as f:
            xgboost_model = pickle.load(f)
        with open("ordinal_encoder.pkl", "rb") as f:
            ordinal_encoder = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open("feature_names.pkl", "rb") as f:
            feature_names = pickle.load(f)

        return logistic_model, xgboost_model, ordinal_encoder, scaler, feature_names
    except FileNotFoundError as e:
        st.error(
            f"Required files not found: {str(e)}\nPlease run 'python retrain_models.py' first to generate all required files."
        )
        return None, None, None, None, None


@st.cache_data
def load_data():
    """Load and preprocess the dataset"""
    try:
        dataset = pd.read_csv("fraud-detection.csv", encoding="utf-8-sig")
        return dataset
    except FileNotFoundError:
        st.error(
            "Dataset not found! Please ensure fraud-detection.csv is in the current directory."
        )
        return None


def preprocess_data(data):
    """Preprocess data exactly as done in training"""
    # Create a copy to avoid modifying original
    processed_data = data.copy()

    # Define columns
    categorical_columns = ["Location", "MerchantCategory"]
    label_columns = ["CardHolderAge", "IsFraud"]

    # Encode categorical columns
    oe = OrdinalEncoder()
    processed_data[categorical_columns] = oe.fit_transform(
        processed_data[categorical_columns]
    )

    # Handle missing values
    numeric_cols = processed_data.select_dtypes(include="number").columns
    feature_cols = [col for col in numeric_cols if col not in label_columns]

    # Fill NaN in numeric features
    processed_data[feature_cols] = processed_data[feature_cols].fillna(
        processed_data[feature_cols].mean()
    )

    # Fill NaN in label columns and round
    processed_data[label_columns] = (
        processed_data[label_columns]
        .fillna(processed_data[label_columns].mean())
        .round()
    )

    # Scale numeric feature columns
    scaler = StandardScaler()
    processed_data[feature_cols] = scaler.fit_transform(processed_data[feature_cols])

    return processed_data, oe, scaler, feature_cols


def preprocess_single_transaction(transaction_data, oe, scaler, feature_names, dataset):
    """Preprocess a single transaction for prediction using saved preprocessors"""
    # Create a DataFrame from the input
    df = pd.DataFrame([transaction_data])

    # Handle categorical columns - fill missing values first
    categorical_columns = ["Location", "MerchantCategory"]
    df["Location"] = df["Location"].fillna("Unknown")
    df["MerchantCategory"] = df["MerchantCategory"].fillna("Unknown")

    # Use the saved ordinal encoder
    df[categorical_columns] = oe.transform(df[categorical_columns])

    # Handle missing values in numeric columns using dataset means
    numeric_cols = ["TransactionID", "Amount", "Time", "CardHolderAge"]
    for col in numeric_cols:
        if pd.isna(df[col].iloc[0]):
            df[col] = dataset[col].mean()

    # Get only the feature columns used in training (excluding CardHolderAge and IsFraud)
    df_features = df[feature_names]

    # Scale features using saved scaler
    df_features_scaled = pd.DataFrame(
        scaler.transform(df_features), columns=feature_names
    )

    return df_features_scaled


# Main app
def main():
    st.markdown(
        '<h1 class="main-header">🔒 Fraud Detection System</h1>', unsafe_allow_html=True
    )

    # Load data and models
    dataset = load_data()
    logistic_model, xgboost_model, oe, scaler, feature_names = (
        load_models_and_preprocessors()
    )

    if (
        dataset is None
        or logistic_model is None
        or any(x is None for x in [oe, scaler, feature_names])
    ):
        st.stop()

    # Preprocess the dataset for analytics
    processed_dataset, _, _, _ = preprocess_data(dataset)

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🔍 Single Prediction", "📊 Dataset Overview", "📈 Model Analytics"],
    )

    if page == "🔍 Single Prediction":
        single_prediction_page(
            dataset, logistic_model, xgboost_model, oe, scaler, feature_names
        )
    elif page == "📊 Dataset Overview":
        dataset_overview_page(dataset, processed_dataset)
    elif page == "📈 Model Analytics":
        model_analytics_page(
            dataset, processed_dataset, logistic_model, xgboost_model, feature_names
        )


def single_prediction_page(
    dataset, logistic_model, xgboost_model, oe, scaler, feature_names
):
    st.header("🔍 Fraud Detection - Single Transaction")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Transaction Details")

        # Get unique values for dropdowns
        locations = [""] + list(dataset["Location"].dropna().unique())
        merchant_categories = [""] + list(dataset["MerchantCategory"].dropna().unique())

        transaction_id = st.number_input("Transaction ID", min_value=0, value=1000)
        amount = st.number_input("Amount ($)", min_value=0.0, value=500.0, step=0.01)
        time = st.number_input(
            "Time", min_value=0, value=85000, help="Time in seconds since epoch"
        )
        location = st.selectbox("Location", locations)
        merchant_category = st.selectbox("Merchant Category", merchant_categories)
        cardholder_age = st.number_input(
            "Cardholder Age", min_value=18, max_value=100, value=35
        )

        if st.button("🔍 Predict Fraud", use_container_width=True):
            # Prepare transaction data
            transaction_data = {
                "TransactionID": transaction_id,
                "Amount": amount,
                "Time": time,
                "Location": location if location else np.nan,
                "MerchantCategory": merchant_category if merchant_category else np.nan,
                "CardHolderAge": cardholder_age,
                "IsFraud": 0,  # Placeholder
            }

            try:
                # Preprocess the transaction
                processed_transaction = preprocess_single_transaction(
                    transaction_data, oe, scaler, feature_names, dataset
                )

                # Make predictions
                lr_prediction = logistic_model.predict(processed_transaction)[0]
                lr_probability = logistic_model.predict_proba(processed_transaction)[0]

                # Handle XGBoost prediction with fallback
                if xgboost_model is not None:
                    try:
                        xg_prediction = xgboost_model.predict(processed_transaction)[0]
                        xg_probability = xgboost_model.predict_proba(
                            processed_transaction
                        )[0]
                    except Exception as e:
                        st.warning(f"XGBoost prediction failed: {str(e)}")
                        xg_prediction = lr_prediction  # Fallback to logistic
                        xg_probability = lr_probability
                else:
                    st.warning(
                        "XGBoost model unavailable, using Logistic Regression only"
                    )
                    xg_prediction = lr_prediction
                    xg_probability = lr_probability

                with col2:
                    st.subheader("Prediction Results")

                    # Logistic Regression Results
                    st.markdown("### Logistic Regression")
                    fraud_class = "fraud-detected" if lr_prediction == 1 else "no-fraud"
                    result_text = (
                        "🚨 FRAUD DETECTED"
                        if lr_prediction == 1
                        else "✅ LEGITIMATE TRANSACTION"
                    )

                    st.markdown(
                        f"""
                    <div class="prediction-box {fraud_class}">
                        <h4>{result_text}</h4>
                        <p><strong>Fraud Probability:</strong> {lr_probability[1]:.2%}</p>
                        <p><strong>Legitimate Probability:</strong> {lr_probability[0]:.2%}</p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    # XGBoost Results
                    if xgboost_model is not None:
                        st.markdown("### XGBoost")
                        fraud_class = (
                            "fraud-detected" if xg_prediction == 1 else "no-fraud"
                        )
                        result_text = (
                            "🚨 FRAUD DETECTED"
                            if xg_prediction == 1
                            else "✅ LEGITIMATE TRANSACTION"
                        )

                        st.markdown(
                            f"""
                        <div class="prediction-box {fraud_class}">
                            <h4>{result_text}</h4>
                            <p><strong>Fraud Probability:</strong> {xg_probability[1]:.2%}</p>
                            <p><strong>Legitimate Probability:</strong> {xg_probability[0]:.2%}</p>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown("### XGBoost")
                        st.warning(
                            "XGBoost model unavailable - showing Logistic Regression results"
                        )

                    # Probability comparison chart
                    if xgboost_model is not None:
                        fig = go.Figure(
                            data=[
                                go.Bar(
                                    name="Logistic Regression",
                                    x=["Legitimate", "Fraud"],
                                    y=[lr_probability[0], lr_probability[1]],
                                ),
                                go.Bar(
                                    name="XGBoost",
                                    x=["Legitimate", "Fraud"],
                                    y=[xg_probability[0], xg_probability[1]],
                                ),
                            ]
                        )
                        fig.update_layout(
                            title="Model Probability Comparison",
                            xaxis_title="Prediction",
                            yaxis_title="Probability",
                            barmode="group",
                            height=400,
                        )
                    else:
                        fig = go.Figure(
                            data=[
                                go.Bar(
                                    name="Logistic Regression",
                                    x=["Legitimate", "Fraud"],
                                    y=[lr_probability[0], lr_probability[1]],
                                ),
                            ]
                        )
                        fig.update_layout(
                            title="Logistic Regression Probabilities",
                            xaxis_title="Prediction",
                            yaxis_title="Probability",
                            height=400,
                        )
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")


def dataset_overview_page(dataset, processed_dataset):
    st.header("📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Transactions", len(dataset))
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        fraud_count = dataset["IsFraud"].sum()
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Fraud Cases", fraud_count)
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        fraud_rate = (fraud_count / len(dataset)) * 100
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        avg_amount = dataset["Amount"].mean()
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Avg Transaction", f"${avg_amount:.2f}")
        st.markdown("</div>", unsafe_allow_html=True)

    # Dataset preview
    st.subheader("Dataset Sample")
    st.dataframe(dataset.head(10), use_container_width=True)

    # Visualizations
    col1, col2 = st.columns(2)

    with col1:
        # Fraud distribution
        fraud_dist = dataset["IsFraud"].value_counts()
        fig = px.pie(
            values=fraud_dist.values,
            names=["Legitimate", "Fraud"],
            title="Fraud vs Legitimate Transactions",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Amount distribution by fraud
        fig = px.box(
            dataset,
            x="IsFraud",
            y="Amount",
            title="Transaction Amount Distribution by Fraud Status",
        )
        fig.update_xaxis(ticktext=["Legitimate", "Fraud"], tickvals=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Merchant category distribution
        merchant_counts = dataset["MerchantCategory"].value_counts()
        fig = px.bar(
            x=merchant_counts.index,
            y=merchant_counts.values,
            title="Transactions by Merchant Category",
        )
        fig.update_xaxis(title="Merchant Category")
        fig.update_yaxis(title="Count")
        st.plotly_chart(fig, use_container_width=True)

        # Location distribution
        location_counts = dataset["Location"].value_counts().head(10)
        fig = px.bar(
            x=location_counts.index,
            y=location_counts.values,
            title="Top 10 Transaction Locations",
        )
        fig.update_xaxis(title="Location")
        fig.update_yaxis(title="Count")
        st.plotly_chart(fig, use_container_width=True)


def model_analytics_page(
    dataset, processed_dataset, logistic_model, xgboost_model, feature_names
):
    st.header("📈 Model Analytics")

    # Prepare data for evaluation
    X = processed_dataset.drop(columns=["CardHolderAge", "IsFraud"])
    y = processed_dataset["IsFraud"]

    # Make predictions
    lr_predictions = logistic_model.predict(X)
    lr_probabilities = logistic_model.predict_proba(X)[:, 1]

    # Handle XGBoost predictions with error handling
    if xgboost_model is not None:
        try:
            xg_predictions = xgboost_model.predict(X)
            xg_probabilities = xgboost_model.predict_proba(X)[:, 1]
            xg_available = True
        except Exception as e:
            st.warning(f"XGBoost evaluation failed: {str(e)}")
            xg_predictions = lr_predictions
            xg_probabilities = lr_probabilities
            xg_available = False
    else:
        xg_predictions = lr_predictions
        xg_probabilities = lr_probabilities
        xg_available = False

    # Model accuracy
    lr_accuracy = (lr_predictions == y).mean()
    xg_accuracy = (xg_predictions == y).mean() if xg_available else 0

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Logistic Regression Accuracy", f"{lr_accuracy:.1%}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if xg_available:
            st.metric("XGBoost Accuracy", f"{xg_accuracy:.1%}")
        else:
            st.metric("XGBoost Accuracy", "Unavailable")
        st.markdown("</div>", unsafe_allow_html=True)

    # Probability distributions
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Logistic Regression - Fraud Probabilities")
        fig = px.histogram(
            x=lr_probabilities,
            nbins=50,
            title="Distribution of Fraud Probabilities (Logistic Regression)",
        )
        fig.update_xaxis(title="Fraud Probability")
        fig.update_yaxis(title="Count")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("XGBoost - Fraud Probabilities")
        if xg_available:
            fig = px.histogram(
                x=xg_probabilities,
                nbins=50,
                title="Distribution of Fraud Probabilities (XGBoost)",
            )
            fig.update_xaxis(title="Fraud Probability")
            fig.update_yaxis(title="Count")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("XGBoost model unavailable")

    # Feature importance (for XGBoost)
    if xg_available and hasattr(xgboost_model, "feature_importances_"):
        st.subheader("XGBoost Feature Importance")
        feature_importance = pd.DataFrame(
            {"feature": feature_names, "importance": xgboost_model.feature_importances_}
        ).sort_values("importance", ascending=False)

        fig = px.bar(
            feature_importance,
            x="importance",
            y="feature",
            orientation="h",
            title="Feature Importance (XGBoost)",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    elif not xg_available:
        st.info("XGBoost feature importance unavailable - model not loaded")

    # Confusion matrices
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Logistic Regression - Confusion Matrix")
        cm_lr = confusion_matrix(y, lr_predictions)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Logistic Regression")
        st.pyplot(fig)

    with col2:
        st.subheader("XGBoost - Confusion Matrix")
        if xg_available:
            cm_xg = confusion_matrix(y, xg_predictions)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm_xg, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("XGBoost")
            st.pyplot(fig)
        else:
            st.warning("XGBoost confusion matrix unavailable")


if __name__ == "__main__":
    main()

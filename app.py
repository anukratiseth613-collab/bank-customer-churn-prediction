import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------------

st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

# ---------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "bank_churn_model.pkl"
FEATURES_PATH = BASE_DIR / "bank_churn_features.pkl"

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

try:
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)

except Exception as e:
    st.error("Unable to load the prediction model.")
    st.write("Please make sure the model files are uploaded correctly.")
    st.code(str(e))
    st.stop()

# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("🏦 Bank Customer Churn Prediction")

st.write(
    "Predict customer churn probability and identify retention risk."
)

st.divider()

# ---------------------------------------------------------
# SIDEBAR - CUSTOMER INFORMATION
# ---------------------------------------------------------

st.sidebar.header("👤 Customer Information")

credit_score = st.sidebar.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=650
)

geography = st.sidebar.selectbox(
    "Geography",
    ["France", "Germany", "Spain"]
)

gender = st.sidebar.selectbox(
    "Gender",
    ["Female", "Male"]
)

age = st.sidebar.slider(
    "Age",
    min_value=18,
    max_value=100,
    value=40
)

tenure = st.sidebar.slider(
    "Tenure (Years)",
    min_value=0,
    max_value=10,
    value=5
)

balance = st.sidebar.number_input(
    "Account Balance",
    min_value=0.0,
    value=75000.0
)

products = st.sidebar.slider(
    "Number of Products",
    min_value=1,
    max_value=4,
    value=2
)

credit_card = st.sidebar.selectbox(
    "Has Credit Card?",
    ["Yes", "No"]
)

active_member = st.sidebar.selectbox(
    "Is Active Member?",
    ["Yes", "No"]
)

salary = st.sidebar.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=100000.0
)

# ---------------------------------------------------------
# CONVERT INPUTS
# ---------------------------------------------------------

has_credit_card = 1 if credit_card == "Yes" else 0

is_active = 1 if active_member == "Yes" else 0

# ---------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------

balance_salary = balance / (salary + 1)

product_density = products / (tenure + 1)

engagement_product = is_active * products

age_tenure = age * tenure

# ---------------------------------------------------------
# CREATE CUSTOMER DATAFRAME
# ---------------------------------------------------------

customer = pd.DataFrame({
    "CreditScore": [credit_score],
    "Geography": [geography],
    "Gender": [gender],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [products],
    "HasCrCard": [has_credit_card],
    "IsActiveMember": [is_active],
    "EstimatedSalary": [salary],
    "Balance_to_Salary": [balance_salary],
    "Product_Density": [product_density],
    "Engagement_Product": [engagement_product],
    "Age_Tenure": [age_tenure]
})

# ---------------------------------------------------------
# ONE-HOT ENCODING
# ---------------------------------------------------------

customer = pd.get_dummies(
    customer,
    columns=["Geography", "Gender"]
)

# Make sure columns exactly match the model features
customer = customer.reindex(
    columns=features,
    fill_value=0
)

# ---------------------------------------------------------
# PREDICTION SECTION
# ---------------------------------------------------------

st.subheader("🔍 Churn Prediction")

if st.button("Predict Churn Risk", type="primary"):

    try:

        probability = model.predict_proba(customer)[0, 1]

        percentage = probability * 100

        # -------------------------------------------------
        # RISK CATEGORY
        # -------------------------------------------------

        if probability < 0.30:
            risk = "Low Risk"

        elif probability < 0.60:
            risk = "Medium Risk"

        else:
            risk = "High Risk"

        # -------------------------------------------------
        # DISPLAY RESULTS
        # -------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Churn Probability",
                f"{percentage:.2f}%"
            )

        with col2:
            st.metric(
                "Risk Category",
                risk
            )

        # Probability bar
        st.progress(float(probability))

        # -------------------------------------------------
        # RISK MESSAGE
        # -------------------------------------------------

        if risk == "High Risk":

            st.error(
                "⚠️ High churn risk. "
                "Consider proactive customer retention action."
            )

        elif risk == "Medium Risk":

            st.warning(
                "⚠️ Medium churn risk. "
                "Customer engagement should be monitored."
            )

        else:

            st.success(
                "✅ Low churn risk. "
                "Customer appears relatively stable."
            )

        st.divider()

        # -------------------------------------------------
        # CUSTOMER SUMMARY
        # -------------------------------------------------

        st.subheader("📋 Customer Summary")

        summary_col1, summary_col2, summary_col3 = st.columns(3)

        with summary_col1:

            st.write("**Age:**", age)

            st.write("**Gender:**", gender)

            st.write("**Geography:**", geography)

        with summary_col2:

            st.write("**Credit Score:**", credit_score)

            st.write("**Tenure:**", f"{tenure} years")

            st.write("**Products:**", products)

        with summary_col3:

            st.write(
                "**Balance:**",
                f"₹{balance:,.2f}"
            )

            st.write(
                "**Estimated Salary:**",
                f"₹{salary:,.2f}"
            )

            st.write(
                "**Active Member:**",
                active_member
            )

    except Exception as e:

        st.error("Prediction could not be completed.")

        st.code(str(e))

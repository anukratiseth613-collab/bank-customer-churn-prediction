import streamlit as st
import pandas as pd
import numpy as np
import joblib


# --------------------------------------------------
# LOAD MODEL FILES
# --------------------------------------------------

model = joblib.load("bank_churn_model.pkl")
features = joblib.load("bank_churn_features.pkl")


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🏦 Bank Customer Churn Prediction")
st.write("Predict customer churn probability and identify retention risk.")


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("Customer Information")


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


# --------------------------------------------------
# CONVERT INPUTS
# --------------------------------------------------

has_credit_card = 1 if credit_card == "Yes" else 0
is_active = 1 if active_member == "Yes" else 0


# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------

balance_salary = balance / (salary + 1)
product_density = products / (tenure + 1)
engagement_product = is_active * products
age_tenure = age * tenure


# --------------------------------------------------
# CREATE CUSTOMER DATAFRAME
# --------------------------------------------------

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


# --------------------------------------------------
# ONE-HOT ENCODING
# --------------------------------------------------

customer = pd.get_dummies(
    customer,
    columns=["Geography", "Gender"]
)


# --------------------------------------------------
# MATCH TRAINING FEATURES
# --------------------------------------------------

customer = customer.reindex(
    columns=features,
    fill_value=0
)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button("Predict Churn Risk"):

    probability = model.predict_proba(customer)[0, 1]

    percentage = probability * 100


    # --------------------------------------------------
    # RESULT
    # --------------------------------------------------

    st.subheader("Churn Risk Result")


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Churn Probability",
            f"{percentage:.2f}%"
        )


    with col2:

        if probability < 0.30:

            risk = "Low Risk"

        elif probability < 0.60:

            risk = "Medium Risk"

        else:

            risk = "High Risk"


        st.metric(
            "Risk Category",
            risk
        )


    # --------------------------------------------------
    # PROGRESS BAR
    # --------------------------------------------------

    st.progress(float(probability))


    # --------------------------------------------------
    # RISK MESSAGE
    # --------------------------------------------------

    if risk == "High Risk":

        st.error(
            "⚠️ High churn risk. Consider proactive retention action."
        )

    elif risk == "Medium Risk":

        st.warning(
            "⚠️ Medium churn risk. Customer engagement should be monitored."
        )

    else:

        st.success(
            "✅ Low churn risk. Customer appears relatively stable."
        )


    # --------------------------------------------------
    # CHURN PROBABILITY GRAPH
    # --------------------------------------------------

    st.subheader("Churn Probability Distribution")

    chart_data = pd.DataFrame({
        "Probability": [probability]
    })

    st.bar_chart(chart_data)


    # --------------------------------------------------
    # CUSTOMER SUMMARY
    # --------------------------------------------------

    st.subheader("Customer Summary")

    summary = pd.DataFrame({
        "Feature": [
            "Credit Score",
            "Geography",
            "Gender",
            "Age",
            "Tenure",
            "Account Balance",
            "Number of Products",
            "Credit Card",
            "Active Member",
            "Estimated Salary"
        ],

        "Value": [
            credit_score,
            geography,
            gender,
            age,
            tenure,
            balance,
            products,
            credit_card,
            active_member,
            salary
        ]
    })

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

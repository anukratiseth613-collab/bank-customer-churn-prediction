import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load("/content/bank_churn_model.pkl")
features = joblib.load("/content/bank_churn_features.pkl")


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("🏦 Bank Customer Churn Prediction")

st.write(
    "Predict customer churn probability and analyze customer retention risk."
)


# =========================================================
# SIDEBAR - CUSTOMER INFORMATION
# =========================================================

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


# =========================================================
# CONVERT INPUTS
# =========================================================

has_credit_card = 1 if credit_card == "Yes" else 0

is_active = 1 if active_member == "Yes" else 0


# =========================================================
# FUNCTION TO CREATE CUSTOMER DATA
# =========================================================

def create_customer(
    credit_score,
    geography,
    gender,
    age,
    tenure,
    balance,
    products,
    has_credit_card,
    is_active,
    salary
):

    balance_salary = balance / (salary + 1)

    product_density = products / (tenure + 1)

    engagement_product = is_active * products

    age_tenure = age * tenure

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

    customer = pd.get_dummies(
        customer,
        columns=["Geography", "Gender"]
    )

    customer = customer.reindex(
        columns=features,
        fill_value=0
    )

    return customer


# =========================================================
# PREDICT BUTTON
# =========================================================

if st.button("Predict Churn Risk"):

    customer = create_customer(
        credit_score,
        geography,
        gender,
        age,
        tenure,
        balance,
        products,
        has_credit_card,
        is_active,
        salary
    )

    probability = model.predict_proba(customer)[0, 1]

    percentage = probability * 100


    # =====================================================
    # RISK RESULT
    # =====================================================

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


    st.progress(float(probability))


    # =====================================================
    # RISK MESSAGE
    # =====================================================

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


    # =====================================================
    # PROBABILITY DISTRIBUTION
    # =====================================================

    st.subheader("Churn Probability Distribution")

    chart_data = pd.DataFrame({
        "Status": ["Retain", "Churn"],
        "Probability": [
            1 - probability,
            probability
        ]
    })

    st.bar_chart(
        chart_data.set_index("Status")
    )


    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    st.subheader("Top Churn Risk Factors")

    if hasattr(model, "feature_importances_"):

        importance_df = pd.DataFrame({
            "Feature": features,
            "Importance": model.feature_importances_
        })

        importance_df = importance_df.sort_values(
            "Importance",
            ascending=False
        )

        top_features = importance_df.head(10)

        st.bar_chart(
            top_features.set_index("Feature")
        )

    else:

        st.info(
            "Feature importance visualization is not available for this model."
        )


# =========================================================
# WHAT-IF SCENARIO SIMULATOR
# =========================================================

st.divider()

st.header("🎯 What-If Scenario Simulator")

st.write(
    "Change customer engagement and product values to see how the predicted churn probability changes."
)


# =========================================================
# WHAT-IF INPUTS
# =========================================================

whatif_col1, whatif_col2 = st.columns(2)


with whatif_col1:

    whatif_products = st.slider(
        "Scenario - Number of Products",
        min_value=1,
        max_value=4,
        value=products
    )


with whatif_col2:

    whatif_active = st.selectbox(
        "Scenario - Active Member",
        ["Yes", "No"],
        index=0 if active_member == "Yes" else 1
    )


# =========================================================
# WHAT-IF PREDICTION
# =========================================================

whatif_credit_card = has_credit_card

whatif_is_active = 1 if whatif_active == "Yes" else 0


whatif_customer = create_customer(
    credit_score,
    geography,
    gender,
    age,
    tenure,
    balance,
    whatif_products,
    whatif_credit_card,
    whatif_is_active,
    salary
)


whatif_probability = model.predict_proba(
    whatif_customer
)[0, 1]


whatif_percentage = whatif_probability * 100


# =========================================================
# DISPLAY WHAT-IF RESULT
# =========================================================

st.subheader("Scenario Result")

scenario_col1, scenario_col2 = st.columns(2)


with scenario_col1:

    st.metric(
        "Scenario Churn Probability",
        f"{whatif_percentage:.2f}%"
    )


with scenario_col2:

    if whatif_probability < 0.30:

        scenario_risk = "Low Risk"

    elif whatif_probability < 0.60:

        scenario_risk = "Medium Risk"

    else:

        scenario_risk = "High Risk"


    st.metric(
        "Scenario Risk",
        scenario_risk
    )


# =========================================================
# COMPARISON
# =========================================================

st.subheader("Current vs What-If Scenario")


current_customer = create_customer(
    credit_score,
    geography,
    gender,
    age,
    tenure,
    balance,
    products,
    has_credit_card,
    is_active,
    salary
)


current_probability = model.predict_proba(
    current_customer
)[0, 1]


comparison_df = pd.DataFrame({
    "Scenario": [
        "Current Customer",
        "What-If Scenario"
    ],
    "Churn Probability": [
        current_probability,
        whatif_probability
    ]
})


st.bar_chart(
    comparison_df.set_index("Scenario")
)


# =========================================================
# INTERPRETATION
# =========================================================

change = whatif_probability - current_probability


if change < 0:

    st.success(
        f"✅ The scenario reduces predicted churn risk by "
        f"{abs(change) * 100:.2f} percentage points."
    )

elif change > 0:

    st.warning(
        f"⚠️ The scenario increases predicted churn risk by "
        f"{change * 100:.2f} percentage points."
    )

else:

    st.info(
        "The scenario does not change the predicted churn probability."
    )
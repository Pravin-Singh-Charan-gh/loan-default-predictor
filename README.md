import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide"
)

# ── Load model ────────────────────────────────────────────
model     = joblib.load('model.pkl')
feat_names = joblib.load('features.pkl')

# ── Header ────────────────────────────────────────────────
st.title("🏦 Loan Approval Prediction")
st.markdown("Enter applicant details in the sidebar and click **Predict** to see the result.")
st.divider()

# ── Sidebar: Input form ───────────────────────────────────
st.sidebar.header("Applicant Details")

gender     = st.sidebar.selectbox("Gender",          ["Male", "Female"])
married    = st.sidebar.selectbox("Married",          ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents",       ["0", "1", "2", "3+"])
education  = st.sidebar.selectbox("Education",        ["Graduate", "Not Graduate"])
self_emp   = st.sidebar.selectbox("Self Employed",    ["No", "Yes"])
income     = st.sidebar.number_input("Applicant Income (₹)",    min_value=0, value=5000, step=500)
co_income  = st.sidebar.number_input("Co-applicant Income (₹)", min_value=0, value=0,    step=500)
loan_amt   = st.sidebar.number_input("Loan Amount (thousands)", min_value=0, value=150,  step=10)
loan_term  = st.sidebar.selectbox("Loan Term (months)", [360, 180, 120, 84, 60, 36, 12])
credit     = st.sidebar.selectbox("Credit History",  ["Good (1.0)", "Bad (0.0)"])
property_a = st.sidebar.selectbox("Property Area",   ["Urban", "Semiurban", "Rural"])

predict_btn = st.sidebar.button("🔍 Predict Approval", use_container_width=True)

# ── Encode inputs ─────────────────────────────────────────
enc = {
    "Male": 1,    "Female": 0,
    "Yes": 1,     "No": 0,
    "Graduate": 0,"Not Graduate": 1,
    "Urban": 2,   "Semiurban": 1,   "Rural": 0,
    "0": 0, "1": 1, "2": 2, "3+": 3,
    "Good (1.0)": 1.0, "Bad (0.0)": 0.0
}

input_data = pd.DataFrame([[
    enc[gender],
    enc[married],
    enc[dependents],
    enc[education],
    enc[self_emp],
    income,
    co_income,
    loan_amt,
    loan_term,
    enc[credit],
    enc[property_a]
]], columns=feat_names)

# ── Prediction result ─────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Prediction Result")
    if predict_btn:
        result = model.predict(input_data)[0]
        prob   = model.predict_proba(input_data)[0]

        if result == 1:
            st.success("✅ Loan APPROVED")
            st.metric("Approval Confidence", f"{prob[1]*100:.1f}%")
        else:
            st.error("❌ Loan REJECTED")
            st.metric("Rejection Confidence", f"{prob[0]*100:.1f}%")

        # Input summary table
        st.markdown("**Applicant Summary**")
        summary = {
            "Gender": gender, "Married": married,
            "Dependents": dependents, "Education": education,
            "Self Employed": self_emp, "Applicant Income": f"₹{income:,}",
            "Co-applicant Income": f"₹{co_income:,}",
            "Loan Amount": f"₹{loan_amt}K", "Loan Term": f"{loan_term} months",
            "Credit History": credit, "Property Area": property_a
        }
        st.dataframe(pd.DataFrame(summary.items(), columns=["Field", "Value"]),
                     hide_index=True, use_container_width=True)
    else:
        st.info("Fill in the details on the left sidebar and click Predict.")

with col2:
    st.subheader("Model Accuracy Comparison")
    fig, ax = plt.subplots(figsize=(5, 3))
    models_list = ['Logistic\nRegression', 'Random\nForest', 'XGBoost']
    accuracies  = [78.86, 76.42, 73.17]
    colors      = ['#378ADD', '#3B6D11', '#854F0B']
    bars = ax.bar(models_list, accuracies, color=colors, width=0.5)
    ax.set_ylim(60, 90)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Model Comparison")
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f"{acc}%", ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.divider()

# ── EDA Dashboard ─────────────────────────────────────────
st.subheader("📊 Data Analysis Dashboard")
tab1, tab2, tab3, tab4 = st.tabs([
    "Loan Approval Distribution",
    "Income vs Approval",
    "Credit History Impact",
    "Feature Importance"
])

with tab1:
    st.image('plots/target.png', use_container_width=True)
    st.caption("Most loans in the dataset were approved. Class imbalance is visible.")

with tab2:
    st.image('plots/income_vs_status.png', use_container_width=True)
    st.caption("Higher income applicants tend to get approved, but outliers exist.")

with tab3:
    st.image('plots/credit_history.png', use_container_width=True)
    st.caption("Credit history is the single strongest predictor of loan approval.")

with tab4:
    st.image('plots/feature_importance.png', use_container_width=True)
    st.caption("Credit history and loan amount are the most important features in the Random Forest model.")

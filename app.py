# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle

# # --- Page Setup ---
# st.set_page_config(page_title="Loan Risk Assessment", layout="centered")

# @st.cache_resource
# def load_assets():
#     with open("model.pkl", "rb") as f:
#         model = pickle.load(f)
#     with open("scaler.pkl", "rb") as f:
#         scaler = pickle.load(f)
#     return model, scaler

# model, scaler = load_assets()

# # --- Mappings ---
# edu_map = {"Bachelor's": 0, "High School": 1, "Master's": 2, "PhD": 3}
# emp_map = {"Full-time": 0, "Part-time": 1, "Self-employed": 2, "Unemployed": 3}
# mar_map = {"Divorced": 0, "Married": 1, "Single": 2}
# pur_map = {"Auto": 0, "Business": 1, "Education": 2, "Home": 3, "Other": 4}
# bin_map = {"No": 0, "Yes": 1}

# st.title("🏦 Bank Loan Risk Assessment")
# st.write("Enter details to evaluate eligibility and risk levels.")
# st.markdown("---")

# # --- Inputs ---
# c1, c2 = st.columns(2)
# with c1:
#     age = st.number_input("Age", 18, 100, 30)
#     income = st.number_input("Annual Income ($)", value=50000)
#     loan_amt = st.number_input("Loan Amount ($)", value=15000)
#     c_score = st.slider("Credit Score", 300, 850, 650)
#     emp_mo = st.number_input("Months Employed", value=24)
#     c_lines = st.number_input("Open Credit Lines", value=3)

# with c2:
#     int_rate = st.number_input("Interest Rate (%)", value=10.0)
#     term = st.selectbox("Term (Months)", [12, 24, 36, 48, 60])
#     dti = st.slider("DTI Ratio", 0.0, 1.0, 0.3)
#     edu = st.selectbox("Education", list(edu_map.keys()))
#     emp = st.selectbox("Employment", list(emp_map.keys()))
#     marital = st.selectbox("Marital Status", list(mar_map.keys()))

# purpose = st.selectbox("Purpose", list(pur_map.keys()))
# mortgage = st.radio("Has Mortgage?", ["No", "Yes"], horizontal=True)
# dependents = st.radio("Has Dependents?", ["No", "Yes"], horizontal=True)
# cosigner = st.radio("Has Co-signer?", ["No", "Yes"], horizontal=True)

# if st.button("Calculate Eligibility", use_container_width=True):
#     # 1. Prepare Features
#     features = np.array([[
#         age, income, loan_amt, c_score, emp_mo, c_lines,
#         int_rate, term, dti, edu_map[edu], emp_map[emp],
#         mar_map[marital], bin_map[mortgage], bin_map[dependents],
#         pur_map[purpose], bin_map[cosigner]
#     ]])
    
#     # 2. Prediction
#     scaled_feat = scaler.transform(features)
#     prob = float(model.predict_proba(scaled_feat)[0][1]) # Float conversion fixed
    
#     # 3. Decision Logic (Balanced Threshold)
#     threshold = 0.40 
    
#     st.markdown("### Assessment Result")
#     if prob > threshold:
#         status = "NOT ELIGIBLE"
#         st.error(f"### ❌ {status}")
#         st.write(f"**Risk Analysis:** High ({prob:.2%})")
#         st.warning("Decision: Application rejected due to high risk.")
#     elif prob > 0.20:
#         status = "ELIGIBLE (WITH CAUTION)"
#         st.warning(f"### ⚠️ {status}")
#         st.write(f"**Risk Analysis:** Moderate ({prob:.2%})")
#         st.info("Decision: Approved, but consider higher interest or co-signer.")
#     else:
#         status = "ELIGIBLE"
#         st.success(f"### 🎉 {status}")
#         st.write(f"**Risk Analysis:** Low ({prob:.2%})")
#         st.info("Decision: Standard Approval.")
    
#     st.progress(prob)

import streamlit as st
import pandas as pd
import pickle

# =========================================
# 1. LOAD ARTIFACTS
# =========================================
model = pickle.load(open("model.pkl", "rb"))
preprocessor = pickle.load(open("preprocessor.pkl", "rb"))

st.set_page_config(page_title="Bank Loan Risk Assessment", layout="centered")

st.title("🏦 Bank Loan Risk Assessment")
st.write("Enter details to evaluate eligibility and risk levels.")

# =========================================
# 2. USER INPUTS
# =========================================
age = st.number_input("Age", min_value=18, max_value=100, value=20)
income = st.number_input("Annual Income ($)", min_value=0, value=120000)
loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=15000)
credit_score = st.slider("Credit Score", 300, 850, 300)
months_employed = st.number_input("Months Employed", min_value=0, value=72)
credit_lines = st.number_input("Open Credit Lines", min_value=0, value=1)
interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=6.5)
loan_term = st.number_input("Term (Months)", min_value=6, value=12)
dti = st.slider("DTI Ratio", 0.0, 1.0, 0.0)

education = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"])
employment = st.selectbox("Employment", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
loan_purpose = st.selectbox("Purpose", ["Auto", "Home", "Education", "Personal", "Business"])

has_mortgage = st.radio("Has Mortgage?", ["Yes", "No"])
has_dependents = st.radio("Has Dependents?", ["Yes", "No"])
has_cosigner = st.radio("Has Co-signer?", ["Yes", "No"])

# =========================================
# 3. POLICY RULE ENGINE (CRITICAL FIX)
# =========================================
def apply_policy_rules(df):
    if df["CreditScore"].iloc[0] < 500:
        return {
            "decision": "High Risk",
            "probability": 0.85,
            "reason": "Credit score below minimum policy threshold"
        }

    if df["Age"].iloc[0] < 21:
        return {
            "decision": "High Risk",
            "probability": 0.80,
            "reason": "Applicant below minimum age requirement"
        }

    if df["DTIRatio"].iloc[0] > 0.6:
        return {
            "decision": "High Risk",
            "probability": 0.78,
            "reason": "Debt-to-Income ratio exceeds policy limit"
        }

    return None  # No policy breach

# =========================================
# 4. PREDICTION
# =========================================
if st.button("Evaluate Risk"):

    input_data = pd.DataFrame([{
        "Age": age,
        "Income": income,
        "LoanAmount": loan_amount,
        "CreditScore": credit_score,
        "MonthsEmployed": months_employed,
        "NumCreditLines": credit_lines,
        "InterestRate": interest_rate,
        "LoanTerm": loan_term,
        "DTIRatio": dti,
        "Education": education,
        "EmploymentType": employment,
        "MaritalStatus": marital_status,
        "HasMortgage": has_mortgage,
        "HasDependents": has_dependents,
        "LoanPurpose": loan_purpose,
        "HasCoSigner": has_cosigner
    }])

    # -------- Policy Check First --------
    policy_result = apply_policy_rules(input_data)

    if policy_result:
        decision = policy_result["decision"]
        probability = policy_result["probability"]
        reason = policy_result["reason"]
        source = "Policy Rule"
    else:
        X_processed = preprocessor.transform(input_data)
        probability = model.predict_proba(X_processed)[0][1]
        decision = "High Risk" if probability >= 0.5 else "Low Risk"
        reason = "Model-driven assessment"
        source = "ML Model"

    # =========================================
    # 5. OUTPUT
    # =========================================
    st.subheader("📊 Risk Assessment Result")

    if decision == "High Risk":
        st.error("🔴 High Risk of Default")
    else:
        st.success("🟢 Low Risk")

    st.write(f"**Probability of Default:** {probability:.2%}")
    st.write(f"**Decision Source:** {source}")
    st.write(f"**Reason:** {reason}")

    if probability >= 0.7:
        st.write("**Risk Band:** Very High")
    elif probability >= 0.4:
        st.write("**Risk Band:** Medium")
    else:
        st.write("**Risk Band:** Low")

    st.caption("Model + Policy driven assessment | For decision support only")

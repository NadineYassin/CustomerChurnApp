import streamlit as st
import pandas as pd
import joblib

# --- Page Configuration ---
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# --- Load Model & Preprocessor ---
try:
    model = joblib.load("best_model_voting.pkl")
    preprocessing = joblib.load("preprocessing.pkl")
except FileNotFoundError:
    st.error("Model or preprocessor files not found. Please ensure they exist in the root directory.")
    st.stop()

# --- Tabs Layout ---
tab1, tab2 = st.tabs(["🔮 Single Prediction", "📂 Batch Prediction"])

# ------------------------- SINGLE PREDICTION -------------------------
with tab1:
    st.header("Predict Churn for a Single Customer")
    st.markdown("Fill in the customer details and click **Predict**.")

    # Inputs
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        tenure = st.number_input("Tenure (months)", value=12, min_value=0, max_value=72)
        monthly_charges = st.number_input("Monthly Charges", value=50.0, min_value=0.0, max_value=200.0)
    with col2:
        total_charges = st.number_input("Total Charges", value=1000.0, min_value=0.0, max_value=10000.0)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

    # Prediction button
    if st.button("🚀 Predict Churn"):
        input_data = pd.DataFrame({
            "gender": [gender],
            "SeniorCitizen": [senior],
            "tenure": [tenure],
            "MonthlyCharges": [monthly_charges],
            "TotalCharges": [total_charges],
            "Contract": [contract],
            "InternetService": [internet],
            "PaymentMethod": [payment]
        })

        try:
            processed_data = preprocessing.transform(input_data)
            pred = model.predict(processed_data)[0]
            prob = model.predict_proba(processed_data)[0][1]

            st.subheader("Prediction Result")
            st.write(f"**Churn Probability:** {prob:.2f}")

            if pred == 1:
                st.error("❌ This customer is likely to churn.")
            else:
                st.success("✅ This customer is likely to stay.")

        except Exception as e:
            st.error(f"Error during prediction: {e}")

# ------------------------- BATCH PREDICTION -------------------------
with tab2:
    st.header("Predict Churn for Multiple Customers")
    st.markdown("Upload a CSV file with customer data to predict churn for all customers.")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file:
        try:
            batch_data = pd.read_csv(uploaded_file)
            processed_batch = preprocessing.transform(batch_data)
            predictions = model.predict(processed_batch)
            batch_data["Churn_Prediction"] = predictions

            st.success("✅ Batch prediction completed!")
            st.dataframe(batch_data.head())

            csv_export = batch_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Predictions as CSV",
                data=csv_export,
                file_name="customer_churn_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Error processing the CSV file: {e}")

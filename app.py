import streamlit as st
import pickle
import pandas as pd

# Load the Preprocessor and Model
with open("artifacts/model.pkl", "rb") as file:
    model = pickle.load(file)

with open("artifacts/preprocessor.pkl", "rb") as file:
    preprocessor = pickle.load(file)

st.title("Churn Prediction")

# User Input Fields
credit_score = st.number_input("Enter the Credit Score", min_value=350, max_value=850)
geography = st.selectbox("Select the City", ['France', 'Germany', 'Spain'])
gender = st.selectbox("Select the Gender", ['Male', 'Female'])
age = st.number_input("Enter the Age", min_value=18, max_value=92)
tenure = st.number_input("Enter the Tenure", min_value=0, max_value=10)
num_of_products = st.number_input("Enter the Number of Products", min_value=1, max_value=4)
has_cr_card = st.selectbox("Have Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Enter the Estimated Salary", min_value=11, max_value=199992)
balance= st.number_input("Enter the Balance", min_value=0, max_value=250898)

# Create DataFrame (Wrap values in lists)
input_features = pd.DataFrame({
    "CreditScore": [credit_score],
    "Geography": [geography],
    "Gender": [gender],
    "Age": [age],
    "Tenure": [tenure],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary],
    "Balance":balance
})

# Predict on Button Click
if st.button("Predict Churn"):
    # Ensure Column Names Match
    print("Expected Columns:", preprocessor.feature_names_in_)
    print("Input Columns:", input_features.columns)

    # Transform Input Features
    preprocessed_features = preprocessor.transform(input_features)
    prediction = model.predict(preprocessed_features)

    def pred_output(pred):
        if prediction<0.5:
            return 0
        else:
            return 1
        
    prediction=pred_output(prediction)

    # Display Result
    st.write("Customer will Leave (chrun):", "Yes" if prediction == 1 else "No")
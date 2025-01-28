import streamlit as st
import joblib
import numpy as np

# Load the trained SVM model
try:
    model = joblib.load('svm_model.pkl')  # Path to the trained model file
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Set up the app title and description
st.title("SVM Model Prediction App")
st.write("This app predicts whether a user will make a purchase based on their details.")

# Collect user input in a form
st.header("Input User Details")
with st.form(key="input_form"):
    # Input fields
    gender = st.selectbox("Gender", ["Male", "Female"])  # Dropdown for gender
    age = st.number_input("Age", min_value=1, max_value=100, value=25, step=1)  # Numeric input for age
    salary = st.number_input("Estimated Salary (in $)", min_value=1000, max_value=1000000, value=50000, step=1000)  # Numeric input for salary

    # Submit button
    submit_button = st.form_submit_button(label="Predict")

# Perform prediction when the submit button is clicked
if submit_button:
    # Validate inputs
    if age <= 0 or salary <= 0:
        st.warning("Please enter valid values for age and salary.")
    else:
        # Encoding gender: Male -> 0, Female -> 1
        gender_encoded = 0 if gender == "Male" else 1
        user_input = np.array([[gender_encoded, age, salary]])

        # Get prediction from the model
        try:
            prediction = model.predict(user_input)
            result = "Purchased" if prediction[0] == 1 else "Not Purchased"
            st.success(f"The prediction is: {result}")
        except Exception as e:
            st.error(f"Prediction error: {e}")

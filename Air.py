import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the trained model and scaler
model = joblib.load("final_model.joblib")
scaler = joblib.load("full_pipeline.joblib")  # Assuming scaler was saved separately

# Define the 16 selected features for RUL prediction (using original feature names)
selected_features = [
    "op_setting1", "op_setting2", "s2", "s3", "s4", "s7", "s8",
    "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"
]

# Sidebar for file upload and selection of input mode
st.sidebar.title("Aircraft RUL Prediction App")
upload_toggle = st.sidebar.radio("Choose Input Method:", ("Upload Test Data", "Manual Input"))
uploaded_file = st.sidebar.file_uploader("Upload Test Data", type="txt") if upload_toggle == "Upload Test Data" else None
rul_file = st.sidebar.file_uploader("Upload Actual RUL Data (for evaluation)", type="txt")

# Function to display historical performance metrics
def display_historical_metrics():
    historical_rmse = 25.70  # Replace with actual test set RMSE
    historical_mae = 18.92  # Replace with actual test set MAE
    historical_r2 = 0.615  # Replace with actual test set R² score
    st.write("### Model Expected Performance (Based on Validation Data)")
    st.write(f"**Expected Root Mean Squared Error (RMSE):** {historical_rmse:.2f}")
    st.write(f"**Expected Mean Absolute Error (MAE):** {historical_mae:.2f}")
    st.write(f"**Expected R² Score:** {historical_r2:.2f}")
    st.write("*Note: These metrics represent the model's expected accuracy based on historical data.*")

# Function to process uploaded test data
def parse_test_data(uploaded_file):
    # Load file and drop the last two columns if they exist
    df_test_data = pd.read_csv(uploaded_file, sep=" ", header=None)
    if df_test_data.shape[1] > 26:
        df_test_data = df_test_data.iloc[:, :26]  # Keep only first 26 columns
    
    # Rename columns to match the expected feature names
    df_test_data.columns = ["cycle", "op_setting1", "op_setting2", "op_setting3", "s1", "s2", "s3", "s4",
                            "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15",
                            "s16", "s17", "s18", "s19", "s20", "s21", "RUL"]
    # Filter out the selected features
    df_test_features = df_test_data[selected_features]
    return df_test_features

# Handling file upload mode
if upload_toggle == "Upload Test Data" and uploaded_file is not None:
    try:
        X_test = parse_test_data(uploaded_file)
        st.write("### Processed Test Data (Preview)")
        st.write(X_test.head())

        # Scale the features
        X_test_scaled = scaler.transform(X_test)

        # Predict RUL for each aircraft
        predictions = model.predict(X_test_scaled)
        X_test["Predicted_RUL"] = predictions

        st.write("### Predicted RUL for Each Aircraft")
        st.write(X_test[["Predicted_RUL"]])

        # Check for RUL file for evaluation
        if rul_file is not None:
            y_true = pd.read_csv(rul_file, header=None, names=["Actual_RUL"])
            if len(y_true) == len(predictions):
                # Display comparison and evaluation metrics
                comparison_df = pd.DataFrame({
                    "Actual RUL": y_true["Actual_RUL"],
                    "Predicted RUL": predictions
                })
                st.write("### Comparison of Predicted and Actual RUL")
                st.write(comparison_df)

                rmse = np.sqrt(mean_squared_error(y_true, predictions))
                mae = mean_absolute_error(y_true, predictions)
                r2 = r2_score(y_true, predictions)

                st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
                st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
                st.write(f"**R² Score:** {r2:.2f}")
            else:
                st.error("Length mismatch between Actual RUL data and predictions.")
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")

# Manual input mode
elif upload_toggle == "Manual Input":
    st.write("### Enter Sensor Readings Manually")
    manual_input = {}
    for feature in selected_features:
        manual_input[feature] = st.number_input(f"Enter {feature}", min_value=0.0, step=0.01)

    if st.button("Predict RUL"):
        # Create DataFrame from manual input
        input_df = pd.DataFrame([manual_input])

        # Scale the input as per training pipeline
        input_scaled = scaler.transform(input_df)

        # Predict RUL
        predicted_rul = model.predict(input_scaled)
        
        st.write(f"**Predicted Remaining Useful Life (RUL): {predicted_rul[0]:.2f} cycles**")

        # Display historical model performance
        display_historical_metrics()

# Initial message if no file or inputs are provided
if upload_toggle == "Upload Test Data" and uploaded_file is None:
    st.write("Please upload the test data file in the sidebar to proceed.")

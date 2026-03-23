import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the trained model and preprocessing pipeline
model = joblib.load("final_model.joblib")
pipeline = joblib.load("full_pipeline.joblib")

# 16 selected features for RUL prediction
FEATURES = [
    "op_setting1", "op_setting2", "s2", "s3", "s4", "s7", "s8",
    "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"
]

# All C-MAPSS columns: id, cycle, 3 op_settings, 21 sensors
ALL_COLUMNS = [
    "id", "cycle", "op_setting1", "op_setting2", "op_setting3",
    "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10",
    "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21"
]

st.set_page_config(page_title="Aircraft RUL Prediction", layout="wide")
st.title("Aircraft RUL Prediction App")

# --- Sidebar ---
st.sidebar.header("Configuration")
input_mode = st.sidebar.radio("Input Method", ("Upload Test Data", "Manual Input"))
rul_file = st.sidebar.file_uploader("Upload Actual RUL (optional, for evaluation)", type="txt")


def parse_test_data(uploaded_file):
    """Parse space-delimited C-MAPSS test data."""
    df = pd.read_csv(uploaded_file, sep=r"\s+", header=None)
    # Drop trailing NaN columns from space-delimited format
    df = df.dropna(axis=1, how="all")
    df.columns = ALL_COLUMNS[:df.shape[1]]
    return df


def evaluate_predictions(predictions, rul_file):
    """Show evaluation metrics if actual RUL data is provided."""
    if rul_file is None:
        return
    y_true = pd.read_csv(rul_file, header=None, names=["Actual_RUL"])
    if len(y_true) != len(predictions):
        st.error(f"Length mismatch: {len(predictions)} predictions vs {len(y_true)} actual RUL values.")
        return

    comparison = pd.DataFrame({
        "Predicted RUL": predictions,
        "Actual RUL": y_true["Actual_RUL"],
    })
    comparison["Error"] = comparison["Predicted RUL"] - comparison["Actual RUL"]

    rmse = np.sqrt(mean_squared_error(y_true, predictions))
    mae = mean_absolute_error(y_true, predictions)
    r2 = r2_score(y_true, predictions)

    st.write("### Evaluation Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", f"{rmse:.2f}")
    col2.metric("MAE", f"{mae:.2f}")
    col3.metric("R² Score", f"{r2:.3f}")

    st.write("### Predicted vs Actual RUL")
    st.dataframe(comparison, use_container_width=True)

    st.write("### Error Distribution")
    st.bar_chart(comparison["Error"])


# --- Upload Mode ---
if input_mode == "Upload Test Data":
    uploaded_file = st.file_uploader("Upload Test Data File", type="txt")
    if uploaded_file is not None:
        try:
            df = parse_test_data(uploaded_file)
            X_test = df[FEATURES]

            st.write("### Input Data Preview")
            st.dataframe(X_test.head(10), use_container_width=True)

            X_scaled = pipeline.transform(X_test)
            predictions = model.predict(X_scaled)

            results = df[["id", "cycle"]].copy() if "id" in df.columns else pd.DataFrame()
            results["Predicted_RUL"] = predictions

            st.write(f"### Predictions ({len(predictions)} engines)")
            st.dataframe(results, use_container_width=True)

            evaluate_predictions(predictions, rul_file)

        except Exception:
            st.error("Failed to process file. Ensure it matches the C-MAPSS space-delimited format.")
    else:
        st.info("Upload a test data file to get started.")

# --- Manual Mode ---
elif input_mode == "Manual Input":
    st.write("### Enter Sensor Readings")
    cols = st.columns(4)
    manual_input = {}
    for i, feature in enumerate(FEATURES):
        with cols[i % 4]:
            manual_input[feature] = st.number_input(feature, step=0.01, format="%.4f")

    if st.button("Predict RUL", type="primary"):
        input_df = pd.DataFrame([manual_input])
        input_scaled = pipeline.transform(input_df)
        predicted_rul = model.predict(input_scaled)[0]

        st.write("---")
        st.metric("Predicted Remaining Useful Life", f"{predicted_rul:.1f} cycles")

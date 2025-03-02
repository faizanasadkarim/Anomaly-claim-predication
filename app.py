import streamlit as st
import joblib
import pandas as pd
import numpy as np

CREDENTIALS = {
    "admin": "admin",
}

def perform_login():
    """Display the login form and update session state on success."""
    st.title("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if username in CREDENTIALS and CREDENTIALS[username] == password:
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.success("Login successful!")
            st.rerun()  # Rerun to refresh the interface
        else:
            st.error("Invalid username or password")

def load_resources():
    """Load the ML model and encoders."""
    model = joblib.load("model.pkl")
    drg_encoder = joblib.load("drg_enc.pkl")
    hrr_encoder = joblib.load("region_enc.pkl")
    return model, drg_encoder, hrr_encoder

def render_prediction_interface(model, drg_encoder, hrr_encoder):
    """Render the anomaly detection interface for authenticated users."""
    st.success(f"Welcome, {st.session_state['username']}!")
    st.header("Medicare Claim Anomaly Detection")
    
    # Categorical inputs with dropdowns
    procedure = st.selectbox("Procedure_Code", options=drg_encoder.classes_)
    region = st.selectbox("Region", options=hrr_encoder.classes_)
    
    # Numeric inputs
    total_discharges = st.number_input("Total Discharges", min_value=1)
    avg_charge = st.number_input("Average Covered Charges", min_value=0.0)
    avg_payment = st.number_input("Average Payment", min_value=0.0)
    avg_medicare_payment = st.number_input("Average Medicare Payment", min_value=0.0)
    
    # Encode selected categorical values
    proc_encoded = drg_encoder.transform([procedure])[0]
    region_encoded = hrr_encoder.transform([region])[0]
    
    # Prepare input data for prediction
    input_data = pd.DataFrame({
        "Procedure_Code_Encoded": [proc_encoded],
        "Region_Encoded": [region_encoded],
        "Volume": [total_discharges],
        "Avg_Charge": [avg_charge],
        "Avg_Payment": [avg_payment],
        "Average_Medicare_Payments": [avg_medicare_payment]
    })
    
    if st.button("Predict Anomaly"):
        # Compute the decision function score.
        # Lower values indicate more anomalous observations.
        raw_score = model.decision_function(input_data)[0]
        
        # Invert the score so that higher values indicate higher anomaly likelihood.
        inverted_score = -raw_score
        
        # Map the inverted score to a probability between 0 and 1 using a sigmoid function.
        anomaly_proba = 1 / (1 + np.exp(-inverted_score))
        
        st.write(f"**Anomaly Probability:** {anomaly_proba:.2%}")
        st.progress(float(anomaly_proba))
        
        # Provide feedback based on probability thresholds
        if anomaly_proba > 0.75:
            st.error("High probability of anomaly - recommend investigation")
        elif anomaly_proba > 0.5:
            st.warning("Moderate probability of anomaly - suggest review")
        else:
            st.success("Low probability of anomaly - appears normal")

def main():
    """Main function that checks authentication and renders appropriate UI."""
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    
    if not st.session_state["authenticated"]:
        perform_login()
    else:
        model, drg_encoder, hrr_encoder = load_resources()
        render_prediction_interface(model, drg_encoder, hrr_encoder)

if __name__ == "__main__":
    main()

import streamlit as st

# Hardcoded credentials (for demonstration purposes)
USER_CREDENTIALS = {
    "admin": "admin",
    "user1": "securepass"
}

def login():
    """Handles user authentication."""
    st.title("Login Page")
    
    # Username and password input fields
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.success("Login successful!")
            st.rerun()  # Updated from st.experimental_rerun()
        else:
            st.error("Invalid username or password")

def main():
    """Main function to manage authentication."""
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        login()
    else:
        st.success(f"Welcome, {st.session_state['username']}!")
        
        import joblib
        import pandas as pd
        
        # Load model and encoders
        model = joblib.load('model.pkl')
        le_drg = joblib.load('drg_enc.pkl')
        le_hrr = joblib.load('region_enc.pkl')
        
        st.title('Medicare Claim Anomaly Detection')
        
        # Input fields
        # Dropdowns with original labels
        drg = st.selectbox('Procedure_Code', options=le_drg.classes_)
        hrr = st.selectbox('Region', options=le_hrr.classes_)
        
        # Encode selected values
        drg_encoded = le_drg.transform([drg])[0]
        hrr_encoded = le_hrr.transform([hrr])[0]
        
        discharges = st.number_input('Total Discharges', min_value=1)
        charge = st.number_input('Average Covered Charges', min_value=0.0)
        payment = st.number_input('Average Payment', min_value=0.0)
        medicare = st.number_input('Average Medicare Payment', min_value=0.0)
        
        
        
        
        
        # Preprocess inputs
        input_df = pd.DataFrame({
            'Procedure_Code_Encoded': [le_drg.transform([drg])[0]],
            'Region_Encoded': [le_hrr.transform([hrr])[0]],
            'Volume': [discharges],
            'Avg_Charge':[charge],
            'Avg_Payment':[payment],
            'Average_Medicare_Payments':[medicare]
        
        })
        
        
        if st.button('Predict Anomaly'):
            proba = model.predict_proba(input_df)[0][1]
            st.write(f"**Anomaly Probability:** {proba:.2%}")
            st.progress(float(proba))  # Cast np.float32 to native float
            
            if proba > 0.75:
                st.error("High probability of anomaly - recommend investigation")
            elif proba > 0.5:
                st.warning("Moderate probability of anomaly - suggest review")
            else:
                st.success("Low probability of anomaly - appears normal")

if __name__ == "__main__":
    main()

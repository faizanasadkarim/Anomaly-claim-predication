import streamlit as st
import joblib
import pandas as pd

# Load model and encoders
model = joblib.load('model.pkl')
le_drg = joblib.load('drg_enc.pkl')
le_hrr = joblib.load('region_enc.pkl')
hrr_drg_means = pd.read_csv('region.csv') 

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

# Get reference charge
try:
    hrr_mean_charge = hrr_drg_means[
        (hrr_drg_means['Procedure_Code'] == drg) & 
        (hrr_drg_means['Region'] == hrr)
    ]['Avg_Charge'].values[0]
except IndexError:
    # Fallback to national average for DRG
    hrr_mean_charge = hrr_drg_means[
        hrr_drg_means['Procedure_Code'] == drg
    ]['Avg_Charge'].mean()

# Calculate charge ratio
charge_ratio = charge / hrr_mean_charge if hrr_mean_charge != 0 else 0



# Preprocess inputs
input_df = pd.DataFrame({
    'Procedure_Code_Encoded': [le_drg.transform([drg])[0]],
    'Region_Encoded': [le_hrr.transform([hrr])[0]],
    'Charge_Ratio': [charge_ratio],
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


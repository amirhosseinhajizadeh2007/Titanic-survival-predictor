import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Dashboard configuration
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

# 1. Main dashboard header
st.title("🚢 Titanic Survival Predictor")
st.markdown("---")
st.write("Enter the hypothetical passenger's information to let the AI model (Voting Classifier) predict their survival.")

# Helper function to load model and scaler with caching for better performance
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('titanic_voting_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("⚠️ Model files (.pkl) not found! Ensure you have saved them from the notebook first.")
        return None, None

voting_model, scaler = load_artifacts()

if voting_model is not None:
    # 2. User input section (Interactive form)
    st.subheader("📋 Passenger Details:")
    
    # Column layout for better UI
    col1, col2 = st.columns(2)
    
    with col1:
        pclass = st.selectbox("Passenger Class (Pclass):", options=[1, 2, 3], index=2, 
                             help="Class 1: Upper/Rich, Class 3: Lower/Poor")
        sex = st.selectbox("Gender (Sex):", options=["Male", "Female"], index=0)
        age = st.slider("Age:", min_value=1, max_value=100, value=28)
        fare = st.number_input("Ticket Fare ($):", min_value=0.0, value=32.0, step=5.0)

    with col2:
        sibsp = st.number_input("Number of Siblings/Spouses Aboard (SibSp):", min_value=0, value=0, step=1)
        parch = st.number_input("Number of Parents/Children Aboard (Parch):", min_value=0, value=0, step=1)
        embarked_choice = st.selectbox("Port of Embarkation (Embarked):", options=["Southampton", "Cherbourg", "Queenstown"], index=0)

    # 3. Input preprocessing (Matches notebook logic exactly)
    sex_encoded = 0 if sex == "Male" else 1
    
    embarked_map = {"Southampton": 0, "Cherbourg": 1, "Queenstown": 2}
    embarked_encoded = embarked_map[embarked_choice]
    
    family_size = sibsp + parch + 1
    is_alone = 1 if family_size == 1 else 0

    st.markdown("---")
    
    # 4. Predict button click
    if st.button("🔮 Predict Survival Chances", use_container_width=True):
        
        # Fix: Standardization is applied only to the 2 numerical columns (Age and Fare)
        age_fare = np.array([[age, fare]])
        scaled_age_fare = scaler.transform(age_fare)
        
        scaled_age = scaled_age_fare[0][0]
        scaled_fare = scaled_age_fare[0][1]
        
        # Creating the final array with scaled features in the order expected by the model
        final_features = np.array([[
            pclass, sex_encoded, scaled_age, sibsp, parch, 
            scaled_fare, embarked_encoded, family_size, is_alone
        ]])
        
        # Make prediction
        prediction = voting_model.predict(final_features)
        prediction_proba = voting_model.predict_proba(final_features)[0][1] if hasattr(voting_model, "predict_proba") else None

        # Display output with colored effects
        st.subheader("🎯 Prediction Result:")
        if prediction[0] == 1:
            st.success("🎉 The passenger survives! (Survived)")
            if prediction_proba is not None:
                st.info(f"📊 Probability of Survival: {prediction_proba * 100:.2f}%")
        else:
            st.error("💀 The passenger does not survive! (Not Survived)")
            if prediction_proba is not None:
                st.info(f"📊 Probability of Death: {(1 - prediction_proba) * 100:.2f}%")


                
# --- About Developer (Sidebar) ---
st.sidebar.title("👨‍💻 About Developer")
st.sidebar.markdown("**AmirHossein HajiZadeh**")
st.sidebar.markdown("📧 [amirhosseinhajizadeh.0007@gmail.com](mailto:amirhosseinhajizadeh.0007@gmail.com)")
st.sidebar.markdown("---")

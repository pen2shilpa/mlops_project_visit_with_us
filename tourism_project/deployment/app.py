import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file or in artifacts)
model_path = os.path.join(os.path.dirname(__file__), "../artifacts/best_model.joblib")
if not os.path.exists(model_path):
    model_path = os.path.join(os.path.dirname(__file__), "best_model.joblib")

model = joblib.load(model_path)

# Streamlit UI
st.title("Tourism Wellness package purchase prediction Model")
st.write("""
This application predicts if a customer will purchase the newly introduced Wellness Tourism Package or not based on the customer details.
""")

# Get customer details for prediction

gender = st.selectbox("Gender", ["Female", "Male"])
age = st.number_input("Age", min_value=1, max_value=100, value=25)
type_of_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
marital_status = st.selectbox("Marital Status", ["Unmarried", "Married", "Divorced"])

occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
designation = st.selectbox("Designation", ["Manager", "Executive", "Associate", "Senior Manager", "AVP"])
monthly_income = st.number_input("Monthly Income", min_value=1, max_value=10000000, value=20000)
city_tier = st.selectbox("City Tier", [1, 2, 3])

preferred_property_star = st.number_input("Preferred Property Star", min_value=1, max_value=5, value=3)
number_of_trips = st.number_input("Number of Trips", min_value=1, max_value=100, value=1)

number_of_person_visiting = st.number_input("Number of Person Visiting", min_value=1, max_value=50, value=2)
number_of_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=0)

# Streamlit UI for passport
passport_selection = st.selectbox("Passport", ["Yes", "No"])
# Map selection string to binary
passport_val = 1 if passport_selection == "Yes" else 0

own_car_selection = st.selectbox("Own Car", ["Yes", "No"])
own_car_val = 1 if own_car_selection == "Yes" else 0

product_pitched = st.selectbox("Product Pitched", ["Deluxe", "Basic", "Standard", "Super Deluxe", "King"])
duration_of_pitch = st.number_input("Duration of Pitch", min_value=1, max_value=100, value=5)
pitch_satisfaction_score = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
number_of_followups = st.number_input("Number of Followups", min_value=1, max_value=50, value=1)

# Assemble input into DataFrame matching training schema exactly
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': type_of_contact,
    'CityTier': city_tier,
    'DurationOfPitch': duration_of_pitch,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': number_of_person_visiting,
    'NumberOfFollowups': number_of_followups,
    'ProductPitched': product_pitched,
    'PreferredPropertyStar': preferred_property_star,
    'MaritalStatus': marital_status,
    'NumberOfTrips': number_of_trips,
    'Passport': passport_val,
    'PitchSatisfactionScore': pitch_satisfaction_score,
    'OwnCar': own_car_val,
    'NumberOfChildrenVisiting': number_of_children_visiting,
    'Designation': designation,
    'MonthlyIncome': monthly_income
}])

# Predict button
if st.button("Predict Wellness Tour Package Purchase"):
    prediction = model.predict(input_data)[0]
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success("**Yes**, The customer is likely to buy this tour package.")
    else:
        st.error("**No**, The customer is unlikely to buy this tour package.")

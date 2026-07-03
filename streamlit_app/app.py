import streamlit as st
import pandas as pd
import joblib

# Load trained model
import os
import streamlit as st
import pandas as pd
import joblib

# Load trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "..", "model", "model.pkl")
model = joblib.load(model_path)

# Page Configuration
st.set_page_config(page_title="House Price Prediction")

st.title("House Price Prediction System")
st.write("Enter the house details below:")

# Inputs
area = st.number_input("Area", min_value=500, value=5000)
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
stories = st.number_input("Stories", min_value=1, max_value=5, value=2)
parking = st.number_input("Parking", min_value=0, max_value=5, value=1)

mainroad = st.selectbox("Main Road", ["No", "Yes"])
guestroom = st.selectbox("Guest Room", ["No", "Yes"])
basement = st.selectbox("Basement", ["No", "Yes"])
hotwaterheating = st.selectbox("Hot Water Heating", ["No", "Yes"])
airconditioning = st.selectbox("Air Conditioning", ["No", "Yes"])
prefarea = st.selectbox("Preferred Area", ["No", "Yes"])

furnishing = st.selectbox(
    "Furnishing Status",
    ["Furnished", "Semi-Furnished", "Unfurnished"]
)

# Create DataFrame
input_data = pd.DataFrame({
    "area": [area],
    "bedrooms": [bedrooms],
    "bathrooms": [bathrooms],
    "stories": [stories],
    "parking": [parking],
    "mainroad_yes": [1 if mainroad == "Yes" else 0],
    "guestroom_yes": [1 if guestroom == "Yes" else 0],
    "basement_yes": [1 if basement == "Yes" else 0],
    "hotwaterheating_yes": [1 if hotwaterheating == "Yes" else 0],
    "airconditioning_yes": [1 if airconditioning == "Yes" else 0],
    "prefarea_yes": [1 if prefarea == "Yes" else 0],
    "furnishingstatus_semi-furnished": [1 if furnishing == "Semi-Furnished" else 0],
    "furnishingstatus_unfurnished": [1 if furnishing == "Unfurnished" else 0]
})

# Prediction
if st.button("Predict House Price"):
    prediction = model.predict(input_data)
    st.success(f"Predicted House Price: Rs. {prediction[0]:,.2f}")
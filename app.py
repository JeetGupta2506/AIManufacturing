import streamlit as st
import joblib
import pandas as pd

# Load trained models and preprocessor
demand_model = joblib.load("demand_forecast_model.pkl")
discount_model = joblib.load("discount_prediction_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")  # Load the preprocessor

# Streamlit UI
st.title("📈 Dynamic Pricing Prediction")
st.write("Enter product details and get demand & discount predictions.")

# Input Fields
store_id = st.text_input("Store ID")
product_id = st.text_input("Product ID")
inventory = st.number_input("Inventory Level", min_value=0)
units_sold = st.number_input("Units Sold", min_value=0)
units_ordered = st.number_input("Units Ordered", min_value=0)
price = st.number_input("Price", min_value=0.0, format="%.2f")
weather = st.selectbox("Weather Condition", ["Sunny", "Rainy", "Snowy", "Cloudy"])
competitor_price = st.number_input("Competitor Pricing", min_value=0.0, format="%.2f")
seasonality = st.selectbox("Seasonality", ["Spring", "Summer", "Autumn", "Winter"])

# Predict Button
if st.button("Predict"):
    # Check if all inputs are zero
    if all(value == 0 for value in [inventory, units_sold, units_ordered, price, competitor_price]):
        st.warning("All input fields are set to zero. Please enter realistic values for accurate predictions.")
    else:
        # Convert input to DataFrame
        data = pd.DataFrame([{
            "Store ID": store_id,
            "Product ID": product_id,
            "Inventory Level": inventory,
            "Units Sold": units_sold,
            "Units Ordered": units_ordered,
            "Price": price,
            "Weather Condition": weather,
            "Competitor Pricing": competitor_price,
            "Seasonality": seasonality
        }])

        try:
            # Preprocess the data using the pre-fitted preprocessor
            data_preprocessed = preprocessor.transform(data)

            # Make predictions
            predicted_demand = demand_model.predict(data_preprocessed)[0]
            predicted_discount = discount_model.predict(data_preprocessed)[0]

            # Display results
            st.success(f"📊 Predicted Demand: {predicted_demand:.2f}")
            st.success(f"💰 Predicted Discount: {predicted_discount:.2f}%")
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")

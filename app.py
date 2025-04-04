import streamlit as st
import joblib
import pandas as pd

# Load trained models and preprocessor
demand_model = joblib.load("demand_forecast_model.pkl")
discount_model = joblib.load("discount_prediction_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Streamlit UI Design
st.set_page_config(page_title="Dynamic Pricing Prediction", page_icon="📈", layout="wide")

# Custom CSS for Styling
st.markdown("""
    <style>
        .big-font { font-size:25px !important; font-weight: bold; color: #FF4B4B; }
        .success-box { 
            border-radius: 10px;
            background-color: #dff0d8;
            padding: 10px;
            font-size: 20px;
            text-align: center;
        }
        .warning-box {
            border-radius: 10px;
            background-color: #fcf8e3;
            padding: 10px;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)
st.markdown("""
    <style>
        /* Reduce space above "Enter Product Details" */
        h2 {
            margin-top: -40px !important;
        }
    </style>
""", unsafe_allow_html=True)


# Header Section
st.title("📈 Dynamic Pricing Prediction")
st.markdown("<p style='font-size:25px; font-weight:bold; color:#007BFF;'>✅ Optimize Pricing with AI-powered Predictions</p>", unsafe_allow_html=True)
st.write("Enter product details below to get **demand & discount** predictions.")

# Load the dataset 
df_prices = pd.read_csv("retail_store_inventory.csv") 

# Sidebar for User Input
with st.sidebar:
    st.header("🔍 Enter Product Details")
    store_id = st.text_input("🏪 Store ID", value="S001")
    product_id = st.text_input("📦 Product ID", value="P0001")
    inventory = st.number_input("📊 Inventory Level", min_value=0)
    units_sold = st.number_input("📉 Units Sold", min_value=0)
    units_ordered = st.number_input("📦 Units Ordered", min_value=0)
    weather = st.selectbox("🌦️ Weather Condition", ["Sunny", "Rainy", "Snowy", "Cloudy"])
    competitor_price = st.number_input("🏷️ Competitor Pricing", min_value=0)
    seasonality = st.selectbox("📅 Seasonality", ["Spring", "Summer", "Autumn", "Winter"])

# Fetch price from dataset based on store_id and product_id
    product_row = df_prices[(df_prices["Store ID"] == store_id) & (df_prices["Product ID"] == product_id)]
    
    price = product_row["Price"].values[0] if not product_row.empty else 0.0
    base_cost = product_row["Base Cost"].values[0] if not product_row.empty else 0.0

# Predict Button
st.markdown("---")  # Adds a separator for better UI
if st.button("🚀 Predict", help="Click to get demand & discount predictions", use_container_width=True):
    if all(value == 0 for value in [inventory, units_sold, units_ordered, price, competitor_price]):
        st.markdown(f"""<div style="border-radius: 10px; background-color: #fff3cd; padding: 10px; text-align: center;">
            <b style="color: #856404; font-size: 20px;">⚠️ Warning: All input fields are set to zero. Please enter realistic values for accurate predictions.</b></div> """, unsafe_allow_html=True)
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
            # Preprocess input data
            data_preprocessed = preprocessor.transform(data)

            # Predictions
            predicted_demand = demand_model.predict(data_preprocessed)[0]
            predicted_discount = discount_model.predict(data_preprocessed)[0]
            discounted_price = price * (1 - predicted_discount / 100)
            Profit = discounted_price - base_cost

            # Display Results in Columns
            # First Row with Proper Spacing
            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
              st.markdown(f"""<div style="border-radius: 10px; background-color: #dff0d8; padding: 15px; margin: 5px; text-align: center;">
             <b style="color: #155724; font-size: 22px;">💰 ORIGINAL MRP: {price} INR</b> </div> """, unsafe_allow_html=True)

            with col2:
             st.markdown(f""" <div style="border-radius: 10px; background-color: #dff0d8; padding: 15px; margin: 5px; text-align: center;">
             <b style="color: #155724; font-size: 22px;">📊 Predicted Demand: {predicted_demand:.2f}</b></div>""", unsafe_allow_html=True)

            with col3:
             st.markdown(f"""<div style="border-radius: 10px; background-color: #dff0d8; padding: 15px; margin: 5px; text-align: center;">
             <b style="color: #155724; font-size: 22px;">📉 Predicted Discount: {predicted_discount:.2f}%</b> </div> """, unsafe_allow_html=True)

# Add some spacing between rows
            st.markdown("<br>", unsafe_allow_html=True)

# Second Row with Proper Spacing
            col4, col5 = st.columns([1, 1])

            with col4:
             st.markdown(f"""<div style="border-radius: 10px; background-color: #dff0d8; padding: 15px; margin: 5px; text-align: center;">
             <b style="color: #155724; font-size: 22px;">🏷️ Discounted Price: {discounted_price:.2f} INR</b> </div> """, unsafe_allow_html=True)

            with col5:
             st.markdown(f"""<div style="border-radius: 10px; background-color: #dff0d8; padding: 15px; margin: 5px; text-align: center;">
             <b style="color: #155724; font-size: 22px;">💵 Profit: {Profit:.2f} INR</b> </div> """, unsafe_allow_html=True)

        except Exception as e:
            st.markdown(f"""<div style="border-radius: 10px; background-color: #f8d7da; padding: 10px; text-align: center;">
            <b style="color: #721c24; font-size: 20px;">❌ Error: {str(e)}</b></div> """, unsafe_allow_html=True)
# Footer
st.markdown("---")
st.write("🔹 **Powered by AI & Machine Learning**")
st.write("🔹 **Made by Driti Rathod , Jeet Gupta and Kalp Sheladiya**")

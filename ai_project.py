# -*- coding: utf-8 -*-
"""AI_Project.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/github/JeetGupta2506/AIManufacturing/blob/main/AI_Project.ipynb
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('retail_store_inventory.csv')

df.head(10)

df.drop(columns=['Category','Region'], inplace=True)

df.drop(columns=['Holiday/Promotion'], inplace=True)

df.isnull().sum()

df.dropna(inplace=True)

df.head()

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Feature and target separation
X = df.drop(['Discount', 'Demand Forecast'], axis=1)
y_discount = df['Discount']
y_demand = df['Demand Forecast']

# Identify categorical and numerical columns
categorical_cols = ['Store ID', 'Product ID', 'Weather Condition', 'Seasonality']
numerical_cols = ['Inventory Level', 'Units Sold', 'Units Ordered', 'Price', 'Competitor Pricing']

# Define preprocessor (only fit once)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# Train-test split
X_train, X_test, y_train_d, y_test_d = train_test_split(X, y_discount, test_size=0.2, random_state=42)
_, _, y_train_f, y_test_f = train_test_split(X, y_demand, test_size=0.2, random_state=42)

# Fit preprocessor on the entire training data
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Use the same transformed data for both targets
X_train_d = X_train_transformed
X_train_f = X_train_transformed
X_test_d = X_test_transformed
X_test_f = X_test_transformed

X_train_d.toarray()

X_test_d.shape

from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Function to evaluate models
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    accuracy = 100 - mape
    print(f"{model_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}, Accuracy: {accuracy:.2f}%")

# Base models for stacking
base_models = [
    ('lgbm', LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, subsample=0.8)),
    ('xgb', XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, subsample=0.8)),
    ('rf', RandomForestRegressor(n_estimators=200, max_depth=10)),
    ('gb', GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5))
]

# Meta-model (Linear Regression for final prediction)
meta_model = LinearRegression()

# Stacking Regressors for Demand and Discount
demand_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)
discount_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)

# Train models
demand_model.fit(X_train_f, y_train_f)
discount_model.fit(X_train_d, y_train_d)

# Make predictions
y_demand_pred = demand_model.predict(X_test_f)
y_discount_pred = discount_model.predict(X_test_d)

# Evaluate models
evaluate_model(y_test_f, y_demand_pred, 'Ensemble Demand Forecast Model')
evaluate_model(y_test_d, y_discount_pred, 'Ensemble Discount Model')

import joblib

# Save the trained models
joblib.dump(demand_model, 'demand_forecast_model.pkl')
joblib.dump(discount_model, 'discount_prediction_model.pkl')

print("Models saved successfully!")

import joblib

# Load trained models
demand_model = joblib.load('demand_forecast_model.pkl')
discount_model = joblib.load('discount_prediction_model.pkl')

import pandas as pd

data = pd.DataFrame([
    {
        'Date': '01-01-2023',
        'Store ID': 'S001',
        'Product ID': 'P0001',
        'Inventory Level': 231.0,
        'Units Sold': 127.0,
        'Units Ordered': 55.0,
        'Price': 33.50,
        'Weather Condition': 'Rainy',

        'Competitor Pricing': 29.69,
        'Seasonality': 'Autumn'
    },
    {
        'Date': '01-01-2023',
        'Store ID': 'S001',
        'Product ID': 'P0002',
        'Inventory Level': 204.0,
        'Units Sold': 150.0,
        'Units Ordered': 66.0,
        'Price': 63.01,
        'Weather Condition': 'Sunny',

        'Competitor Pricing': 66.16,
        'Seasonality': 'Autumn'
    }
])

# Drop target columns if present
input_features = data.drop(columns=['Demand_Forecast', 'Discount'], errors='ignore')

print(input_features)

input_features=preprocessor.transform(input_features)
predicted_demand=demand_model.predict(input_features)
predicted_discount=discount_model.predict(input_features)

preprocessor.feature_names_in_


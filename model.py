# retail_demand_forecaster.py

import pandas as pd
import numpy as np
import lightgbm as lgb
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

st.set_page_config(layout="wide")
st.title("🛍️ Real-Time Retail Demand Forecaster")

@st.cache_data
def load_data():
    df = pd.read_csv("Retail_Dataset2.csv")
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=False, errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values(['Product_Code', 'Date'])

    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Weekday'] = df['Date'].dt.weekday

    df['Warehouse'] = df['Warehouse'].astype('category').cat.codes
    df['Product_Category'] = df['Product_Category'].astype('category').cat.codes
    df['Product_Code'] = df['Product_Code'].astype('category').cat.codes
    df['StateHoliday'] = df['StateHoliday'].astype('category').cat.codes

    df['lag_1'] = df.groupby('Product_Code')['Order_Demand'].shift(1)
    df['lag_7'] = df.groupby('Product_Code')['Order_Demand'].shift(7)
    df['lag_14'] = df.groupby('Product_Code')['Order_Demand'].shift(14)
    df['rolling_mean_7'] = df.groupby('Product_Code')['Order_Demand'].shift(1).rolling(window=7).mean().reset_index(0, drop=True)
    df['rolling_mean_30'] = df.groupby('Product_Code')['Order_Demand'].shift(1).rolling(window=30).mean().reset_index(0, drop=True)

    return df.dropna()

df = load_data()

product_codes = df['Product_Code'].unique()
selected_product = st.sidebar.selectbox("Select a Product Code:", product_codes)

product_df = df[df['Product_Code'] == selected_product]

features = ['Warehouse', 'Product_Category', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday',
            'Petrol_price', 'Day', 'Month', 'Weekday', 'lag_1', 'lag_7', 'lag_14',
            'rolling_mean_7', 'rolling_mean_30']

X = product_df[features]
y = product_df['Order_Demand']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = lgb.LGBMRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
st.metric("📉 RMSE", f"{np.sqrt(mse):,.2f}")

# Plot actual vs predicted
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(y_test.values, label='Actual', marker='o')
ax.plot(y_pred, label='Predicted', linestyle='--')
ax.set_title("Actual vs Predicted Demand")
ax.set_xlabel("Time")
ax.set_ylabel("Demand")
ax.legend()
st.pyplot(fig)

# Forecasting next 7 days
def forecast_next_7_days(df, model):
    future = df.sort_values("Date").iloc[-1:].copy()
    forecasts = []
    for _ in range(7):
        future['lag_1'] = future['Order_Demand']
        future['lag_7'] = df['Order_Demand'].iloc[-7]
        future['lag_14'] = df['Order_Demand'].iloc[-14]
        future['rolling_mean_7'] = df['Order_Demand'].iloc[-7:].mean()
        future['rolling_mean_30'] = df['Order_Demand'].iloc[-30:].mean()

        features_future = future[features]
        prediction = model.predict(features_future)[0]
        future['Order_Demand'] = prediction
        future['Date'] += pd.Timedelta(days=1)
        df = pd.concat([df, future], ignore_index=True)
        forecasts.append((future['Date'].values[0], prediction))

    return pd.DataFrame(forecasts, columns=["Date", "Forecasted_Demand"])

st.subheader("📆 Forecast for Next 7 Days")
forecast_df = forecast_next_7_days(product_df.copy(), model)
st.dataframe(forecast_df)

fig2, ax2 = plt.subplots()
ax2.plot(forecast_df['Date'], forecast_df['Forecasted_Demand'], marker='o')
ax2.set_title("7-Day Forecasted Demand")
ax2.set_ylabel("Forecasted Demand")
ax2.set_xlabel("Date")
st.pyplot(fig2)

st.caption("Built for the Gen AI Hackathon 🧠⚡")

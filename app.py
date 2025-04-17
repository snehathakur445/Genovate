# retail_demand_forecaster.py with Deep Learning (LSTM)

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

st.set_page_config(layout="wide")
st.title("🧠 Predictrix: Retail Demand Forecaster")

@st.cache_data
def load_data():
    df = pd.read_csv("Retail_Dataset2.csv")
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=False, errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values(['Product_Code', 'Date'])

    df['Warehouse'] = df['Warehouse'].astype('category').cat.codes
    df['Product_Category'] = df['Product_Category'].astype('category').cat.codes
    df['Product_Code'] = df['Product_Code'].astype('category').cat.codes
    df['StateHoliday'] = df['StateHoliday'].astype('category').cat.codes

    return df.dropna()

df = load_data()

product_codes = df['Product_Code'].unique()
selected_product = st.sidebar.selectbox("Select a Product Code:", product_codes)

product_df = df[df['Product_Code'] == selected_product]
product_df = product_df.groupby("Date")["Order_Demand"].sum().reset_index()
product_df = product_df.sort_values("Date")

# Scaling the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(product_df[['Order_Demand']])

# Create sequences for LSTM
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i])
    if len(X) == 0:
        return np.empty((0, window_size, 1)), np.empty((0,))
    X = np.array(X)
    return X.reshape((X.shape[0], X.shape[1], 1)), np.array(y)

window_size = 30
X, y = create_sequences(scaled_data, window_size)

# Train-test split
split_index = int(len(X) * 0.9)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Build LSTM model
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(X_train.shape[1], 1), return_sequences=False))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=8, callbacks=[early_stopping], verbose=0)

# Make predictions

if len(X_test) == 0:
    st.error("🚫 Not enough data to make predictions. Try selecting another product or reducing the window size.")
    st.stop()

y_pred = model.predict(X_test)
y_pred_rescaled = scaler.inverse_transform(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Evaluation
rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
st.metric("📉 RMSE", f"{rmse:,.2f}")

# Plotting actual vs predicted
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(product_df['Date'][-len(y_test_rescaled):], y_test_rescaled, label='Actual', marker='o')
ax.plot(product_df['Date'][-len(y_test_rescaled):], y_pred_rescaled, label='Predicted', linestyle='--')
ax.set_title("Actual vs Predicted Demand (Deep Learning)")
ax.set_xlabel("Date")
ax.set_ylabel("Demand")
ax.legend()
st.pyplot(fig)

# Forecast next 7 days
def forecast_next_7_days_lstm(model, last_sequence, steps, scaler):
    forecast = []
    input_seq = last_sequence.copy()
    for _ in range(steps):
        pred = model.predict(input_seq[np.newaxis, :, :])[0]
        forecast.append(pred)
        input_seq = np.concatenate((input_seq[1:], [pred]), axis=0)
    return scaler.inverse_transform(forecast)

st.subheader("📆 Forecast for Next 7 Days")
last_sequence = X[-1]  # use the latest known sequence
future_forecast = forecast_next_7_days_lstm(model, last_sequence, 7, scaler)
future_dates = pd.date_range(start=product_df['Date'].max() + pd.Timedelta(days=1), periods=7)

forecast_df = pd.DataFrame({"Date": future_dates, "Forecasted_Demand": future_forecast.flatten()})

# Show recent actuals for comparison
actual_df = product_df[['Date', 'Order_Demand']].copy()
recent_actuals = actual_df.tail(7).rename(columns={"Order_Demand": "Actual_Demand"})
recent_actuals.reset_index(drop=True, inplace=True)
forecast_df.reset_index(drop=True, inplace=True)
recent_actuals = recent_actuals.drop(columns=["Date"])
combined_df = pd.concat([forecast_df, recent_actuals], axis=1)

st.dataframe(combined_df)

fig2, ax2 = plt.subplots()
ax2.plot(combined_df['Date'], combined_df['Forecasted_Demand'], marker='o', label='Forecasted Demand')
ax2.plot(combined_df['Date'], combined_df['Actual_Demand'], marker='x', linestyle='--', label='Actual Demand')
ax2.set_title("7-Day Forecasted vs Actual Demand (Deep Learning)")
ax2.set_ylabel("Demand")
ax2.set_xlabel("Date")
ax2.legend()
st.pyplot(fig2)



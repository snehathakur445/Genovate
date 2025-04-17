# 🧠 Predictrix: Retail Demand Forecaster

Predictrix is a Streamlit-powered web application that leverages deep learning (LSTM) to forecast retail product demand based on historical order data. This tool is ideal for retailers, inventory managers, and data analysts aiming to make data-driven demand planning decisions.

---

## 🚀 Features

- 📦 **Product-specific Forecasting** – Choose individual products to get tailored forecasts.
- 📈 **Time-Series Modeling** – Uses LSTM (Long Short-Term Memory) networks to model temporal dependencies.
- 📊 **Visual Insights** – Graphs for actual vs predicted demand, along with 7-day future forecast.
- 📉 **Performance Metrics** – Displays RMSE to evaluate prediction accuracy.
- 📅 **Future Demand Forecast** – Predicts next 7 days of demand with confidence.
- 🧩 **Interactive Streamlit UI** – Easy to use and runs in-browser.

---

## 📁 Dataset Format

The app expects a CSV file named `Retail_Dataset2.csv` with the following structure:

| Date       | Warehouse | Product_Category | Product_Code | Order_Demand | StateHoliday |
|------------|-----------|------------------|--------------|--------------|---------------|
| 2021-01-01 | W001      | Electronics       | P1001        | 123          | 0             |

> 🔁 Categorical features are automatically encoded internally.

---

## ⚙️ How It Works

1. Loads and preprocesses retail demand data.
2. Applies MinMax scaling to normalize demand values.
3. Creates time-series sequences using a sliding window (default: 30 days).
4. Trains an LSTM neural network on 90% of the data.
5. Evaluates performance on the remaining 10%.
6. Forecasts the next 7 days using the trained model.
7. Displays results with visual plots and interactive tables.

---

## 🧪 Technologies Used

- **Python**
- **Streamlit**
- **TensorFlow / Keras**
- **scikit-learn**
- **pandas / numpy / matplotlib**

---
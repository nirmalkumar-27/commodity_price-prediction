import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Commodity Price Prediction", layout="centered")

st.title("📈 Commodity Price Prediction System")

# Load dataset
df = pd.read_csv("commodity_prices_india_daily_2021_2026.csv")
df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year

# Commodity mapping
commodity_map = {
    "Gold": ("Gold_INR_per_10g", "₹ per 10 grams"),
    "Silver": ("Silver_INR_per_1kg", "₹ per 1 kg"),
    "Platinum": ("Platinum_INR_per_10g", "₹ per 10 grams")
}

# User inputs
commodity = st.selectbox("Select Commodity", list(commodity_map.keys()))
year = st.number_input("Enter Year to Predict", min_value=2026, max_value=2035, step=1)

column, unit = commodity_map[commodity]

# Train model
X = df[["Year"]]
y = df[column]

model = LinearRegression()
model.fit(X, y)

# Prediction
predicted_price = model.predict([[year]])[0]

# Previous year price for comparison
prev_price = df[df["Year"] == df["Year"].max()][column].mean()

trend = "Increase 📈" if predicted_price > prev_price else "Decrease 📉"

# Display output
st.subheader("🔮 Prediction Result")
st.write(f"**Commodity:** {commodity}")
st.write(f"**Predicted Price for {year}:** ₹ {predicted_price:,.2f} ({unit})")
st.write(f"**Trend:** {trend}")

# Visualization
st.subheader("📊 Price Trend Visualization")

historical = df.groupby("Year")[column].mean().reset_index()
future = pd.DataFrame({"Year": [year], column: [predicted_price]})

plot_df = pd.concat([historical, future])

plt.figure()
plt.plot(plot_df["Year"], plot_df[column])
plt.xlabel("Year")
plt.ylabel(f"Price ({unit})")
plt.title(f"{commodity} Price Trend")
st.pyplot(plt)

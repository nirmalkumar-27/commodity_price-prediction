import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Commodity Price Prediction",
    layout="centered"
)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("commodity_prices_india_daily_2021_2026.csv")
df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year

# ---------------- COMMODITY CONFIG ----------------
commodity_map = {
    "Gold": ("Gold_INR_per_10g", "₹ per 10 grams"),
    "Silver": ("Silver_INR_per_1kg", "₹ per 1 kg"),
    "Platinum": ("Platinum_INR_per_10g", "₹ per 10 grams")
}

# ---------------- BACKGROUND THEMES ----------------
def set_background(commodity):
    if commodity == "Gold":
        bg = "linear-gradient(to right, #f6d365, #fda085)"
    elif commodity == "Silver":
        bg = "linear-gradient(to right, #d7d2cc, #304352)"
    elif commodity == "Platinum":
        bg = "linear-gradient(to right, #434343, #000000)"
    else:
        bg = "#f5f5f5"

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {bg};
            color: black;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Default background
set_background(None)

# ---------------- TITLE ----------------
st.markdown("## 📈 Commodity Price Prediction System")

# ---------------- USER INPUTS ----------------
commodity = st.selectbox(
    "Select Commodity",
    ["-- Select Commodity --", "Gold", "Silver", "Platinum"]
)

year = st.number_input(
    "Enter Year to Predict",
    min_value=2026,
    max_value=2035,
    step=1
)

# Change background when selected
if commodity in commodity_map:
    set_background(commodity)

# ---------------- PREDICT BUTTON ----------------
if st.button("🔮 Predict Price"):
    if commodity not in commodity_map:
        st.warning("⚠ Please select a commodity")
    else:
        column, unit = commodity_map[commodity]

        # Train model
        X = df[["Year"]]
        y = df[column]

        model = LinearRegression()
        model.fit(X, y)

        predicted_price = model.predict([[year]])[0]

        prev_price = df[df["Year"] == df["Year"].max()][column].mean()
        trend = "Increase 📈" if predicted_price > prev_price else "Decrease 📉"

        # ---------------- RESULT ----------------
        st.subheader("✨ Prediction Result")
        st.write(f"**Commodity:** {commodity}")
        st.write(f"**Predicted Price for {year}:** ₹ {predicted_price:,.2f} ({unit})")
        st.write(f"**Trend:** {trend}")

        # ---------------- VISUALIZATION ----------------
        st.subheader("📊 Price Trend Visualization")

        hist = df.groupby("Year")[column].mean().reset_index()
        future = pd.DataFrame({"Year": [year], column: [predicted_price]})
        plot_df = pd.concat([hist, future])

        plt.figure()
        plt.plot(plot_df["Year"], plot_df[column])
        plt.xlabel("Year")
        plt.ylabel(unit)
        plt.title(f"{commodity} Price Trend")
        st.pyplot(plt)

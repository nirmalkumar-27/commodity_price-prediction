import streamlit as st
import pandas as pd
import base64
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Commodity Price Prediction",
    layout="centered"
)

# ---------------- BACKGROUND IMAGE FUNCTION ----------------
def set_bg_image(image_path):
    with open(image_path, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                linear-gradient(rgba(0,0,0,0.55), rgba(0,0,0,0.55)),
                url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ---------------- DEFAULT THEME (BEFORE SELECTION) ----------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #141E30, #243B55);
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("commodity_prices_india_daily_2021_2026.csv")
df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year

commodity_map = {
    "Gold": ("Gold_INR_per_10g", "₹ per 10 grams", "images/gold.jpg"),
    "Silver": ("Silver_INR_per_1kg", "₹ per 1 kg", "images/silver.jpg"),
    "Platinum": ("Platinum_INR_per_10g", "₹ per 10 grams", "images/platinum.jpg")
}

# ---------------- UI STYLES ----------------
st.markdown(
    """
    <style>
    .card {
        background: #ffffff;
        padding: 35px;
        border-radius: 18px;
        max-width: 650px;
        margin: auto;
        box-shadow: 0px 15px 35px rgba(0,0,0,0.35);
        color: #111111;
    }

    h2, h3, label, p {
        color: #111111 !important;
        font-weight: 600;
    }

    .stButton > button {
        width: 100%;
        padding: 10px;
        font-size: 16px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- BANNER (FIXES EMPTY BOX ISSUE) ----------------
st.markdown(
    """
    <div style="
        background: rgba(255,255,255,0.95);
        padding: 14px 22px;
        border-radius: 14px;
        text-align: center;
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 18px;">
        🇮🇳 India-Based Commodity Price Prediction using Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------- UI CARD START ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)

st.markdown("## 📈 Commodity Price Prediction System")

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

# ---------------- CHANGE BACKGROUND AFTER SELECTION ----------------
if commodity in commodity_map:
    set_bg_image(commodity_map[commodity][2])

# ---------------- PREDICTION ----------------
if st.button("🔮 Predict Price"):
    if commodity not in commodity_map:
        st.warning("Please select a commodity")
    else:
        column, unit, _ = commodity_map[commodity]

        X = df[["Year"]]
        y = df[column]

        model = LinearRegression()
        model.fit(X, y)

        predicted_price = model.predict([[year]])[0]
        last_price = df[df["Year"] == df["Year"].max()][column].mean()
        trend = "Increase 📈" if predicted_price > last_price else "Decrease 📉"

        st.subheader("✨ Prediction Result")
        st.write(f"**Commodity:** {commodity}")
        st.write(f"**Predicted Price for {year}:** ₹ {predicted_price:,.2f} ({unit})")
        st.write(f"**Trend:** {trend}")

        st.subheader("📊 Price Trend")

        hist = df.groupby("Year")[column].mean().reset_index()
        hist.loc[len(hist)] = [year, predicted_price]

        plt.figure(figsize=(7,4))
        plt.plot(hist["Year"], hist[column], marker="o")
        plt.xlabel("Year")
        plt.ylabel(unit)
        plt.title(f"{commodity} Price Trend")
        plt.grid(True)
        st.pyplot(plt)

# ---------------- UI CARD END ----------------
st.markdown('</div>', unsafe_allow_html=True)

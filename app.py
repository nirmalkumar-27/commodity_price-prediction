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

# ---------------- DEFAULT BACKGROUND ----------------
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

# ---------------- HIDE STREAMLIT TOP SPACE ----------------
st.markdown(
    """
    <style>
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {
        padding-top: 2rem;
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

# ---------------- MAIN CARD STYLE ----------------
st.markdown(
    """
    <style>
    .main-card {
        background: rgba(255,255,255,0.96);
        padding: 40px;
        border-radius: 20px;
        max-width: 650px;
        margin: auto;
        box-shadow: 0px 20px 40px rgba(0,0,0,0.4);
        color: #111;
    }

    h2, h3, label {
        color: #111 !important;
        font-weight: 700;
    }

    .stButton > button {
        width: 100%;
        padding: 12px;
        font-size: 16px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- MAIN CARD START ----------------
st.markdown('<div class="main-card">', unsafe_allow_html=True)

st.markdown("## 📈 Commodity Price Prediction System")
st.caption("India-based ML prediction using historical commodity prices")

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

# ---------------- BACKGROUND CHANGE ----------------
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

# ---------------- MAIN CARD END ----------------
st.markdown('</div>', unsafe_allow_html=True)

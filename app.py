import streamlit as st
import pandas as pd
import base64
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Commodity Price Prediction",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- REMOVE STREAMLIT DEFAULT PADDING ----------------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- BACKGROUND IMAGE FUNCTION ----------------
def set_bg_image(image_path):
    with open(image_path, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
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
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
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

# ---------------- CARD STYLE ----------------
st.markdown(
    """
    <style>
    .center-card {
        background: white;
        padding: 40px;
        border-radius: 20px;
        width: 520px;
        margin: 120px auto;
        box-shadow: 0px 20px 45px rgba(0,0,0,0.45);
    }

    .title {
        text-align: center;
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .subtitle {
        text-align: center;
        font-size: 14px;
        color: #666;
        margin-bottom: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- CENTER CONTAINER ----------------
with st.container():
    st.markdown('<div class="center-card">', unsafe_allow_html=True)

    st.markdown('<div class="title">📈 Commodity Price Prediction System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">India-based ML prediction using historical commodity prices</div>',
        unsafe_allow_html=True
    )

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

    # Change background AFTER selection
    if commodity in commodity_map:
        set_bg_image(commodity_map[commodity][2])

    if st.button("🔮 Predict Price", use_container_width=True):
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

            plt.figure()
            plt.plot(hist["Year"], hist[column], marker="o")
            plt.xlabel("Year")
            plt.ylabel(unit)
            plt.title(f"{commodity} Price Trend")
            st.pyplot(plt)

    st.markdown('</div>', unsafe_allow_html=True)

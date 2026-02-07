import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import base64

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Commodity Prediction", layout="wide")

# ---------------- BACKGROUND ----------------
def set_bg(image=None):
    if image:
        with open(image, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{b64}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
        }
        </style>
        """, unsafe_allow_html=True)

set_bg()

# ---------------- DATA ----------------
df = pd.read_csv("commodity_prices_india_daily_2021_2026.csv")
df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year

commodity_map = {
    "Gold": ("Gold_INR_per_10g", "₹ / 10g", "images/gold.jpg"),
    "Silver": ("Silver_INR_per_1kg", "₹ / kg", "images/silver.jpg"),
    "Platinum": ("Platinum_INR_per_10g", "₹ / 10g", "images/platinum.jpg")
}

# ---------------- CENTER CARD ----------------
left, center, right = st.columns([1, 1.2, 1])

with center:
    st.markdown("""
    <style>
    .card {
        background: white;
        padding: 35px;
        border-radius: 18px;
        box-shadow: 0 15px 40px rgba(0,0,0,0.4);
    }
    </style>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.markdown("### 📈 Commodity Price Prediction System")
        st.caption("India-based ML prediction using historical data")

        commodity = st.selectbox(
            "Select Commodity",
            ["-- Select --", "Gold", "Silver", "Platinum"]
        )

        year = st.number_input("Enter Year", min_value=2026, max_value=2035)

        if commodity in commodity_map:
            set_bg(commodity_map[commodity][2])

        if st.button("🔮 Predict Price", use_container_width=True):
            if commodity not in commodity_map:
                st.warning("Please select a commodity")
            else:
                col, unit, _ = commodity_map[commodity]

                X = df[["Year"]]
                y = df[col]

                model = LinearRegression()
                model.fit(X, y)

                pred = model.predict([[year]])[0]

                st.success(f"Predicted Price ({year}) : ₹ {pred:,.2f} {unit}")

                hist = df.groupby("Year")[col].mean().reset_index()
                hist.loc[len(hist)] = [year, pred]

                fig, ax = plt.subplots()
                ax.plot(hist["Year"], hist[col], marker="o")
                ax.set_xlabel("Year")
                ax.set_ylabel(unit)
                ax.set_title(f"{commodity} Price Trend")
                st.pyplot(fig)

        st.markdown('</div>', unsafe_allow_html=True)

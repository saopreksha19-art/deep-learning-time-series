from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import joblib
from tensorflow.keras.models import load_model

# --------------------------------------------------
# Configuration
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

FEATURES = [
    "sales", "day_of_week", "day_of_month", "week_of_year",
    "month", "quarter", "is_weekend", "snap_CA",
    "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_28",
    "rolling_std_7", "rolling_std_28"
]

st.set_page_config(
    page_title="Demand Forecasting",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
/* Overall application */
.stApp {
    background-color: #F7F7F4;
    color: #242620;
}

[data-testid="stHeader"] {
    background-color: #F7F7F4;
}

.block-container {
    max-width: 1440px;
    padding-top: 2.4rem;
    padding-bottom: 3rem;
}

/* Typography */
html, body, [class*="css"], p, label {
    color: #242620;
}

h1, h2, h3 {
    color: #20231F !important;
    font-weight: 650 !important;
    letter-spacing: -0.035em;
}

p, label, .stMarkdown {
    color: #454840;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #EFEFEA;
    border-right: 1px solid #E0E0D9;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] label {
    color: #242620 !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E5E5DF;
    padding: 20px 22px;
    border-radius: 12px;
}

[data-testid="stMetricLabel"] {
    color: #656860 !important;
    font-size: 0.85rem;
}

[data-testid="stMetricValue"] {
    color: #20231F !important;
    font-weight: 650;
}

/* Inputs and selectors */
.stSelectbox div[data-baseweb="select"] > div,
.stSlider,
.stTextInput input {
    background-color: #FFFFFF;
    color: #242620;
}

[data-baseweb="select"] {
    border-radius: 8px;
}

/* Buttons: pastel sage with readable dark text */
.stButton > button,
.stFormSubmitButton > button {
    background-color: #DCE9E2 !important;
    color: #263D32 !important;
    border: 1px solid #B8CEC0 !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.2rem;
    font-weight: 600 !important;
    transition: background-color 0.2s ease;
}

.stButton > button p,
.stFormSubmitButton > button p {
    color: #263D32 !important;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background-color: #C7DCCF !important;
    color: #20372C !important;
    border-color: #9FBDAA !important;
}

.stButton > button:hover p,
.stFormSubmitButton > button:hover p {
    color: #20372C !important;
}

.stButton > button:focus,
.stFormSubmitButton > button:focus {
    color: #20372C !important;
    box-shadow: 0 0 0 2px #B8CEC0 !important;
}

/* Dataframes and tables: light pastel palette */
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    background-color: #FFFFFF !important;
    border: 1px solid #DDE7DE !important;
    border-radius: 10px !important;
    overflow: hidden;
}

/* Data editor / dataframe surface */
[data-testid="stDataFrame"] [data-testid="glideDataEditor"] {
    background-color: #FAFCFA !important;
}

/* Static table headers */
[data-testid="stTable"] thead th {
    background-color: #E5EEE7 !important;
    color: #293B30 !important;
    font-weight: 600 !important;
}

/* Static table cells */
[data-testid="stTable"] tbody td {
    background-color: #FAFCFA !important;
    color: #303A33 !important;
    border-color: #E7ECE7 !important;
}

/* Alternating rows for static tables */
[data-testid="stTable"] tbody tr:nth-child(even) td {
    background-color: #F0F5F0 !important;
}

/* Chart containers */
[data-testid="stPlotlyChart"] {
    background-color: #FFFFFF;
    border: 1px solid #E5E5DF;
    border-radius: 12px;
    padding: 12px;
}

/* Forecast notice */
[data-testid="stAlert"] {
    background-color: #EAF1EB !important;
    color: #304438 !important;
    border: 1px solid #D6E3D8 !important;
    border-radius: 10px !important;
}

[data-testid="stAlert"] p {
    color: #304438 !important;
}

/* Dividers and muted text */
hr {
    border-color: #E1E1DA;
}

small, [data-testid="stCaptionContainer"] {
    color: #73766D !important;
}

/* Remove unnecessary toolbar visual weight */
[data-testid="stToolbar"] {
    background: transparent;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Chart styling
# --------------------------------------------------

def style_chart(fig):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#F7F7F4",
        plot_bgcolor="#FFFFFF",
        font=dict(
            family="Arial, sans-serif",
            color="#292D28",
            size=13
        ),
        title_font=dict(size=17, color="#20231F"),
        margin=dict(l=24, r=24, t=56, b=24),
        legend=dict(
            bgcolor="rgba(255,255,255,0)",
            font=dict(color="#454840", size=12)
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="#E9EAE4",
            zeroline=False,
            linecolor="#D8DAD2",
            tickfont=dict(color="#454840")
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#E9EAE4",
            zeroline=False,
            linecolor="#D8DAD2",
            tickfont=dict(color="#454840")
        )
    )
    return fig

# --------------------------------------------------
# Load artifacts
# --------------------------------------------------
@st.cache_data
def load_sales():
    df = pd.read_csv(DATA_DIR / "features.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["item_id", "date"])

@st.cache_data
def load_comparison():
    return pd.read_csv(DATA_DIR / "model_comparison.csv")

@st.cache_resource
def load_gru():
    return load_model(MODEL_DIR / "gru_demand_forecasting.keras")

@st.cache_resource
def load_scaler():
    return joblib.load(MODEL_DIR / "scaler.pkl")

try:
    sales_df = load_sales()
    comparison_df = load_comparison()
    model = load_gru()
    scaler = load_scaler()
except Exception as exc:
    st.error(f"Could not load project files: {exc}")
    st.stop()

# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("Neural Demand Forecasting")
st.caption(
    "Retail demand forecasting using a trained GRU model."
    "with a 28-day lookback and a 7-day forecast horizon."
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("Forecast Settings")

items = sorted(sales_df["item_id"].unique())

selected_item = st.sidebar.selectbox(
    "Select product",
    items,
    index=items.index("FOODS_3_090") if "FOODS_3_090" in items else 0
)

item_df = (
    sales_df[sales_df["item_id"] == selected_item]
    .sort_values("date")
    .copy()
)

history_days = st.sidebar.slider(
    "Historical chart window (days)",
    min_value=30,
    max_value=min(365, len(item_df)),
    value=min(90, len(item_df))
)

# --------------------------------------------------
# Sales Data Validation
# --------------------------------------------------
st.subheader("Sales Data Validation")

st.write("Product:", selected_item)
st.write(
    "Date range:",
    item_df["date"].min(),
    "to",
    item_df["date"].max()
)
st.write("Rows:", len(item_df))
st.write(
    "Non-zero sales days:",
    int((item_df["sales"] > 0).sum())
)
st.write(
    "Zero-sales days:",
    int((item_df["sales"] == 0).sum())
)

st.dataframe(
    item_df[["date", "sales"]].tail(35),
    use_container_width=True,
    hide_index=True
)
# --------------------------------------------------
# Debug selected product's sales history
# --------------------------------------------------
st.write(
    "Date range:",
    item_df["date"].min().date(),
    "to",
    item_df["date"].max().date()
)

st.write("Total rows:", len(item_df))
st.write("Days with sales:", int((item_df["sales"] > 0).sum()))
st.write("Total units sold:", int(item_df["sales"].sum()))

# --------------------------------------------------
# KPI metrics
# --------------------------------------------------
latest_sales = float(item_df["sales"].iloc[-1])
total_sales = float(item_df["sales"].sum())
active_days = int((item_df["sales"] > 0).sum())

c1, c2, c3 = st.columns(3)
c1.metric("Latest recorded sales", f"{latest_sales:.0f} units")
c2.metric("Total recorded sales", f"{total_sales:,.0f} units")
c3.metric("Days with sales", f"{active_days:,}")


# --------------------------------------------------
# Historical sales
# --------------------------------------------------
st.subheader("Historical Sales")

history = item_df.tail(history_days)

fig_history = px.line(
    history,
    x="date",
    y="sales",
    title=f"Historical demand — {selected_item}",
    labels={"date": "Date", "sales": "Units sold"}
)

fig_history.update_traces(
    line=dict(color="#52796F", width=2.5)
)

fig_history.update_layout(hovermode="x unified")
fig_history = style_chart(fig_history)

st.plotly_chart(fig_history, use_container_width=True)

# --------------------------------------------------
# Forecast
# --------------------------------------------------
st.subheader("Next 7-Day Forecast")

if st.button("Generate forecast", type="primary"):
    if len(item_df) < 28:
        st.error("At least 28 days of history are required.")
        st.stop()

    missing_features = [
        col for col in FEATURES if col not in item_df.columns
    ]

    if missing_features:
        st.error(
            f"Missing model features: {missing_features}. "
            "Ensure the dashboard loads feature-engineered data."
        )
        st.stop()

    last_28 = item_df.tail(28).copy()

    if last_28[FEATURES].isnull().any().any():
        st.error("The latest 28 days contain missing feature values.")
        st.stop()

    # Scale the latest 28 days using the training-fitted scaler.
    x = last_28[FEATURES].to_numpy(dtype=np.float32)
    x_scaled = scaler.transform(x)
    X_input = x_scaled[np.newaxis, :, :]

    # Predict the next 7 days.
    pred_scaled = model.predict(X_input, verbose=0)[0]

    # Inverse-transform the predicted sales.
    pred = (pred_scaled - scaler.min_[0]) / scaler.scale_[0]
    pred = np.maximum(pred, 0)

    last_date = item_df["date"].max()
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=7,
        freq="D"
    )

    forecast_df = pd.DataFrame({
        "date": future_dates,
        "predicted_sales": pred
    })

    st.dataframe(
        forecast_df.assign(
            predicted_sales=forecast_df["predicted_sales"].round(2)
        ),
        use_container_width=True,
        hide_index=True
    )

    fig_forecast = px.line(
        forecast_df,
        x="date",
        y="predicted_sales",
        markers=True,
        title=f"Seven-day forecast — {selected_item}",
        labels={
            "date": "Forecast date",
            "predicted_sales": "Predicted units"
        }
    )

    fig_forecast.update_traces(
        line=dict(color="#B77952", width=2.5),
        marker=dict(color="#B77952", size=7)
    )

    fig_forecast = style_chart(fig_forecast)
    st.plotly_chart(fig_forecast, use_container_width=True)

    st.info(
        "Forecasts are model estimates, not guaranteed sales. "
        "Predicted values are clipped at zero because negative sales "
        "are not meaningful."
    )


# --------------------------------------------------
# Model comparison
# --------------------------------------------------
st.subheader("Model Performance")

if {"Model", "MAE", "RMSE"}.issubset(comparison_df.columns):
    st.dataframe(
        comparison_df.round(3),
        use_container_width=True,
        hide_index=True
    )

    metrics_long = comparison_df.melt(
        id_vars="Model",
        value_vars=["MAE", "RMSE"],
        var_name="Metric",
        value_name="Error"
    )

    fig_metrics = px.bar(
        metrics_long,
        x="Model",
        y="Error",
        color="Metric",
        barmode="group",
        title="Test-set model comparison",
        labels={"Error": "Error (sales units)"},
        color_discrete_map={
            "MAE": "#52796F",
            "RMSE": "#B77952"
        }
    )

    fig_metrics = style_chart(fig_metrics)
    st.plotly_chart(fig_metrics, use_container_width=True)

else:
    st.warning(
        "The model comparison CSV does not have the expected columns."
    )


st.caption(
    "Models trained on the M5 Forecasting Accuracy dataset, "
    "using a selected subset of products from store CA_1."
)

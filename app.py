"""
AgriTrack Palay Price Predictor
Run locally: streamlit run app.py

This version forecasts automatically from the latest monthly observation
in the CSV. It does not ask the farmer to type six prices manually.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "palay_price_model.joblib"
METRICS_PATH = BASE_DIR / "model_metrics.json"
DATA_CANDIDATES = [
    BASE_DIR / "palay_price_prediction_dataset.csv",
    BASE_DIR / "palay_price_prediction_training(1).csv",
    BASE_DIR / "palay_price_prediction_training.csv",
]

st.set_page_config(
    page_title="AgriTrack | Palay Price Prediction",
    page_icon="🌾",
    layout="wide",
)
st.title("🌾 AgriTrack Palay Price Prediction")
st.write(
    "Monthly palay price forecasts generated from the latest available "
    "historical price records."
)
st.caption(
    "Forecasts are estimates, not guaranteed market prices. Accuracy depends "
    "on the quality, recency, and amount of verified historical data."
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_monthly_history(data_path_string, file_mtime):
    # file_mtime is included to refresh the cache if the CSV changes.
    del file_mtime
    df = pd.read_csv(data_path_string)
    required = {"price_date", "palay_price_per_kg"}
    if not required.issubset(df.columns):
        raise ValueError(
            "CSV must contain price_date and palay_price_per_kg columns."
        )
    df["price_date"] = pd.to_datetime(df["price_date"], errors="coerce")
    df["palay_price_per_kg"] = pd.to_numeric(
        df["palay_price_per_kg"], errors="coerce"
    )
    df = df.dropna(subset=["price_date", "palay_price_per_kg"])
    df = df[df["palay_price_per_kg"] >= 0].copy()
    df["month_start"] = df["price_date"].dt.to_period("M").dt.to_timestamp()

    monthly = (
        df.groupby("month_start", as_index=False)["palay_price_per_kg"]
        .mean()
        .sort_values("month_start")
        .reset_index(drop=True)
    )
    if monthly.empty:
        raise ValueError("The CSV has no valid price records.")
    return monthly


def make_feature_row(history, target_date, origin_date):
    values = np.asarray(history, dtype=float)
    target_date = pd.Timestamp(target_date)
    origin_date = pd.Timestamp(origin_date)
    time_index = (
        (target_date.year - origin_date.year) * 12
        + target_date.month - origin_date.month
    )
    return pd.DataFrame([{
        "current_price": float(values[-1]),
        "previous_price": float(values[-2]) if len(values) >= 2 else float(values[-1]),
        "avg_3_month": float(np.mean(values[-3:])),
        "avg_6_month": float(np.mean(values[-6:])),
        "month_sin": float(np.sin(2 * np.pi * target_date.month / 12)),
        "month_cos": float(np.cos(2 * np.pi * target_date.month / 12)),
        "time_index": int(time_index),
    }])


def forecast_months(bundle, history, latest_month, months_ahead):
    model = bundle["model"]
    columns = bundle["feature_columns"]
    origin = pd.Timestamp(bundle["origin_date"])
    history = list(history)
    current_month = pd.Timestamp(latest_month).to_period("M").to_timestamp()
    rows = []

    for step in range(1, months_ahead + 1):
        target_date = current_month + pd.DateOffset(months=step)
        X = make_feature_row(history, target_date, origin)[columns]
        predicted_price = max(0.0, float(model.predict(X)[0]))
        previous_price = float(history[-1])
        change = predicted_price - previous_price
        direction = (
            "Increase" if change > 0.005
            else "Decrease" if change < -0.005
            else "No major change"
        )
        rows.append({
            "Month": target_date.strftime("%B %Y"),
            "Predicted price (₱/kg)": round(predicted_price, 2),
            "Change vs. previous month (₱/kg)": round(change, 2),
            "Expected direction": direction,
        })
        history.append(predicted_price)

    return pd.DataFrame(rows)


data_path = next((p for p in DATA_CANDIDATES if p.exists()), None)
if data_path is None:
    st.error(
        "Historical price CSV not found. Upload one of these files beside app.py: "
        + ", ".join(p.name for p in DATA_CANDIDATES)
    )
    st.stop()

if not MODEL_PATH.exists():
    st.error("Model file not found. Run `python train_model.py` first.")
    st.stop()

try:
    bundle = load_model()
    monthly = load_monthly_history(str(data_path), data_path.stat().st_mtime)
except Exception as exc:
    st.error(f"Could not load the model or historical data: {exc}")
    st.stop()

if len(monthly) < 2:
    st.error("At least two monthly observations are needed to create a forecast.")
    st.stop()

latest_month = pd.Timestamp(monthly["month_start"].iloc[-1])
latest_price = float(monthly["palay_price_per_kg"].iloc[-1])
history = monthly["palay_price_per_kg"].astype(float).tolist()

with st.sidebar:
    st.header("Forecast settings")
    horizon = st.selectbox("Forecast horizon", [3, 6, 12], index=2)
    st.caption("The forecast updates when the source CSV and trained model are updated.")

st.subheader("1. Latest available historical price")
c1, c2, c3 = st.columns(3)
c1.metric("Latest observed price", f"₱{latest_price:.2f}/kg")
c2.metric("Latest month in data", latest_month.strftime("%B %Y"))
c3.metric("Monthly records", f"{len(monthly)}")

if len(monthly) < 24:
    st.warning(
        "There are fewer than 24 monthly observations. Treat the forecast as "
        "experimental and collect more verified historical prices."
    )

result = forecast_months(bundle, history, latest_month, horizon)
st.subheader("2. Forecast results")

first = result.iloc[0]
first_change = float(first["Change vs. previous month (₱/kg)"])
st.metric(
    f"Predicted price for {first['Month']}",
    f"₱{first['Predicted price (₱/kg)']:.2f}/kg",
    delta=f"₱{first_change:+.2f}/kg vs. latest observed price",
)

increases = result[result["Expected direction"] == "Increase"]
if not increases.empty:
    next_increase = increases.iloc[0]
    st.success(
        f"First predicted month with an increase: **{next_increase['Month']}**, "
        f"at an estimated **₱{next_increase['Predicted price (₱/kg)']:.2f}/kg**."
    )
else:
    st.info("The model did not predict an increase during this forecast period.")

st.dataframe(result, use_container_width=True, hide_index=True)
st.line_chart(result.set_index("Month")[["Predicted price (₱/kg)"]])

st.download_button(
    "Download forecast CSV",
    data=result.to_csv(index=False).encode("utf-8"),
    file_name="agritrack_palay_price_forecast.csv",
    mime="text/csv",
)

st.caption(
    "This is a recursive forecast: later months use earlier model predictions "
    "as inputs, so uncertainty generally grows farther into the future."
)

st.divider()
st.subheader("Model information")
m1, m2, m3 = st.columns(3)
m1.write(f"**Model type:** {bundle.get('model_type', 'Unknown')}")
m2.write(f"**Training data:** {bundle.get('source_csv', 'Unknown')}")
m3.write(f"**Last training month:** {bundle.get('last_training_date', 'Unknown')}")
st.write(f"**Training examples:** {bundle.get('training_rows', 'Unknown')}")

if METRICS_PATH.exists():
    try:
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        a, b, c = st.columns(3)
        a.metric("Test MAE", f"₱{metrics['mae_pesos_per_kg']:.2f}/kg")
        b.metric("Test RMSE", f"₱{metrics['rmse_pesos_per_kg']:.2f}/kg")
        c.metric(
            "Last-price baseline MAE",
            f"₱{metrics['persistence_baseline_mae_pesos_per_kg']:.2f}/kg",
        )
        if not metrics.get("model_beats_baseline", True):
            st.warning(
                "The model did not beat the simple last-price baseline on its "
                "historical test. Do not claim improved accuracy until it does."
            )
        st.caption(metrics.get("warning", "Metrics are historical estimates, not guarantees."))
    except Exception:
        st.warning("Model metrics could not be read. Retrain the model.")

st.info(
    "New farmer-entered prices will not automatically retrain this CSV-based app. "
    "Verified observed prices must be added to the training source and the model "
    "retrained, or the app must be connected to a secure database/API integration."
)

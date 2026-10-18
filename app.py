
from pathlib import Path
import calendar
import os

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AgriTrack Palay Price Prediction",
    page_icon="🌾",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "palay_price_model.joblib"
METRICS_PATH = BASE_DIR / "model_metrics.json"

st.title("🌾 AgriTrack Palay Price Prediction")
st.caption(
    "Monthly palay selling-price history and model-based forecasts "
    "using AgriTrack farmer records."
)


# ============================================================
# 2. API CONFIGURATION
# ============================================================

def get_secret(name, default=""):
    """Read a Streamlit Secret, then fall back to environment."""
    try:
        value = st.secrets.get(name, default)
    except Exception:
        value = default

    return str(value or os.getenv(name, default)).strip()


API_URL = get_secret("PALAY_PRICE_API_URL")
API_KEY = get_secret("PALAY_PRICE_API_KEY")


# ============================================================
# 3. LOAD ACTUAL SELLING PRICES FROM BYETHOST API
# ============================================================

@st.cache_data(ttl=60, show_spinner=False)
def fetch_palay_prices(api_url, api_key):
    if not api_url or not api_key:
        raise ValueError(
            "Missing PALAY_PRICE_API_URL or PALAY_PRICE_API_KEY "
            "in Streamlit Secrets."
        )
response = requests.get(
    api_url,
    headers={
    "X-API-Key": api_key,
        "Accept": "application/json",
        "User-Agent": "AgriTrackPriceApp/1.0",
        "Connection": "close",
    },
    timeout=30,
)

response.raise_for_status()
payload = response.json()
if response.status_code == 401:
        raise PermissionError(
            "API authentication failed. Check the API key in "
            "Streamlit Secrets and api_config.php."
        )

    response.raise_for_status()

    try:
        payload = response.json()
    except ValueError as exc:
        raise ValueError(
            "The API did not return valid JSON."
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError("Unexpected API response format.")

    if not payload.get("success"):
        raise ValueError(
            payload.get("message", "The API returned an error.")
        )

    records = payload.get("data", [])

    if not isinstance(records, list):
        raise ValueError("The API data field is not a list.")

    df = pd.DataFrame(records)

    required_columns = [
        "price_date",
        "palay_price_per_kg",
    ]

    if df.empty:
        return pd.DataFrame(
            columns=[
                "price_date",
                "palay_price_per_kg",
                "record_count",
            ]
        )

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "API response is missing columns: "
            + ", ".join(missing)
        )

    df["price_date"] = pd.to_datetime(
        df["price_date"], errors="coerce"
    )

    df["palay_price_per_kg"] = pd.to_numeric(
        df["palay_price_per_kg"], errors="coerce"
    )

    if "record_count" not in df.columns:
        df["record_count"] = 0

    df["record_count"] = pd.to_numeric(
        df["record_count"], errors="coerce"
    ).fillna(0).astype(int)

    df = df.dropna(
        subset=["price_date", "palay_price_per_kg"]
    )

    df = df[df["palay_price_per_kg"] > 0].copy()

    # Normalize every record to the first day of its month.
    df["price_date"] = (
        df["price_date"].dt.to_period("M").dt.to_timestamp()
    )

    # Combine any duplicate months into a single monthly average.
    df = (
        df.groupby("price_date", as_index=False)
        .agg(
            palay_price_per_kg=("palay_price_per_kg", "mean"),
            record_count=("record_count", "sum"),
        )
        .sort_values("price_date")
        .reset_index(drop=True)
    )

    df["palay_price_per_kg"] = (
        df["palay_price_per_kg"].round(2)
    )

    return df


# ============================================================
# 4. LOAD THE TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model_bundle(model_path, modified_time):
    # modified_time allows Streamlit to reload the model
    # when the file changes.
    del modified_time
    return joblib.load(model_path)


def get_model_bundle():
    if not MODEL_PATH.exists():
        return None, (
            f"Model file not found: {MODEL_PATH.name}. "
            "Train the model and include this file in your "
            "GitHub repository."
        )

    try:
        bundle = load_model_bundle(
            str(MODEL_PATH),
            MODEL_PATH.stat().st_mtime,
        )

        if not isinstance(bundle, dict):
            return None, (
                "The model file is not a dictionary bundle. "
                "Check the output format of train_model.py."
            )

        required_keys = [
            "model",
            "feature_columns",
            "origin_date",
        ]

        missing = [
            key for key in required_keys
            if key not in bundle
        ]

        if missing:
            return None, (
                "The trained model bundle is missing: "
                + ", ".join(missing)
            )

        return bundle, None

    except Exception as exc:
        return None, (
            "Unable to load the trained model: "
            + str(exc)
        )


# ============================================================
# 5. FEATURE ENGINEERING
# ============================================================

FEATURE_COLUMNS = [
    "current_price",
    "previous_price",
    "avg_3_month",
    "avg_6_month",
    "month_sin",
    "month_cos",
    "time_index",
]


def make_features(prices, target_date, origin_date):
    """
    Create the seven features expected by the trained model.

    prices: observed and/or recursively forecast monthly prices.
    target_date: month being predicted.
    origin_date: reference date saved by train_model.py.
    """
    target_date = pd.Timestamp(target_date)
    origin_date = pd.Timestamp(origin_date)

    target_date = target_date.to_period("M").to_timestamp()
    origin_date = origin_date.to_period("M").to_timestamp()

    history = prices.sort_index()

    if history.empty:
        raise ValueError("No prices available for prediction.")

    # Get the latest known monthly prices before the target month.
    history = history[history.index < target_date]

    if history.empty:
        raise ValueError(
            "No prior monthly price is available for this forecast."
        )

    values = history.astype(float).tolist()

    current_price = values[-1]
    previous_price = values[-2] if len(values) >= 2 else values[-1]

    avg_3_month = float(np.mean(values[-3:]))
    avg_6_month = float(np.mean(values[-6:]))

    month_number = target_date.month

    month_sin = np.sin(
        2 * np.pi * month_number / 12
    )

    month_cos = np.cos(
        2 * np.pi * month_number / 12
    )

    time_index = (
        (target_date.year - origin_date.year) * 12
        + target_date.month
        - origin_date.month
    )

    return {
        "current_price": float(current_price),
        "previous_price": float(previous_price),
        "avg_3_month": float(avg_3_month),
        "avg_6_month": float(avg_6_month),
        "month_sin": float(month_sin),
        "month_cos": float(month_cos),
        "time_index": float(time_index),
    }


# ============================================================
# 6. GENERATE MONTHLY FORECASTS
# ============================================================

def forecast_prices(history_df, bundle, horizon):
    if len(history_df) < 6:
        raise ValueError(
            "At least 6 distinct monthly prices are required "
            "for this app's forecast safeguard. More historical "
            "months are recommended for reliable predictions."
        )

    model = bundle["model"]
    feature_columns = list(bundle["feature_columns"])
    origin_date = pd.Timestamp(bundle["origin_date"])

    if not set(feature_columns).issubset(FEATURE_COLUMNS):
        raise ValueError(
            "The saved model uses feature names that this app "
            "does not recognize. Make sure train_model.py and "
            "app.py use the same feature definitions."
        )

    prices = history_df.set_index(
        "price_date"
    )["palay_price_per_kg"].astype(float)

    prices.index = pd.DatetimeIndex(prices.index)

    last_date = prices.index.max()
    forecast_rows = []

    for step in range(1, horizon + 1):
        target_date = (
            last_date.to_period("M") + step
        ).to_timestamp()

        features = make_features(
            prices=prices,
            target_date=target_date,
            origin_date=origin_date,
        )

        feature_frame = pd.DataFrame(
            [[features[name] for name in feature_columns]],
            columns=feature_columns,
        )

        prediction = float(
            np.asarray(model.predict(feature_frame)).reshape(-1)[0]
        )

        # Selling prices cannot be negative.
        prediction = max(0.0, prediction)

        forecast_rows.append(
            {
                "price_date": target_date,
                "predicted_price_per_kg": round(prediction, 2),
            }
        )

        # Recursive forecast: the prediction becomes the next
        # month's current/previous-price input.
        prices.loc[target_date] = prediction

    return pd.DataFrame(forecast_rows)


# ============================================================
# 7. API CONNECTION AND HISTORICAL PRICE DISPLAY
# ============================================================

st.subheader("1. Actual Farmer Selling Prices")

if not API_URL or not API_KEY:
    st.error(
        "API settings are missing. Add PALAY_PRICE_API_URL and "
        "PALAY_PRICE_API_KEY under Streamlit Cloud → Settings → Secrets."
    )
    st.stop()

try:
    with st.spinner("Loading selling prices from AgriTrack..."):
        historical_df = fetch_palay_prices(API_URL, API_KEY)

except PermissionError as exc:
    st.error(str(exc))
    st.info(
        "The API URL can return Unauthorized when opened directly "
        "in a browser. The Streamlit request must send the "
        "X-API-Key header."
    )
    st.stop()

except requests.RequestException as exc:
    st.error(
        "Could not connect to the AgriTrack API. "
        "Check the API URL, hosting status, and network connection."
    )
    st.caption(str(exc))
    st.stop()

except Exception as exc:
    st.error("Unable to load price data.")
    st.caption(str(exc))
    st.stop()


if historical_df.empty:
    st.warning(
        "The API connected successfully, but no valid selling "
        "prices were returned. Check the farmer's Post-Harvest "
        "selling records in farm_activities."
    )
    st.stop()


latest_row = historical_df.iloc[-1]
latest_price = float(latest_row["palay_price_per_kg"])

metric_cols = st.columns(3)

metric_cols[0].metric(
    "Latest Recorded Price",
    f"₱{latest_price:,.2f}/kg",
)

metric_cols[1].metric(
    "Monthly Records",
    f"{len(historical_df)}",
)

metric_cols[2].metric(
    "Selling Entries",
    f"{int(historical_df['record_count'].sum()):,}",
)

st.dataframe(
    historical_df.rename(
        columns={
            "price_date": "Month",
            "palay_price_per_kg": "Average Price (₱/kg)",
            "record_count": "Selling Records",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

st.line_chart(
    historical_df.set_index("price_date")[
        "palay_price_per_kg"
    ],
    y_label="Price per kg (₱)",
)

st.download_button(
    label="Download Historical Prices (CSV)",
    data=historical_df.to_csv(index=False).encode("utf-8"),
    file_name="agritrack_historical_palay_prices.csv",
    mime="text/csv",
)

st.divider()


# ============================================================
# 8. PRICE FORECASTING
# ============================================================

st.subheader("2. Future Palay Price Forecast")

bundle, model_error = get_model_bundle()

if bundle is None:
    st.warning(
        "Historical prices are available, but forecasting is "
        "not ready because the trained model could not be loaded."
    )
    st.code(model_error)
    st.info(
        "Train the model with train_model.py, save "
        "palay_price_model.joblib, and push the model file to "
        "the deployed app's repository."
    )
    st.stop()


if len(historical_df) < 6:
    st.warning(
        f"Only {len(historical_df)} distinct monthly price "
        "record(s) are available. This app requires at least "
        "6 months before displaying a forecast. Save more real "
        "selling records across different months first."
    )
    st.stop()


horizon = st.selectbox(
    "Forecast period",
    options=[3, 6, 12],
    format_func=lambda n: f"Next {n} months",
    index=1,
)

try:
    forecast_df = forecast_prices(
        history_df=historical_df,
        bundle=bundle,
        horizon=horizon,
    )

except Exception as exc:
    st.error("Unable to generate a forecast.")
    st.caption(str(exc))
    st.info(
        "Verify that the model's feature columns and feature "
        "engineering match the training script."
    )
    st.stop()


latest_month = historical_df["price_date"].max()
last_actual_price = float(
    historical_df.iloc[-1]["palay_price_per_kg"]
)

first_forecast = forecast_df.iloc[0]
last_forecast = forecast_df.iloc[-1]

first_forecast_price = float(
    first_forecast["predicted_price_per_kg"]
)

change = first_forecast_price - last_actual_price
change_percent = (
    change / last_actual_price * 100
    if last_actual_price > 0
    else 0
)

forecast_cols = st.columns(3)

forecast_cols[0].metric(
    "Latest Actual Price",
    f"₱{last_actual_price:,.2f}/kg",
)

forecast_cols[1].metric(
    f"Forecast: {first_forecast['price_date']:%b %Y}",
    f"₱{first_forecast_price:,.2f}/kg",
    delta=f"{change:+.2f} ({change_percent:+.1f}%)",
)

forecast_cols[2].metric(
    f"Forecast: {last_forecast['price_date']:%b %Y}",
    f"₱{float(last_forecast['predicted_price_per_kg']):,.2f}/kg",
)


# Find the first predicted month with a higher price than
# the latest actual recorded monthly price.
increases = forecast_df[
    forecast_df["predicted_price_per_kg"] > last_actual_price
]

if not increases.empty:
    increase_row = increases.iloc[0]

    st.success(
        f"First forecast month above the latest actual price: "
        f"{increase_row['price_date']:%B %Y}, at an estimated "
        f"₱{increase_row['predicted_price_per_kg']:,.2f}/kg."
    )
else:
    st.info(
        "The model does not forecast a month above the latest "
        "actual recorded price within this selected period."
    )


st.markdown("#### Forecast Table")

st.dataframe(
    forecast_df.rename(
        columns={
            "price_date": "Forecast Month",
            "predicted_price_per_kg": "Predicted Price (₱/kg)",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

# Combine actual and forecast prices for visualization.
actual_chart = historical_df[
    ["price_date", "palay_price_per_kg"]
].rename(
    columns={
        "price_date": "Month",
        "palay_price_per_kg": "Actual Price (₱/kg)",
    }
).set_index("Month")

forecast_chart = forecast_df[
    ["price_date", "predicted_price_per_kg"]
].rename(
    columns={
        "price_date": "Month",
        "predicted_price_per_kg": "Forecast Price (₱/kg)",
    }
).set_index("Month")

combined_chart = actual_chart.join(
    forecast_chart, how="outer"
)

st.markdown("#### Actual vs. Forecast Price")
st.line_chart(combined_chart)

st.download_button(
    label="Download Forecast (CSV)",
    data=forecast_df.to_csv(index=False).encode("utf-8"),
    file_name="agritrack_palay_price_forecast.csv",
    mime="text/csv",
)

st.caption(
    "Forecasts are estimates generated by a trained machine "
    "learning model, not guaranteed future market prices. "
    "Accuracy depends on the quality, quantity, and time coverage "
    "of actual historical selling-price records."
)

# Optional training metrics display.
if METRICS_PATH.exists():
    try:
        import json

        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            metrics = json.load(file)

        with st.expander("Model training information"):
            st.json(metrics)

    except Exception:
        pass

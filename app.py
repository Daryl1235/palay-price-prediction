
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AgriTrack | Palay Price Prediction",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

APP_TITLE = "AgriTrack Palay Price Prediction"
MIN_HISTORY_MONTHS = 24
DEFAULT_HORIZON = 6

FEATURE_COLUMNS = [
    "current_price",
    "previous_price",
    "avg_3_month",
    "avg_6_month",
    "month_sin",
    "month_cos",
    "time_index",
]


# ============================================================
# 3. STYLING
# ============================================================

st.markdown(
    """
    <style>
    .main-header {
        padding: 1.2rem 1.4rem;
        border-radius: 14px;
        background: linear-gradient(120deg, #164b2c, #39834c);
        color: white;
        margin-bottom: 1.2rem;
    }

    .main-header h1 {
        color: white;
        margin-bottom: 0.35rem;
    }

    .main-header p {
        color: #edf8ef;
        margin-bottom: 0;
    }

    .info-card {
        border: 1px solid #dce7dd;
        border-radius: 12px;
        padding: 16px;
        background: #f7faf7;
        margin-bottom: 12px;
    }

    .small-note {
        color: #66756a;
        font-size: 0.9rem;
    }

    div[data-testid="stMetric"] {
        border: 1px solid #e1e8e1;
        border-radius: 10px;
        padding: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 4. API SETTINGS
# ============================================================

def get_secret(name: str, default: str = "") -> str:
    """Read configuration from Streamlit Secrets or environment."""

    try:
        value = st.secrets.get(name, default)
    except Exception:
        value = default

    if value:
        return str(value).strip()

    return os.environ.get(name, default).strip()


API_URL = get_secret("PALAY_PRICE_API_URL")
API_KEY = get_secret("PALAY_PRICE_API_KEY")


# ============================================================
# 5. LOAD SELLING PRICES FROM AGRITRACK MYSQL API
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def load_agritrack_prices(api_url: str, api_key: str):
    """
    Fetch monthly selling prices from the existing
    farm_activities table through palay_price_api.php.
    """

    if not api_url or not api_key:
        raise ValueError(
            "API configuration is missing. Configure "
            "PALAY_PRICE_API_URL and PALAY_PRICE_API_KEY "
            "in Streamlit Secrets."
        )

    response = requests.get(
        api_url,
        headers={"X-API-Key": api_key},
        timeout=30,
    )

    response.raise_for_status()

    try:
        payload = response.json()
    except ValueError as exc:
        raise ValueError(
            "The API did not return valid JSON. "
            "Check palay_price_api.php."
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError("Unexpected API response format.")

    if payload.get("success") is not True:
        raise ValueError(
            payload.get("message", "The API request failed.")
        )

    records = payload.get("data", [])

    if not isinstance(records, list):
        raise ValueError(
            "The API data field must contain a list of records."
        )

    if not records:
        return pd.DataFrame(
            columns=[
                "price_date",
                "palay_price_per_kg",
                "record_count",
            ]
        )

    df = pd.DataFrame(records)

    required_columns = [
        "price_date",
        "palay_price_per_kg",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing API columns: "
            + ", ".join(missing_columns)
        )

    df["price_date"] = pd.to_datetime(
        df["price_date"],
        errors="coerce",
    )

    df["palay_price_per_kg"] = pd.to_numeric(
        df["palay_price_per_kg"],
        errors="coerce",
    )

    if "record_count" not in df.columns:
        df["record_count"] = 1

    df["record_count"] = pd.to_numeric(
        df["record_count"],
        errors="coerce",
    ).fillna(1)

    df = df.dropna(
        subset=["price_date", "palay_price_per_kg"]
    )

    df = df[df["palay_price_per_kg"] > 0].copy()

    if df.empty:
        return df

    # Normalize every date to the first day of its month.
    df["price_date"] = (
        df["price_date"].dt.to_period("M").dt.to_timestamp()
    )

    # Combine any duplicate monthly rows safely.
    # The PHP API normally already returns monthly averages.
    df = (
        df.groupby("price_date", as_index=False)
        .agg(
            palay_price_per_kg=("palay_price_per_kg", "mean"),
            record_count=("record_count", "sum"),
        )
        .sort_values("price_date")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# 6. CHECK MONTHLY HISTORY
# ============================================================

def check_monthly_history(df: pd.DataFrame):
    """
    Do not invent missing monthly prices.
    Require consecutive monthly observations before forecasting.
    """

    if df.empty:
        return False, "No valid selling-price records were found."

    if len(df) < MIN_HISTORY_MONTHS:
        return (
            False,
            f"Only {len(df)} monthly price observations are "
            f"available. At least {MIN_HISTORY_MONTHS} "
            "consecutive months are required before this app "
            "will generate a forecast.",
        )

    periods = pd.PeriodIndex(
        df["price_date"],
        freq="M",
    )

    expected_periods = pd.period_range(
        start=periods.min(),
        end=periods.max(),
        freq="M",
    )

    missing_periods = expected_periods.difference(periods)

    if len(missing_periods) > 0:
        examples = ", ".join(
            str(period)
            for period in list(missing_periods[:5])
        )

        return (
            False,
            "Some months have no recorded selling prices. "
            f"Missing month(s): {examples}. The app will not "
            "fill those months with invented prices. Record "
            "actual selling prices for the missing months "
            "before generating a forecast.",
        )

    return True, "Monthly history is ready for model training."


# ============================================================
# 7. CREATE MODEL FEATURES
# ============================================================

def make_feature_row(
    prices: list[float],
    target_month: pd.Timestamp,
    origin_month: pd.Timestamp,
) -> dict:
    """
    Build features using only prices observed before the
    month being predicted.

    prices:
        Historical monthly prices followed by any earlier
        predictions when forecasting recursively.
    """

    target_month = pd.Timestamp(target_month)
    origin_month = pd.Timestamp(origin_month)

    if len(prices) < 6:
        raise ValueError(
            "At least six preceding monthly prices are required."
        )

    month_number = target_month.month

    month_sin = np.sin(
        2 * np.pi * month_number / 12
    )

    month_cos = np.cos(
        2 * np.pi * month_number / 12
    )

    time_index = (
        (target_month.year - origin_month.year) * 12
        + target_month.month
        - origin_month.month
    )

    return {
        "current_price": float(prices[-1]),
        "previous_price": float(prices[-2]),
        "avg_3_month": float(np.mean(prices[-3:])),
        "avg_6_month": float(np.mean(prices[-6:])),
        "month_sin": float(month_sin),
        "month_cos": float(month_cos),
        "time_index": float(time_index),
    }


# ============================================================
# 8. BUILD TRAINING DATA FROM LIVE API HISTORY
# ============================================================

def build_training_data(
    monthly: pd.DataFrame,
):
    """
    Create supervised learning samples from the actual monthly
    prices retrieved from AgriTrack.

    No external training CSV is used by this function.
    """

    prices = monthly["palay_price_per_kg"].astype(float).tolist()

    origin_month = pd.Timestamp(
        monthly["price_date"].iloc[0]
    )

    feature_rows = []
    target_values = []

    # A target month's features use the six preceding months.
    for target_index in range(6, len(prices)):
        target_month = pd.Timestamp(
            monthly["price_date"].iloc[target_index]
        )

        historical_prices = prices[:target_index]

        feature_row = make_feature_row(
            prices=historical_prices,
            target_month=target_month,
            origin_month=origin_month,
        )

        feature_rows.append(feature_row)
        target_values.append(prices[target_index])

    X = pd.DataFrame(
        feature_rows,
        columns=FEATURE_COLUMNS,
    )

    y = np.asarray(target_values, dtype=float)

    return X, y, origin_month


# ============================================================
# 9. EVALUATE MODEL USING A CHRONOLOGICAL HOLDOUT
# ============================================================

def evaluate_model(X: pd.DataFrame, y: np.ndarray):
    """
    Compare Linear Regression against a last-observed-price
    baseline on the latest portion of the historical samples.
    """

    sample_count = len(X)

    if sample_count < 5:
        return None

    test_count = max(1, int(np.ceil(sample_count * 0.20)))

    if sample_count - test_count < 3:
        return None

    split_index = sample_count - test_count

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    evaluation_model = LinearRegression()
    evaluation_model.fit(X_train, y_train)

    predictions = evaluation_model.predict(X_test)

    # The current_price feature is the latest observed price.
    baseline_predictions = X_test["current_price"].to_numpy()

    mae = float(
        mean_absolute_error(y_test, predictions)
    )

    rmse = float(
        np.sqrt(mean_squared_error(y_test, predictions))
    )

    baseline_mae = float(
        mean_absolute_error(y_test, baseline_predictions)
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "baseline_mae": baseline_mae,
        "beats_baseline": mae < baseline_mae,
        "test_months": test_count,
    }


# ============================================================
# 10. TRAIN ON ALL AVAILABLE MONTHLY HISTORY
# ============================================================

def train_live_model(monthly: pd.DataFrame):
    X, y, origin_month = build_training_data(monthly)

    if len(X) < 5:
        raise ValueError(
            "There are too few training examples. "
            "More monthly selling-price history is required."
        )

    evaluation = evaluate_model(X, y)

    model = LinearRegression()
    model.fit(X, y)

    return model, X, y, origin_month, evaluation


# ============================================================
# 11. FORECAST FUTURE MONTHS
# ============================================================

def forecast_months(
    monthly: pd.DataFrame,
    model: LinearRegression,
    origin_month: pd.Timestamp,
    horizon: int,
):
    """
    Predict the next months recursively using the model trained
    from actual AgriTrack monthly selling-price history.
    """

    historical_prices = (
        monthly["palay_price_per_kg"].astype(float).tolist()
    )

    last_observed_month = pd.Timestamp(
        monthly["price_date"].iloc[-1]
    )

    forecast_rows = []

    for step in range(1, horizon + 1):
        target_month = (
            last_observed_month
            + pd.DateOffset(months=step)
        )

        feature_row = make_feature_row(
            prices=historical_prices,
            target_month=target_month,
            origin_month=origin_month,
        )

        feature_frame = pd.DataFrame(
            [feature_row],
            columns=FEATURE_COLUMNS,
        )

        predicted_price = float(
            model.predict(feature_frame)[0]
        )

        # A price cannot be negative.
        predicted_price = max(0.0, predicted_price)

        previous_price = historical_prices[-1]

        change = predicted_price - previous_price

        if change > 0.005:
            direction = "Possible increase"
        elif change < -0.005:
            direction = "Possible decrease"
        else:
            direction = "Little or no change"

        forecast_rows.append(
            {
                "Forecast Month": target_month,
                "Predicted Price (PHP/kg)": round(
                    predicted_price, 2
                ),
                "Previous Month Price (PHP/kg)": round(
                    previous_price, 2
                ),
                "Change (PHP/kg)": round(change, 2),
                "Direction": direction,
            }
        )

        # Use the forecast as the latest value for the next step.
        historical_prices.append(predicted_price)

    return pd.DataFrame(forecast_rows)


# ============================================================
# 12. HEADER
# ============================================================

st.markdown(
    """
    <div class="main-header">
        <h1>🌾 AgriTrack Palay Price Prediction</h1>
        <p>
            Analyze recorded palay selling prices and explore
            possible future monthly price trends.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 13. SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Prediction Settings")

    horizon = st.selectbox(
        "Forecast period",
        options=[3, 6, 12],
        index=1,
        format_func=lambda value: f"Next {value} months",
    )

    st.divider()

    st.markdown("### Data source")
    st.caption("AgriTrack MySQL → farm_activities")

    st.markdown("### Model")
    st.caption("Linear Regression")

    st.markdown("### Data requirement")
    st.caption(
        f"At least {MIN_HISTORY_MONTHS} consecutive months "
        "of actual selling-price observations."
    )

    refresh_clicked = st.button(
        "Refresh selling prices",
        use_container_width=True,
    )

    if refresh_clicked:
        load_agritrack_prices.clear()
        st.rerun()


# ============================================================
# 14. LOAD API DATA
# ============================================================

try:
    with st.spinner(
        "Loading selling prices from AgriTrack..."
    ):
        monthly = load_agritrack_prices(
            API_URL,
            API_KEY,
        )

except requests.exceptions.Timeout:
    st.error(
        "The AgriTrack API took too long to respond. "
        "Please try refreshing the selling prices."
    )
    st.stop()

except requests.exceptions.HTTPError as exc:
    status_code = (
        exc.response.status_code
        if exc.response is not None
        else "unknown"
    )

    if status_code == 401:
        st.error(
            "API authentication failed. Check that the API key "
            "in Streamlit Secrets matches api_config.php."
        )
    elif status_code == 500:
        st.error(
            "The PHP API encountered a server error. "
            "Check the SQL query and PHP error log."
        )
    else:
        st.error(
            f"The API returned HTTP status {status_code}."
        )

    st.stop()

except requests.exceptions.RequestException:
    st.error(
        "Unable to connect to the AgriTrack API. "
        "Check the API URL and your internet connection."
    )
    st.stop()

except ValueError as exc:
    st.error(str(exc))
    st.stop()

except Exception as exc:
    st.error(
        "An unexpected error occurred while loading prices: "
        f"{exc}"
    )
    st.stop()


st.success(
    "Connected to the AgriTrack selling-price API."
)


# ============================================================
# 15. HANDLE EMPTY DATA
# ============================================================

if monthly.empty:
    st.warning(
        "No valid selling prices were returned by the API. "
        "Check that farmers have saved Post-Harvest selling "
        "records containing a positive price_per_kg value."
    )

    st.info(
        "Once real selling records are available, they will "
        "appear here automatically after the next refresh."
    )

    st.stop()


# ============================================================
# 16. DISPLAY HISTORY SUMMARY
# ============================================================

first_month = pd.Timestamp(monthly["price_date"].min())
last_month = pd.Timestamp(monthly["price_date"].max())

total_records = int(monthly["record_count"].sum())

latest_price = float(
    monthly["palay_price_per_kg"].iloc[-1]
)

average_price = float(
    monthly["palay_price_per_kg"].mean()
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Monthly Observations",
    f"{len(monthly)}",
)

col2.metric(
    "Latest Recorded Price",
    f"₱{latest_price:,.2f}/kg",
)

col3.metric(
    "Average Monthly Price",
    f"₱{average_price:,.2f}/kg",
)

col4.metric(
    "Selling Records",
    f"{total_records:,}",
)

st.caption(
    f"Observed monthly history: "
    f"{first_month.strftime('%B %Y')} to "
    f"{last_month.strftime('%B %Y')}."
)


# ============================================================
# 17. DISPLAY ACTUAL SELLING PRICE HISTORY
# ============================================================

st.subheader("Recorded Palay Selling Prices")

st.write(
    "These monthly averages come from farmers' saved "
    "Post-Harvest selling records in AgriTrack."
)

history_display = monthly.rename(
    columns={
        "price_date": "Month",
        "palay_price_per_kg": "Average Price (PHP/kg)",
        "record_count": "Selling Records",
    }
).copy()

history_display["Month"] = (
    pd.to_datetime(history_display["Month"])
    .dt.strftime("%B %Y")
)

history_display["Average Price (PHP/kg)"] = (
    history_display["Average Price (PHP/kg)"]
    .round(2)
)

history_display["Selling Records"] = (
    history_display["Selling Records"].astype(int)
)

st.dataframe(
    history_display,
    use_container_width=True,
    hide_index=True,
)

st.line_chart(
    monthly.set_index("price_date")[
        ["palay_price_per_kg"]
    ].rename(
        columns={
            "palay_price_per_kg": "Recorded Price (PHP/kg)"
        }
    ),
    use_container_width=True,
)


# ============================================================
# 18. DATA SUFFICIENCY CHECK
# ============================================================
increase_df = forecast_df[
    forecast_df["direction"] == "Increase"
]

first_increase = (
    increase_df.iloc[0]
    if not increase_df.empty
    else None
)

if first_increase is not None:
    st.success(
        f"The first predicted increase is in "
        f"{first_increase['forecast_month'].strftime('%B %Y')}: "
        f"₱{first_increase['predicted_price']:,.2f}/kg, "
        f"up by approximately "
        f"₱{first_increase['price_change']:,.2f}/kg "
        "from the preceding month."
    )
else:
    st.info(
        "The model did not predict a monthly increase "
        "within the selected forecast period."
    )
# ============================================================
# 19. FORECAST SUMMARY
# ============================================================

first_forecast = forecast_df.iloc[0]

forecast_col1, forecast_col2, forecast_col3 = st.columns(3)

with forecast_col1:
    st.metric(
        "Next Month's Forecast",
        f"₱{first_forecast['predicted_price']:,.2f}/kg",
        delta=(
            f"₱{first_forecast['price_change']:+,.2f}/kg"
        ),
    )

with forecast_col2:
    st.metric(
        "Forecast Month",
        first_forecast[
            "forecast_month"
        ].strftime("%B %Y"),
    )

with forecast_col3:
    st.metric(
        "Forecast Direction",
        first_forecast["direction"],
    )
# ============================================================
# 20. FORECAST CHART
# ============================================================

st.subheader("Recorded Prices vs. Predicted Prices")

observed_chart = monthly_df[
    ["month", "palay_price_per_kg"]
].copy()

observed_chart = observed_chart.rename(
    columns={
        "month": "Month",
        "palay_price_per_kg": "Recorded Price (₱/kg)",
    }
)

observed_chart = observed_chart.set_index("Month")

predicted_chart = forecast_df[
    ["forecast_month", "predicted_price"]
].copy()

predicted_chart = predicted_chart.rename(
    columns={
        "forecast_month": "Month",
        "predicted_price": "Predicted Price (₱/kg)",
    }
)

predicted_chart = predicted_chart.set_index("Month")

combined_chart = observed_chart.join(
    predicted_chart,
    how="outer",
)

st.line_chart(
    combined_chart,
    y=[
        "Recorded Price (₱/kg)",
        "Predicted Price (₱/kg)",
    ],
)
# ============================================================
# 21. FORECAST TABLE
# ============================================================

st.subheader("Monthly Forecast Results")

display_forecast = forecast_df.copy()

display_forecast["forecast_month"] = (
    display_forecast["forecast_month"]
    .dt.strftime("%B %Y")
)

display_forecast = display_forecast.rename(
    columns={
        "forecast_month": "Forecast Month",
        "predicted_price": "Predicted Price (₱/kg)",
        "previous_month_price": "Previous Price (₱/kg)",
        "price_change": "Price Change (₱/kg)",
        "direction": "Expected Direction",
    }
)

for column in [
    "Predicted Price (₱/kg)",
    "Previous Price (₱/kg)",
    "Price Change (₱/kg)",
]:
    display_forecast[column] = display_forecast[
        column
    ].map(lambda value: f"₱{value:,.2f}")

st.dataframe(
    display_forecast,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# 22. DOWNLOAD FORECAST
# ============================================================

download_df = forecast_df.copy()

download_df["forecast_month"] = (
    download_df["forecast_month"]
    .dt.strftime("%Y-%m-01")
)

forecast_csv = download_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Forecast Results (CSV)",
    data=forecast_csv,
    file_name="agritrack_palay_price_forecast.csv",
    mime="text/csv",
)
# ============================================================
# 23. DISCLAIMER
# ============================================================

st.divider()

st.caption(
    "Disclaimer: Forecasts are model estimates, not guaranteed "
    "market prices. Actual palay prices can vary by location, "
    "quality, moisture content, buyer, season, supply, and demand. "
    "Forecast quality depends on the amount and quality of "
    "historical training data."
)

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    BASE_DIR / "palay_price_prediction_dataset.csv",
    BASE_DIR / "palay_price_prediction_training(1).csv",
    BASE_DIR / "palay_price_prediction_training.csv",
]
MODEL_PATH = BASE_DIR / "palay_price_model.joblib"
METRICS_PATH = BASE_DIR / "model_metrics.json"


def load_monthly_prices():
    data_path = next((path for path in DATA_CANDIDATES if path.exists()), None)
    if data_path is None:
        raise FileNotFoundError(
            "CSV not found. Place palay_price_prediction_dataset.csv "
            "beside train_model.py."
        )

    df = pd.read_csv(data_path)
    required = {"price_date", "palay_price_per_kg"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"CSV must contain these columns: {sorted(required)}"
        )

    df["price_date"] = pd.to_datetime(df["price_date"], errors="coerce")
    df["palay_price_per_kg"] = pd.to_numeric(
        df["palay_price_per_kg"], errors="coerce"
    )
    df = df.dropna(subset=["price_date", "palay_price_per_kg"])
    df = df[df["palay_price_per_kg"] >= 0].copy()
    df["month_start"] = (
        df["price_date"].dt.to_period("M").dt.to_timestamp()
    )

    monthly = (
        df.groupby("month_start", as_index=False)["palay_price_per_kg"]
        .mean()
        .sort_values("month_start")
        .reset_index(drop=True)
    )

    if len(monthly) < 24:
        raise ValueError(
            f"Only {len(monthly)} valid months found. "
            "At least 24 monthly prices are recommended."
        )

    # Fill missing calendar months by time interpolation so lag features
    # always refer to consecutive months. Interpolated values are not
    # actual farmer observations, so keep this limitation in mind.
    full_months = pd.date_range(
        monthly["month_start"].min(),
        monthly["month_start"].max(),
        freq="MS",
    )
    monthly = (
        monthly.set_index("month_start")
        .reindex(full_months)
        .rename_axis("month_start")
        .reset_index()
    )
    monthly["palay_price_per_kg"] = (
        monthly["palay_price_per_kg"].interpolate(limit_direction="both")
    )

    return monthly, data_path.name


def make_feature_row(history, target_date, origin_date):
    values = np.asarray(history, dtype=float)
    target_date = pd.Timestamp(target_date)
    origin_date = pd.Timestamp(origin_date)
    time_index = (
        (target_date.year - origin_date.year) * 12
        + target_date.month - origin_date.month
    )

    return {
        "current_price": float(values[-1]),
        "previous_price": float(values[-2]) if len(values) >= 2 else float(values[-1]),
        "avg_3_month": float(np.mean(values[-3:])),
        "avg_6_month": float(np.mean(values[-6:])),
        "month_sin": float(np.sin(2 * np.pi * target_date.month / 12)),
        "month_cos": float(np.cos(2 * np.pi * target_date.month / 12)),
        "time_index": int(time_index),
    }


def build_supervised(monthly):
    prices = monthly["palay_price_per_kg"].astype(float).tolist()
    dates = monthly["month_start"].tolist()
    origin_date = pd.Timestamp(dates[0])
    rows, targets, target_dates = [], [], []

    for i in range(1, len(prices)):
        rows.append(make_feature_row(prices[:i], dates[i], origin_date))
        targets.append(prices[i])
        target_dates.append(dates[i])

    return (
        pd.DataFrame(rows),
        pd.Series(targets, name="target_price"),
        pd.Series(target_dates),
        origin_date,
    )


def main():
    monthly, source_name = load_monthly_prices()
    X, y, target_dates, origin_date = build_supervised(monthly)

    test_size = min(12, max(6, len(X) // 5))
    split = len(X) - test_size
    if split < 2:
        raise ValueError("Not enough monthly history for a time-ordered test.")

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    evaluation_model = LinearRegression()
    evaluation_model.fit(X_train, y_train)
    test_predictions = evaluation_model.predict(X_test)

    mae = float(mean_absolute_error(y_test, test_predictions))
    rmse = float(np.sqrt(mean_squared_error(y_test, test_predictions)))
    baseline_mae = float(
        mean_absolute_error(y_test, X_test["current_price"])
    )

    final_model = LinearRegression()
    final_model.fit(X, y)

    bundle = {
        "model": final_model,
        "feature_columns": list(X.columns),
        "origin_date": str(origin_date.date()),
        "last_training_date": str(monthly["month_start"].iloc[-1].date()),
        "last_training_price": float(monthly["palay_price_per_kg"].iloc[-1]),
        "training_rows": int(len(X)),
        "monthly_price_rows": int(len(monthly)),
        "source_csv": source_name,
        "model_type": "LinearRegression",
    }
    joblib.dump(bundle, MODEL_PATH)

    metrics = {
        "model_type": "LinearRegression",
        "source_csv": source_name,
        "monthly_price_rows": int(len(monthly)),
        "supervised_rows": int(len(X)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_period_start": str(target_dates.iloc[split].date()),
        "test_period_end": str(target_dates.iloc[-1].date()),
        "mae_pesos_per_kg": round(mae, 4),
        "rmse_pesos_per_kg": round(rmse, 4),
        "persistence_baseline_mae_pesos_per_kg": round(baseline_mae, 4),
        "model_beats_baseline": bool(mae < baseline_mae),
        "warning": (
            "Historical test metrics do not guarantee future accuracy. "
            "Retrain after adding verified observed prices."
        ),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Training complete.")
    print(f"CSV: {source_name}")
    print(f"Monthly prices: {len(monthly)} | supervised examples: {len(X)}")
    print(f"Time-test MAE: ₱{mae:.2f}/kg | RMSE: ₱{rmse:.2f}/kg")
    print(f"Last-price baseline MAE: ₱{baseline_mae:.2f}/kg")
    if mae >= baseline_mae:
        print("WARNING: Model did not beat the last-price baseline on this test.")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()

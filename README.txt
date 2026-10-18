AgriTrack Palay Price Prediction - Render package

Files:
- app.py: automatically loads latest month from CSV and shows forecast/table/chart.
- train_model.py: trains LinearRegression model and writes model metrics.
- requirements.txt: required Python packages.

Required source CSV:
- palay_price_prediction_dataset.csv
  OR palay_price_prediction_training(1).csv
  OR palay_price_prediction_training.csv

Required CSV columns:
- price_date
- palay_price_per_kg

Local steps:
1. Install dependencies: pip install -r requirements.txt
2. Train: python train_model.py
3. Run: streamlit run app.py

Render:
- Build command: pip install -r requirements.txt
- Start command: streamlit run app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true

Important limitation:
This version uses the CSV supplied with the app. It does not yet query the
AgriTrack MySQL database or select a farm plot/crop cycle. The SQL schema
provided so far has farms and crop_cycles, but no dedicated verified monthly
palay price history table was identified. Add/identify a price history table and
securely connect it before implementing per-farm/cycle selectors. Do not use
predicted values as actual training observations.

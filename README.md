# Karnataka onion and tomato price forecasts: dashboard

A Streamlit dashboard that shows the results of the forecasting project. It only **reads** the result
files in `output/`. It never retrains a model and never loads the large files in `output/models/`.

## Install and run

```
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL that Streamlit prints (usually http://localhost:8501).

To read the results from another folder, change `OUTPUT_DIR` at the top of `utils/load_data.py`
(a path relative to this folder, or an absolute path).

## Pages

- **Home** (`app.py`): summary cards and a "what to keep in mind" box, calculated from the files.
- **Price Forecast** (`pages/1_Price_Forecast.py`): actual against predicted prices, forecast error,
  and a metrics table against the naive rule, with sidebar filters.
- **Model Comparison** (`pages/2_Model_Comparison.py`): tabs for all results, winners, weather with and
  without, results by district, feature importance and charts.

## Files it reads (inside `output/`)

`model_comparison.csv`, `predictions_<level>_<commodity>_h<N>.csv`, `feature_importance_<commodity>_h<N>.csv`,
`baseline_scores.csv`, `training_report.txt`, `model_data_state_<commodity>.csv`,
`model_data_district_<commodity>.csv`. If a file is missing the page says which one and carries on.

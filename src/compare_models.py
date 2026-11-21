
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

data_dir = Path("data/processed")
model_dir = Path("models")

# Load the common test set
test_data = np.load(data_dir / "test_sequences.npz")
X_test = test_data["X"]
y_test_scaled = test_data["y"]

# Load the scaler
scaler = joblib.load(model_dir / "scaler.pkl")
sales_scale = scaler.scale_[0]
sales_offset = scaler.min_[0]

# Convert scaled targets back to sales units
y_true = (y_test_scaled - sales_offset) / sales_scale

results = []

def evaluate(name, predictions_scaled):
    predictions = (
        predictions_scaled - sales_offset
    ) / sales_scale

    mae = mean_absolute_error(
        y_true.ravel(), predictions.ravel()
    )
    rmse = np.sqrt(mean_squared_error(
        y_true.ravel(), predictions.ravel()
    ))

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse
    })

# LSTM
lstm = load_model(model_dir / "lstm_demand_forecasting.keras")
lstm_pred = lstm.predict(X_test, verbose=0)
evaluate("LSTM", lstm_pred)

# GRU
gru = load_model(model_dir / "gru_demand_forecasting.keras")
gru_pred = gru.predict(X_test, verbose=0)
evaluate("GRU", gru_pred)

# Seasonal naive: repeat the previous week's sales
# Sales is feature index 0; the last 7 inputs are days t-7 through t-1.
naive_pred = X_test[:, -7:, 0]
evaluate("Seasonal Naive", naive_pred)

# Display and save results
results_df = pd.DataFrame(results).sort_values("MAE")
print("\nModel Comparison on the Same Test Set")
print(results_df.round(3).to_string(index=False))

output_path = data_dir / "model_comparison.csv"
results_df.to_csv(output_path, index=False)
print(f"\nResults saved to: {output_path}")


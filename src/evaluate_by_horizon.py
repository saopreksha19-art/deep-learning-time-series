
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

data_dir = Path("data/processed")
model_dir = Path("models")

# Load test sequences
test_data = np.load(data_dir / "test_sequences.npz")
X_test = test_data["X"]
y_test_scaled = test_data["y"]

# Load the saved model and scaler
model = load_model(model_dir / "gru_demand_forecasting.keras")
scaler = joblib.load(model_dir / "scaler.pkl")

print("Test input shape:", X_test.shape)
print("Test target shape:", y_test_scaled.shape)

# Generate predictions once
y_pred_scaled = model.predict(X_test, verbose=0)

# Inverse-transform sales using the fitted scaler
sales_scale = scaler.scale_[0]
sales_offset = scaler.min_[0]

y_test = (y_test_scaled - scaler.min_[0]) / scaler.scale_[0]
y_pred = (y_pred_scaled - scaler.min_[0]) / scaler.scale_[0]

# Overall metrics across all samples and forecast days
overall_mae = mean_absolute_error(
    y_test.ravel(), y_pred.ravel()
)
overall_rmse = np.sqrt(mean_squared_error(
    y_test.ravel(), y_pred.ravel()
))

print("\nOverall Test Results")
print(f"MAE:  {overall_mae:.3f} units")
print(f"RMSE: {overall_rmse:.3f} units")

# Metrics for each forecast day
results = []

for day in range(y_test.shape[1]):
    actual = y_test[:, day]
    predicted = y_pred[:, day]

    results.append({
        "forecast_day": day + 1,
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted))
    })

metrics_df = pd.DataFrame(results)

print("\nMetrics by Forecast Horizon")
print(metrics_df.round(3).to_string(index=False))

output_path = data_dir / "horizon_metrics.csv"
metrics_df.to_csv(output_path, index=False)

print(f"\nMetrics saved to: {output_path}")


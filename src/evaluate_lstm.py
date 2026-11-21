
import numpy as np
import joblib
from pathlib import Path
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Project paths
data_dir = Path("data/processed")
model_dir = Path("models")

# Load test sequences
test_data = np.load(data_dir / "test_sequences.npz")
X_test = test_data["X"]
y_test = test_data["y"]

print("Test input shape:", X_test.shape)
print("Test target shape:", y_test.shape)

# Load the previously trained model and scaler
model = load_model(model_dir / "lstm_demand_forecasting.keras")
scaler = joblib.load(model_dir / "scaler.pkl")

# Generate predictions (still scaled)
y_pred_scaled = model.predict(X_test, verbose=0)

# Sales is feature 0 in the scaler
sales_scale = scaler.scale_[0]
sales_offset = scaler.min_[0]

# Convert scaled values back to actual sales units
y_test_actual = (y_test - sales_offset) / sales_scale
y_pred_actual = (y_pred_scaled - sales_offset) / sales_scale

# Overall evaluation across all 7 forecast days
mae = mean_absolute_error(
    y_test_actual.ravel(),
    y_pred_actual.ravel()
)

rmse = np.sqrt(
    mean_squared_error(
        y_test_actual.ravel(),
        y_pred_actual.ravel()
    )
)

print("\nFinal Test Results")
print(f"MAE:  {mae:.3f} units")
print(f"RMSE: {rmse:.3f} units")

# Evaluate each forecast horizon separately
print("\nMetrics by Forecast Horizon")
for day in range(y_test_actual.shape[1]):
    day_mae = mean_absolute_error(
        y_test_actual[:, day],
        y_pred_actual[:, day]
    )
    day_rmse = np.sqrt(
        mean_squared_error(
            y_test_actual[:, day],
            y_pred_actual[:, day]
        )
    )
    print(f"Day {day + 1}: MAE={day_mae:.3f}, RMSE={day_rmse:.3f}")


import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==========================================
# 1. Load test sequences
# ==========================================

data_dir = Path("data/processed")

test_data = np.load(data_dir / "test_sequences.npz")

X_test = test_data["X"]
y_test = test_data["y"]

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# ==========================================
# 2. Extract sales from the input window
# ==========================================

# Feature 0 is sales, as confirmed by our scaler.
# X_test contains scaled sales values.
history_sales = X_test[:, :, 0]

# ==========================================
# 3. Generate a 7-day seasonal forecast
# ==========================================

# The input window has 28 days.
# Its final 7 days correspond to the same weekdays
# as the next 7 days, assuming consecutive daily data.
y_pred_scaled = history_sales[:, -7:]

# ==========================================
# 4. Convert sales back to actual units
# ==========================================

# Sales was MinMax-scaled using:
# scaled_value = actual_value * scale + offset
import joblib

scaler = joblib.load("models/scaler.pkl")

sales_scale = scaler.scale_[0]
sales_offset = scaler.min_[0]

y_test_actual = (y_test - sales_offset) / sales_scale
y_pred_actual = (y_pred_scaled - sales_offset) / sales_scale

# ==========================================
# 5. Evaluate the baseline
# ==========================================

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

print("\nSeasonal Naive Baseline Results")
print(f"MAE:  {mae:.3f} units")
print(f"RMSE: {rmse:.3f} units")

# ==========================================
# 6. Compare with the LSTM
# ==========================================

lstm_mae = 5.082
lstm_rmse = 7.994

print("\nModel Comparison")
print(f"{'Model':<20}{'MAE':>10}{'RMSE':>10}")
print("-" * 40)
print(f"{'Seasonal Naive':<20}{mae:>10.3f}{rmse:>10.3f}")
print(f"{'LSTM':<20}{lstm_mae:>10.3f}{lstm_rmse:>10.3f}")

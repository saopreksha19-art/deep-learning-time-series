
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error

# 1. Paths and configuration
data_dir = Path("data/processed")
model_dir = Path("models")

LOOKBACK = 28
HORIZON = 7
sample_indices = [0, 100, 500, 1000, 2000]

# 2. Load test sequences, model, and scaler
test_data = np.load(data_dir / "test_sequences.npz")
X_test = test_data["X"]
y_test = test_data["y"]

model = load_model(model_dir / "gru_demand_forecasting.keras")
scaler = joblib.load(model_dir / "scaler.pkl")

# 3. Generate predictions and restore sales units
y_pred_scaled = model.predict(X_test, verbose=0)

sales_scale = scaler.scale_[0]
sales_offset = scaler.min_[0]

y_actual = (y_test - scaler.min_[0]) / scaler.scale_[0]
y_pred = (y_pred_scaled - scaler.min_[0]) / scaler.scale_[0]

# 4. Reconstruct test-sequence metadata in the SAME order
# used by create_sequences.py.
test_df = pd.read_csv(
    data_dir / "test_scaled.csv",
    parse_dates=["date"]
)

context_df = pd.concat(
    [
        pd.read_csv(
            data_dir / "train_scaled.csv",
            parse_dates=["date"]
        ),
        pd.read_csv(
            data_dir / "validation_scaled.csv",
            parse_dates=["date"]
        )
    ],
    ignore_index=True
)

metadata = []

for item_id, target_group in test_df.groupby("item_id", sort=False):
    target_group = target_group.sort_values("date").reset_index(drop=True)

    history = context_df[
        context_df["item_id"] == item_id
    ].sort_values("date").tail(LOOKBACK)

    history_length = len(history)

    if history_length < LOOKBACK:
        raise ValueError(f"Insufficient history for {item_id}")

    # Match the exact target-window loop in create_sequences.py.
    for start in range(
        history_length,
        history_length + len(target_group) - HORIZON + 1
    ):
        metadata.append({
            "item_id": item_id,
            "forecast_start": target_group.iloc[
                start - history_length
            ]["date"],
            "forecast_end": target_group.iloc[
                start - history_length + HORIZON - 1
            ]["date"]
        })

metadata_df = pd.DataFrame(metadata)

if len(metadata_df) != len(X_test):
    raise ValueError(
        f"Metadata mismatch: {len(metadata_df)} rows, "
        f"but {len(X_test)} test sequences."
    )

# 5. Plot selected test sequences with product and date labels
fig, axes = plt.subplots(
    nrows=len(sample_indices),
    ncols=1,
    figsize=(12, 19)
)

forecast_days = np.arange(1, HORIZON + 1)

for ax, idx in zip(axes, sample_indices):
    if idx >= len(X_test):
        raise IndexError(f"Sample index {idx} exceeds test set size.")

    actual = y_actual[idx]
    predicted = y_pred[idx]
    row = metadata_df.iloc[idx]

    sample_mae = mean_absolute_error(actual, predicted)

    ax.plot(
        forecast_days,
        actual,
        marker="o",
        label="Actual Sales"
    )

    ax.plot(
        forecast_days,
        predicted,
        marker="o",
        linestyle="--",
        label="GRU Forecast"
    )

    ax.set_title(
        f"Product: {row['item_id']} | "
        f"{row['forecast_start']:%Y-%m-%d} to "
        f"{row['forecast_end']:%Y-%m-%d} | "
        f"MAE: {sample_mae:.2f} units"
    )

    ax.set_xlabel("Forecast Day")
    ax.set_ylabel("Sales Units")
    ax.set_xticks(forecast_days)
    ax.grid(True, alpha=0.3)
    ax.legend()

fig.suptitle(
    "GRU: Actual vs Predicted Sales — Test Set Samples",
    fontsize=14
)

fig.tight_layout(rect=[0, 0, 1, 0.98])

output_path = model_dir / "forecast_comparison_5_samples.png"
fig.savefig(output_path, dpi=150, bbox_inches="tight")

print("Plot saved to:", output_path)
print("\nSelected test sequences:")
print(
    metadata_df.iloc[sample_indices][
        ["item_id", "forecast_start", "forecast_end"]
    ].to_string(index=True)
)

plt.show()

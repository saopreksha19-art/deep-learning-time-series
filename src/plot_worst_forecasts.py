
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error

data_dir = Path("data/processed")
model_dir = Path("models")

# Load test data and trained GRU
test_data = np.load(data_dir / "test_sequences.npz")
X_test = test_data["X"]
y_test_scaled = test_data["y"]

model = load_model(model_dir / "gru_demand_forecasting.keras")
scaler = joblib.load(model_dir / "scaler.pkl")

# Predict in scaled units, then restore original sales units
y_pred_scaled = model.predict(X_test, verbose=0)

y_true = (
    y_test_scaled - scaler.min_[0]
) / scaler.scale_[0]

y_pred = (
    y_pred_scaled - scaler.min_[0]
) / scaler.scale_[0]

# Reconstruct sequence-to-product/date mapping
test_df = pd.read_csv(
    data_dir / "test_scaled.csv",
    parse_dates=["date"]
)

context_df = pd.concat(
    [
        pd.read_csv(data_dir / "train_scaled.csv"),
        pd.read_csv(data_dir / "validation_scaled.csv")
    ],
    ignore_index=True
)

metadata = []

for item_id, group in test_df.groupby("item_id", sort=False):
    group = group.sort_values("date").reset_index(drop=True)

    history = context_df[
        context_df["item_id"] == item_id
    ].sort_values("date").tail(28)

    for start in range(len(group) - 7 + 1):
        metadata.append({
            "item_id": item_id,
            "start_date": group.iloc[start]["date"],
            "end_date": group.iloc[start + 6]["date"]
        })

metadata = pd.DataFrame(metadata)

if len(metadata) != len(y_true):
    raise ValueError(
        f"Sequence mapping mismatch: {len(metadata)} metadata rows "
        f"but {len(y_true)} sequences."
    )

# Calculate MAE and mean signed error for each forecast window
window_rows = []

for i, row in metadata.iterrows():
    actual = y_true[i]
    predicted = y_pred[i]

    window_rows.append({
        "index": i,
        "item_id": row["item_id"],
        "start_date": row["start_date"],
        "end_date": row["end_date"],
        "MAE": mean_absolute_error(actual, predicted),
        "mean_error": np.mean(predicted - actual)
    })

window_metrics = pd.DataFrame(window_rows)

# Plot the highest-MAE window for each selected product
products = ["FOODS_3_120", "FOODS_3_090"]
fig, axes = plt.subplots(2, 1, figsize=(12, 9))

for ax, item_id in zip(axes, products):
    candidates = window_metrics[
        window_metrics["item_id"] == item_id
    ]

    worst = candidates.loc[candidates["MAE"].idxmax()]
    idx = int(worst["index"])

    days = np.arange(1, 8)

    ax.plot(
        days, y_true[idx],
        marker="o", label="Actual Sales"
    )
    ax.plot(
        days, y_pred[idx],
        marker="o", linestyle="--",
        label="GRU Forecast"
    )

    ax.set_title(
        f"{item_id} | {worst['start_date']:%Y-%m-%d} to "
        f"{worst['end_date']:%Y-%m-%d} | "
        f"MAE: {worst['MAE']:.2f} | "
        f"Mean prediction error: {worst['mean_error']:.2f}"
    )
    ax.set_xlabel("Forecast Day")
    ax.set_ylabel("Sales Units")
    ax.set_xticks(days)
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()

output_path = model_dir / "worst_product_forecasts.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")

print("Plot saved to:", output_path)
print("\nWorst forecast windows:")
print(
    window_metrics[
        window_metrics["item_id"].isin(products)
    ].sort_values("MAE", ascending=False)
    .groupby("item_id")
    .head(1)
    .round(3)
    .to_string(index=False)
)

plt.show()

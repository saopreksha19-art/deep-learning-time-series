
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

# Load trained GRU and scaler
model = load_model(model_dir / "gru_demand_forecasting.keras")
scaler = joblib.load(model_dir / "scaler.pkl")

# Predict and convert sales back to original units
y_pred_scaled = model.predict(X_test, verbose=0)

sales_scale = scaler.scale_[0]
sales_offset = scaler.min_[0]

y_true = (y_test_scaled - sales_offset) / sales_scale
y_pred = (y_pred_scaled - sales_offset) / sales_scale

# Reconstruct product ID order exactly as in create_sequences.py
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

product_ids = []

for item_id, target_group in test_df.groupby("item_id", sort=False):
    target_group = target_group.sort_values("date")
    history = context_df[context_df["item_id"] == item_id]

    # Match the sequence-generation logic
    num_sequences = max(0, len(target_group) - 7 + 1)
    product_ids.extend([item_id] * num_sequences)

if len(product_ids) != len(y_true):
    raise ValueError(
        f"Alignment mismatch: {len(product_ids)} product IDs "
        f"for {len(y_true)} test sequences."
    )

# Calculate metrics separately for each product
rows = []

for item_id in dict.fromkeys(product_ids):
    indices = [
        i for i, product in enumerate(product_ids)
        if product == item_id
    ]

    actual = y_true[indices].ravel()
    predicted = y_pred[indices].ravel()

    rows.append({
        "item_id": item_id,
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "test_sequences": len(indices)
    })

results = pd.DataFrame(rows).sort_values("MAE")

output_path = data_dir / "product_metrics.csv"
results.to_csv(output_path, index=False)

print("\nBest 10 products by MAE")
print(results.head(10).round(3).to_string(index=False))

print("\nWorst 10 products by MAE")
print(results.tail(10).sort_values(
    "MAE", ascending=False
).round(3).to_string(index=False))

print(f"\nSaved product metrics to: {output_path}")

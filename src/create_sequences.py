
import numpy as np
import pandas as pd
from pathlib import Path

LOOKBACK = 28
HORIZON = 7

FEATURES = [
    "sales",
    "day_of_week",
    "day_of_month",
    "week_of_year",
    "month",
    "quarter",
    "is_weekend",
    "snap_CA",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_std_7",
    "rolling_std_28",
]


def create_sequences(context_df, target_df):
    X, y = [], []

    for item_id, target_group in target_df.groupby(
        "item_id", sort=False
    ):
        target_group = target_group.sort_values("date")

        history = context_df[
            context_df["item_id"] == item_id
        ].sort_values("date")

        # Include only observations before this split's target period.
        history = history.tail(LOOKBACK)

        combined = pd.concat(
            [history, target_group],
            ignore_index=True
        ).sort_values("date").reset_index(drop=True)

        history_length = len(history)

        if history_length < LOOKBACK:
            raise ValueError(
                f"Not enough historical rows for item {item_id}"
            )

        x_values = combined[FEATURES].to_numpy(
            dtype=np.float32
        )
        sales_values = combined["sales"].to_numpy(
            dtype=np.float32
        )

        # Forecast targets must begin inside the target split.
        for start in range(
            history_length,
            len(combined) - HORIZON + 1
        ):
            X.append(
                x_values[start - LOOKBACK:start]
            )
            y.append(
                sales_values[start:start + HORIZON]
            )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32)
    )


data_dir = Path("data/processed")

train = pd.read_csv(
    data_dir / "train_scaled.csv",
    parse_dates=["date"]
)
validation = pd.read_csv(
    data_dir / "validation_scaled.csv",
    parse_dates=["date"]
)
test = pd.read_csv(
    data_dir / "test_scaled.csv",
    parse_dates=["date"]
)

# Training sequences use only training observations.
# The first 28 training rows provide input context.
X_train, y_train = [], []

for item_id, group in train.groupby("item_id", sort=False):
    group = group.sort_values("date").reset_index(drop=True)

    x_values = group[FEATURES].to_numpy(dtype=np.float32)
    sales_values = group["sales"].to_numpy(dtype=np.float32)

    for i in range(LOOKBACK, len(group) - HORIZON + 1):
        X_train.append(x_values[i - LOOKBACK:i])
        y_train.append(sales_values[i:i + HORIZON])

X_train = np.asarray(X_train, dtype=np.float32)
y_train = np.asarray(y_train, dtype=np.float32)

# Validation sequences can use the final 28 training days.
X_val, y_val = create_sequences(
    train,
    validation
)

# Test sequences can use the final 28 days before the test period.
test_context = pd.concat(
    [train, validation],
    ignore_index=True
)
X_test, y_test = create_sequences(
    test_context,
    test
)

for name, X, y in [
    ("Train", X_train, y_train),
    ("Validation", X_val, y_val),
    ("Test", X_test, y_test),
]:
    print(f"\n{name} sequences")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

np.savez_compressed(
    data_dir / "train_sequences.npz",
    X=X_train, y=y_train
)
np.savez_compressed(
    data_dir / "validation_sequences.npz",
    X=X_val, y=y_val
)
np.savez_compressed(
    data_dir / "test_sequences.npz",
    X=X_test, y=y_test
)

print("\nSequence generation completed.")

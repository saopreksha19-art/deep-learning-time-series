
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
from pathlib import Path


# ==========================================
# 1. Load datasets
# ==========================================

train = pd.read_csv("data/processed/train.csv")
val = pd.read_csv("data/processed/validation.csv")
test = pd.read_csv("data/processed/test.csv")

print("Train shape:", train.shape)
print("Validation shape:", val.shape)
print("Test shape:", test.shape)


# ==========================================
# 2. Select numerical features
# ==========================================

features = [
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
    "rolling_std_28"
]


# ==========================================
# 3. Create and fit scaler
# ==========================================

scaler = MinMaxScaler()

scaler.fit(train[features])


# ==========================================
# 4. Scale all datasets
# ==========================================

train[features] = scaler.transform(train[features])
val[features] = scaler.transform(val[features])
test[features] = scaler.transform(test[features])


# ==========================================
# 5. Save scaled datasets and scaler
# ==========================================

output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

train.to_csv(output_dir / "train_scaled.csv", index=False)
val.to_csv(output_dir / "validation_scaled.csv", index=False)
test.to_csv(output_dir / "test_scaled.csv", index=False)

joblib.dump(scaler, "models/scaler.pkl")

print("\nScaling completed successfully!")
print("Scaled datasets and scaler saved.")

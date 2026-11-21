import pandas as pd
from pathlib import Path


# ==========================================
# 1. Load feature-engineered data
# ==========================================

sales = pd.read_csv(
    "data/processed/features.csv",
    parse_dates=["date"]
)

print("Dataset shape:", sales.shape)


# ==========================================
# 2. Sort data chronologically
# ==========================================

sales = sales.sort_values(
    ["date", "item_id"]
).reset_index(drop=True)


# ==========================================
# 3. Define chronological split dates
# ==========================================

unique_dates = sorted(sales["date"].unique())

n_dates = len(unique_dates)

train_end = int(n_dates * 0.70)
val_end = int(n_dates * 0.85)

train_dates = unique_dates[:train_end]
val_dates = unique_dates[train_end:val_end]
test_dates = unique_dates[val_end:]


# ==========================================
# 4. Create train, validation and test sets
# ==========================================

train = sales[sales["date"].isin(train_dates)].copy()

val = sales[sales["date"].isin(val_dates)].copy()

test = sales[sales["date"].isin(test_dates)].copy()


# ==========================================
# 5. Display split information
# ==========================================

for name, data in [
    ("Train", train),
    ("Validation", val),
    ("Test", test)
]:
    print(f"\n{name} set:")
    print("Rows:", len(data))
    print("Start date:", data["date"].min())
    print("End date:", data["date"].max())


# ==========================================
# 6. Save the datasets
# ==========================================

output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

train.to_csv(output_dir / "train.csv", index=False)
val.to_csv(output_dir / "validation.csv", index=False)
test.to_csv(output_dir / "test.csv", index=False)

print("\nTrain, validation and test datasets saved successfully.")

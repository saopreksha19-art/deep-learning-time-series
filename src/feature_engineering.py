import pandas as pd


# ==========================================
# 1. Load processed sales data
# ==========================================

sales = pd.read_csv(
    "data/processed/sales_long.csv"
)

print("Sales data shape:")
print(sales.shape)


# ==========================================
# 2. Load calendar data
# ==========================================

calendar = pd.read_csv(
    "data/raw/calendar.csv"
)

print("\nCalendar shape:")
print(calendar.shape)


# ==========================================
# 3. Merge sales with calendar
# ==========================================

sales = sales.merge(
    calendar[
        [
            "d",
            "date",
            "wm_yr_wk",
            "weekday",
            "wday",
            "month",
            "year",
            "event_name_1",
            "event_type_1",
            "snap_CA"
        ]
    ],
    left_on="day",
    right_on="d",
    how="left"
)

# Remove duplicate calendar key
sales.drop(columns=["d"], inplace=True)


# ==========================================
# 4. Convert date to datetime
# ==========================================

sales["date"] = pd.to_datetime(
    sales["date"]
)


# ==========================================
# 5. Create temporal features
# ==========================================

sales["day_of_week"] = sales["date"].dt.dayofweek

sales["day_of_month"] = sales["date"].dt.day

sales["week_of_year"] = sales["date"].dt.isocalendar().week.astype(int)

sales["quarter"] = sales["date"].dt.quarter

sales["is_weekend"] = (
    sales["day_of_week"] >= 5
).astype(int)


# ==========================================
# 6. Sort chronologically
# ==========================================

sales = sales.sort_values(
    ["item_id", "date"]
).reset_index(drop=True)


# ==========================================
# 7. Lag features
# ==========================================

sales["lag_1"] = (
    sales.groupby("item_id")["sales"]
    .shift(1)
)

sales["lag_7"] = (
    sales.groupby("item_id")["sales"]
    .shift(7)
)

sales["lag_14"] = (
    sales.groupby("item_id")["sales"]
    .shift(14)
)

sales["lag_28"] = (
    sales.groupby("item_id")["sales"]
    .shift(28)
)


# ==========================================
# 8. Rolling features
# ==========================================

sales["rolling_mean_7"] = (
    sales.groupby("item_id")["sales"]
    .transform(
        lambda x: x.shift(1).rolling(7).mean()
    )
)

sales["rolling_mean_14"] = (
    sales.groupby("item_id")["sales"]
    .transform(
        lambda x: x.shift(1).rolling(14).mean()
    )
)

sales["rolling_mean_28"] = (
    sales.groupby("item_id")["sales"]
    .transform(
        lambda x: x.shift(1).rolling(28).mean()
    )
)

sales["rolling_std_7"] = (
    sales.groupby("item_id")["sales"]
    .transform(
        lambda x: x.shift(1).rolling(7).std()
    )
)

sales["rolling_std_28"] = (
    sales.groupby("item_id")["sales"]
    .transform(
        lambda x: x.shift(1).rolling(28).std()
    )
)


# ==========================================
# 9. Remove rows with insufficient history
# ==========================================

sales = sales[
    sales["lag_28"].notna()
].reset_index(drop=True)


# ==========================================
# 10. Inspect result
# ==========================================

print("\nFeature engineered shape:")
print(sales.shape)

print("\nColumns:")
print(sales.columns.tolist())

print("\nFirst 5 rows:")
print(sales.head())


# ==========================================
# 11. Save feature dataset
# ==========================================

sales.to_csv(
    "data/processed/features.csv",
    index=False
)

print("\nFeature dataset saved successfully.")
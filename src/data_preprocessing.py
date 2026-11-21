import pandas as pd


# ==========================================
# 1. Load sales data
# ==========================================

sales = pd.read_csv(
    "data/raw/sales_train_validation.csv"
)

print("Original shape:")
print(sales.shape)


# ==========================================
# 2. Select CA_1 store
# ==========================================

sales = sales[
    sales["store_id"] == "CA_1"
].copy()

print("\nAfter selecting CA_1:")
print(sales.shape)


# ==========================================
# 3. Identify daily sales columns
# ==========================================

day_columns = [
    column
    for column in sales.columns
    if column.startswith("d_")
]

print("\nNumber of daily columns:")
print(len(day_columns))


# ==========================================
# 4. Select top 100 products
# ==========================================

sales["total_sales"] = sales[day_columns].sum(axis=1)

sales = (
    sales
    .sort_values("total_sales", ascending=False)
    .head(100)
    .drop(columns="total_sales")
    .copy()
)

print("\nAfter selecting top 100 products:")
print(sales.shape)


# ==========================================
# 5. Convert wide format → long format
# ==========================================

sales_long = sales.melt(
    id_vars=[
        "id",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id"
    ],
    value_vars=day_columns,
    var_name="day",
    value_name="sales"
)


# ==========================================
# 6. Inspect transformed data
# ==========================================

print("\nNew shape:")
print(sales_long.shape)

print("\nFirst 10 rows:")
print(sales_long.head(10))


# ==========================================
# 7. Save processed data
# ==========================================

sales_long.to_csv(
    "data/processed/sales_long.csv",
    index=False
)

print("\nProcessed data saved successfully.")
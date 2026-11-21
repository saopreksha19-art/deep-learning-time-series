import pandas as pd


# -------------------------------
# Load sales data
# -------------------------------

sales = pd.read_csv(
    "data/raw/sales_train_validation.csv"
)


print("Dataset shape:")
print(sales.shape)


print("\nFirst 5 rows:")
print(sales.head())


print("\nNumber of unique products:")
print(sales["item_id"].nunique())


print("\nNumber of unique stores:")
print(sales["store_id"].nunique())


print("\nFirst 10 columns:")
print(sales.columns[:10].tolist())


print("\nLast 10 columns:")
print(sales.columns[-10:].tolist())


# -------------------------------
# Load calendar data
# -------------------------------

calendar = pd.read_csv(
    "data/raw/calendar.csv"
)

print("\nCalendar shape:")
print(calendar.shape)

print("\nCalendar columns:")
print(calendar.columns.tolist())

print("\nCalendar sample:")
print(calendar.head())

day_columns = [
    column
    for column in sales.columns
    if column.startswith("d_")
]



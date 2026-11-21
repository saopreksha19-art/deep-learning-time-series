
import pandas as pd
from pathlib import Path

data_dir = Path("data/processed")

features = pd.read_csv(
    data_dir / "features.csv",
    parse_dates=["date"]
)

metrics = pd.read_csv(data_dir / "product_metrics.csv")

worst_products = metrics.nlargest(2, "MAE")["item_id"].tolist()

for item_id in worst_products:
    product = (
        features[features["item_id"] == item_id]
        .sort_values("date")
        .copy()
    )

    test_start = pd.Timestamp("2015-07-17")
    test_end = pd.Timestamp("2016-04-24")

    test_product = product[
        product["date"].between(test_start, test_end)
    ]

    print(f"\n{'=' * 55}")
    print(f"Product: {item_id}")
    print(f"{'=' * 55}")

    print("Test-period days:", len(test_product))
    print("Total test sales:", round(test_product["sales"].sum(), 2))
    print("Average daily sales:", round(test_product["sales"].mean(), 2))
    print("Sales standard deviation:", round(test_product["sales"].std(), 2))
    print("Zero-sales days:", int((test_product["sales"] == 0).sum()))
    print("Maximum daily sales:", test_product["sales"].max())

    print("\nLast 14 test-period observations:")
    print(
        test_product[["date", "sales"]]
        .tail(14)
        .to_string(index=False)
    )

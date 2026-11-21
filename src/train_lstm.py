
import numpy as np
import joblib
from pathlib import Path

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ==========================================
# 1. Load sequences
# ==========================================

data_dir = Path("data/processed")
model_dir = Path("models")
model_dir.mkdir(parents=True, exist_ok=True)

train_data = np.load(data_dir / "train_sequences.npz")
val_data = np.load(data_dir / "validation_sequences.npz")
test_data = np.load(data_dir / "test_sequences.npz")

X_train, y_train = train_data["X"], train_data["y"]
X_val, y_val = val_data["X"], val_data["y"]
X_test, y_test = test_data["X"], test_data["y"]

print("Training input:", X_train.shape)
print("Training target:", y_train.shape)
print("Validation input:", X_val.shape)
print("Test input:", X_test.shape)


# ==========================================
# 2. Build the LSTM model
# ==========================================

model = Sequential([
    Input(shape=(28, 17)),
    LSTM(64),
    Dense(32, activation="relu"),
    Dense(7)
])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

model.summary()


# ==========================================
# 3. Configure training
# ==========================================

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )
]


# ==========================================
# 4. Train the model
# ==========================================

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=256,
    callbacks=callbacks,
    verbose=1
)


# ==========================================
# 5. Save the trained model
# ==========================================

model_path = model_dir / "lstm_demand_forecasting.keras"
model.save(model_path)

print("\nModel saved to:", model_path)


# ==========================================
# 6. Evaluate on the test set
# ==========================================

y_pred_scaled = model.predict(X_test, verbose=0)

# Convert scaled sales back to actual sales units
scaler = joblib.load(model_dir / "scaler.pkl")
sales_index = [
    "sales", "day_of_week", "day_of_month", "week_of_year",
    "month", "quarter", "is_weekend", "snap_CA",
    "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_28",
    "rolling_std_7", "rolling_std_28"
].index("sales")

sales_scale = scaler.scale_[sales_index]
sales_offset = scaler.min_[sales_index]

y_test_actual = (y_test - sales_offset) / sales_scale
y_pred_actual = (y_pred_scaled - sales_offset) / sales_scale

mae = mean_absolute_error(
    y_test_actual.ravel(),
    y_pred_actual.ravel()
)

rmse = np.sqrt(mean_squared_error(
    y_test_actual.ravel(),
    y_pred_actual.ravel()
))

print("\nFinal test results:")
print(f"MAE:  {mae:.3f} units")
print(f"RMSE: {rmse:.3f} units")

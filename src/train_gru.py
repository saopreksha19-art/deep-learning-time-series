
import numpy as np
import joblib
from pathlib import Path

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error

data_dir = Path("data/processed")
model_dir = Path("models")
model_dir.mkdir(parents=True, exist_ok=True)

# Load the same sequences used by the LSTM
train_data = np.load(data_dir / "train_sequences.npz")
val_data = np.load(data_dir / "validation_sequences.npz")
test_data = np.load(data_dir / "test_sequences.npz")

X_train, y_train = train_data["X"], train_data["y"]
X_val, y_val = val_data["X"], val_data["y"]
X_test, y_test = test_data["X"], test_data["y"]

print("Training:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)

# Build the GRU model
model = Sequential([
    Input(shape=(28, 17)),
    GRU(64),
    Dense(32, activation="relu"),
    Dense(7)
])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

model.summary()

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

# Train
model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=256,
    callbacks=callbacks,
    verbose=1
)

# Save under a separate filename
model_path = model_dir / "gru_demand_forecasting.keras"
model.save(model_path)
print("\nModel saved to:", model_path)

# Evaluate on the same test set
y_pred_scaled = model.predict(X_test, verbose=0)

scaler = joblib.load(model_dir / "scaler.pkl")
sales_scale = scaler.scale_[0]
sales_offset = scaler.min_[0]

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

print("\nGRU Test Results")
print(f"MAE:  {mae:.3f} units")
print(f"RMSE: {rmse:.3f} units")


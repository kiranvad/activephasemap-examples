import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

# Step 1: Generate synthetic data
np.random.seed(42)
X = np.random.randn(1000, 10)
y = np.sum(X, axis=1) + np.random.randn(1000) * 0.1

# Step 2: Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y.reshape(-1, 1), test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

# Step 3: Initialize the TabNet Regressor
model = TabNetRegressor(
    n_d=8, n_a=8,  # Dimension of decision and attention steps
    n_steps=3,
    gamma=1.3,
    n_independent=2,
    n_shared=2,
    lambda_sparse=0.001,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=1e-3),
    mask_type='sparsemax'
)

# Step 4: Train the model
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric=['rmse'],
    max_epochs=100,
    patience=10,
    batch_size=32,
    virtual_batch_size=16,
    num_workers=0,
    drop_last=False
)

# Step 5: Make predictions and evaluate the model
y_pred = model.predict(X_val)
rmse = root_mean_squared_error(y_val, y_pred)
print(f"Validation RMSE: {rmse:.4f}")

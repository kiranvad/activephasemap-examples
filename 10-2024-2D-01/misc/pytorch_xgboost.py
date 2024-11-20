import torch
from torch.autograd import Function
import xgboost as xgb
import numpy as np

class XGBoostWrapper(Function):
    """
    Custom PyTorch function to integrate XGBoost with PyTorch autograd.
    """
    @staticmethod
    def forward(ctx, inputs, targets, params):
        # Convert PyTorch tensors to NumPy arrays
        inputs_np = inputs.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        
        # Train an XGBoost model
        dtrain = xgb.DMatrix(inputs_np, label=targets_np)
        model = xgb.train(params, dtrain)
        
        # Save the model and inputs for backward pass
        ctx.model = model
        ctx.save_for_backward(inputs, targets)
        
        # Predict and return predictions as a PyTorch tensor
        preds = model.predict(dtrain)
        return torch.tensor(preds, dtype=inputs.dtype, device=inputs.device)

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        inputs, targets = ctx.saved_tensors
        inputs_np = inputs.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        # Get feature importances from the trained model
        model = ctx.model
        dtrain = xgb.DMatrix(inputs_np, label=targets_np)
        preds = model.predict(dtrain, output_margin=True)  # Margin ensures raw scores
        
        # Calculate gradients with respect to inputs
        # Approximation: finite differences for gradient w.r.t. inputs
        epsilon = 1e-5
        grads = np.zeros_like(inputs_np)
        for i in range(inputs_np.shape[1]):
            perturbed_inputs = inputs_np.copy()
            perturbed_inputs[:, i] += epsilon
            perturbed_dmatrix = xgb.DMatrix(perturbed_inputs)
            perturbed_preds = model.predict(perturbed_dmatrix, output_margin=True)
            grads[:, i] = (perturbed_preds - preds) / epsilon
        
        # Convert to PyTorch tensor and scale by incoming gradient
        grad_input = torch.tensor(grads, dtype=inputs.dtype, device=inputs.device)
        grad_input = grad_input * grad_output.unsqueeze(1)  # Scale by chain rule
        
        return grad_input, None, None  # Gradients for inputs, targets, params

# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    X = torch.rand(100, 2, requires_grad=True)
    y = (X**2).sum(dim=1) + 0.01*torch.rand(100)
    
    # Define XGBoost parameters
    params = {
        "objective": "reg:squarederror",
        "max_depth": 3,
        "eta": 0.1
    }
    
    # Forward pass
    preds = XGBoostWrapper.apply(X, y, params)
    
    # Define a dummy loss (mean squared error)
    loss = torch.mean((preds - y) ** 2)
    print(loss)
    
    # Backward pass
    loss.backward()
    
    # Inspect gradients
    print("Gradients:", X.grad)

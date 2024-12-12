import torch
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import pdb 

# Custom autograd function to wrap a NumPy operation
class NumpyFunctionWrapper(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Save the input for backward computation
        ctx.save_for_backward(input)
        # Perform the NumPy operation
        result = np.sin(input.detach().cpu().numpy())  # Example: sin(x) in NumPy
        return torch.tensor(result, dtype=torch.float32, device=input.device)

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensor
        input, = ctx.saved_tensors
        grad_input = torch.cos(input) * grad_output
        return grad_input

# Define g(f(x)) = y where g() is a PyTorch operation
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        # Apply the NumPy-based function f() first
        f_out = NumpyFunctionWrapper.apply(x)
        # Then apply g() (a simple linear layer here)
        g_out = 2*f_out

        return g_out

# Example optimization

# Target value
target_y = torch.tensor([2.0], dtype=torch.float32)
# Initial input x
x = torch.tensor([0.0], requires_grad=True)
# Define model
model = Model()
# Loss function
loss_fn = lambda y_hat,y : (y_hat-y)**2
# Optimizer
optimizer = torch.optim.SGD([x], lr=0.01)

for step in range(100):
    optimizer.zero_grad()
    # Compute g(f(x))
    output = model(x.unsqueeze(0))
    # Compute loss
    loss = loss_fn(output, target_y)
    # Backpropagation
    loss.backward()

    # Update parameters
    optimizer.step()
    print(f"Step {step}, Loss: {loss.item()}, x: {x.item()}, dx: {x.grad}")




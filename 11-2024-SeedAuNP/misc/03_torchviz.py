import torch
from torchviz import make_dot

# Simple PyTorch model
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear1 = torch.nn.Linear(2, 4)
        self.linear2 = torch.nn.Linear(4, 1)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        return self.linear2(x)

# Initialize the model
model = SimpleModel()

# Input and target
x = torch.randn(1, 2, requires_grad=True)  # Input tensor
target = torch.tensor([1.0])  # Target value

# Forward pass
output = model(x)

# Loss computation
loss = (output - target).pow(2).mean()

# Backward pass
loss.backward()

# Visualize the gradient flow
dot = make_dot(loss, params=dict(model.named_parameters()))
dot.render("gradient_flow", format="png", cleanup=True)

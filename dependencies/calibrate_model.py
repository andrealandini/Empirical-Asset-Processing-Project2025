import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

# === Load empirical IRFs from output gap shock
print("Loading target IRFs...")
target_irf = pd.read_csv("irf_from_outputgap_shock.csv", index_col=0).iloc[[1, 2, 4, 12, 20]]
target_vector_np = target_irf.values.flatten()
target_vector = torch.tensor(target_vector_np, dtype=torch.float32)  # stay in torch
print("Target vector shape:", target_vector.shape)

# === Simulated model (placeholder version)
def simulate_model(theta):
    bπx, bππ, bπi, bix, biπ, bii = theta[:6]
    torch.manual_seed(0)
    irf = torch.tensor([
        [bix, bπx, bπi],
        [bix**2, bπx**2, bπi**2],
        [torch.sin(torch.tensor(bix)), torch.sin(torch.tensor(bπx)), torch.sin(torch.tensor(bπi))],
        [torch.exp(-torch.tensor(biπ)), torch.exp(-torch.tensor(bππ)), torch.exp(-torch.tensor(bii))],
        [biπ, bππ, bii]
    ]) + 0.01 * torch.randn(5, 3)
    return irf.flatten()

# === Define neural network surrogate
class SurrogateNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 15)
        )
    def forward(self, x):
        return self.net(x)

# === Generate training data
print("Generating training data...")
X_train = np.random.uniform(-1, 1, (1000, 6))
y_train = torch.stack([simulate_model(x).float() for x in X_train])
X_tensor = torch.tensor(X_train, dtype=torch.float32)

# === Train the neural network
print("Training neural network surrogate...")
model = SurrogateNN()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(200):
    model.train()
    pred = model(X_tensor)
    loss = loss_fn(pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

# === Define search space
search_space = [
    Real(-1.0, 1.0, name='bπx'),
    Real(-1.0, 1.0, name='bππ'),
    Real(-1.0, 1.0, name='bπi'),
    Real(-1.0, 1.0, name='bix'),
    Real(-1.0, 1.0, name='biπ'),
    Real(-1.0, 1.0, name='bii')
]

@use_named_args(search_space)
def objective(**params):
    x = torch.tensor([params[k] for k in ['bπx', 'bππ', 'bπi', 'bix', 'biπ', 'bii']], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        y_pred = model(x).squeeze()
    loss = torch.sum((y_pred - target_vector) ** 2)
    return loss.item()

# === Run Bayesian optimization
print("Running Bayesian optimization...")
res = gp_minimize(objective, search_space, n_calls=30, random_state=42)

# === Print best-fit parameters
print("\nBest parameters found:")
for name, val in zip(['bπx', 'bππ', 'bπi', 'bix', 'biπ', 'bii'], res.x):
    print(f"{name}: {val:.4f}")
print(f"Final objective loss: {res.fun:.6f}")


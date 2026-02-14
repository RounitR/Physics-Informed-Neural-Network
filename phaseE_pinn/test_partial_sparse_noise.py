import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from pinn import TrajectoryPINN


# ----------------------------
# Config
# ----------------------------
latent_dim = 8
device = torch.device("cpu")
num_sparse_points = 20
noise_std = 0.01   # try 0.01, 0.05, 0.1 later


# ----------------------------
# Load Model
# ----------------------------
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "multitraj_latent_model_phase1.pt"
)

checkpoint = torch.load(MODEL_PATH, map_location=device)

model = TrajectoryPINN(latent_dim).to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

print("Model loaded.")


# ----------------------------
# Load Unseen Dataset
# ----------------------------
DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "../phaseE_dataset/datasets/unseen_wind_ball.npy"
)

trajectories = np.load(DATA_PATH, allow_pickle=True)

# Choose different unseen trajectory by changing index
traj = trajectories[0]

print(f"Loaded unseen trajectory with {len(traj)} points")


# ----------------------------
# Preprocess
# ----------------------------
t = traj[:, 0]
x = traj[:, 1]
y = traj[:, 2]

t_min, t_max = t[0], t[-1]
T_scale = t_max - t_min
t_norm = (t - t_min) / T_scale

x_scale = np.max(np.abs(x))
y_scale = np.max(np.abs(y))

x_norm = x / x_scale
y_norm = y / y_scale


# ----------------------------
# Sparse Sampling
# ----------------------------
indices = np.sort(
    np.random.choice(len(t_norm), num_sparse_points, replace=False)
)

t_sparse = t_norm[indices]
x_sparse = x_norm[indices]
y_sparse = y_norm[indices]

# Add Gaussian noise
x_sparse_noisy = x_sparse + np.random.normal(0, noise_std, size=x_sparse.shape)
y_sparse_noisy = y_sparse + np.random.normal(0, noise_std, size=y_sparse.shape)


# ----------------------------
# Latent Optimization
# ----------------------------
z = torch.randn(1, latent_dim, requires_grad=True, device=device)

optimizer = torch.optim.Adam([z], lr=1e-2)
criterion = nn.MSELoss()

t_sparse_tensor = torch.tensor(t_sparse, dtype=torch.float32).unsqueeze(1).to(device)
target_sparse_tensor = torch.tensor(
    np.stack([x_sparse_noisy, y_sparse_noisy], axis=1),
    dtype=torch.float32
).to(device)

for step in range(2000):

    z_repeat = z.repeat(len(t_sparse_tensor), 1)
    pred_sparse = model(t_sparse_tensor, z_repeat)

    loss = criterion(pred_sparse, target_sparse_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 500 == 0:
        print(f"Step {step} | Loss: {loss.item():.6f}")

print("Latent inference complete.")


# ----------------------------
# Predict Full Trajectory
# ----------------------------
t_full_tensor = torch.tensor(t_norm, dtype=torch.float32).unsqueeze(1).to(device)
z_full = z.repeat(len(t_full_tensor), 1)

with torch.no_grad():
    pred_full = model(t_full_tensor, z_full)

pred_full = pred_full.cpu().numpy()

x_pred = pred_full[:, 0] * x_scale
y_pred = pred_full[:, 1] * y_scale


# ----------------------------
# Plot
# ----------------------------
plt.figure(figsize=(8,6))
plt.plot(x, y, label="True Full", linewidth=2)
plt.plot(x_pred, y_pred, "--", label="Predicted Full", linewidth=2)

plt.scatter(
    x_sparse_noisy * x_scale,
    y_sparse_noisy * y_scale,
    color="red",
    s=60,
    label="Low Noisy Sparse Observed"
)

plt.title("Stage E4.1 - Sparse Noisy Reconstruction")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

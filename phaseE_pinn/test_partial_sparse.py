import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from pinn import TrajectoryPINN


device = torch.device("cpu")

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "multitraj_latent_model_phase1.pt"
)

UNSEEN_PATH = os.path.join(
    os.path.dirname(__file__),
    "../phaseE_dataset/datasets/unseen_wind_ball.npy"
)


# =============================
# SETTINGS
# =============================

NUM_OBSERVED_POINTS = 10
OPT_STEPS = 2000
LR = 5e-3
ADD_NOISE = False
NOISE_STD = 0.01


# =============================
# LOAD MODEL
# =============================

checkpoint = torch.load(MODEL_PATH, map_location=device)

latent_dim = checkpoint["latent_dim"]

model = TrajectoryPINN(latent_dim).to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

print("Model loaded.")


# =============================
# LOAD UNSEEN TRAJECTORY
# =============================

unseen_data = np.load(UNSEEN_PATH, allow_pickle=True)
traj = unseen_data[4]  # choose one unseen trajectory

t = traj[:, 0]
x = traj[:, 1]
y = traj[:, 2]

print("Loaded unseen trajectory with", len(t), "points")


# =============================
# NORMALIZATION (same as training)
# =============================

t_min, t_max = t[0], t[-1]
T_scale = t_max - t_min
t_norm = (t - t_min) / T_scale

x_scale = np.max(np.abs(x))
y_scale = np.max(np.abs(y))

x_norm = x / x_scale
y_norm = y / y_scale


# =============================
# RANDOM SPARSE SAMPLING
# =============================

indices = np.random.choice(len(t_norm), NUM_OBSERVED_POINTS, replace=False)
indices.sort()

t_sparse = t_norm[indices]
x_sparse = x_norm[indices]
y_sparse = y_norm[indices]

if ADD_NOISE:
    x_sparse += np.random.normal(0, NOISE_STD, size=x_sparse.shape)
    y_sparse += np.random.normal(0, NOISE_STD, size=y_sparse.shape)

t_sparse_tensor = torch.tensor(t_sparse, dtype=torch.float32).unsqueeze(1)
target_sparse = torch.tensor(
    np.stack([x_sparse, y_sparse], axis=1),
    dtype=torch.float32
)


# =============================
# LATENT OPTIMIZATION
# =============================

z = torch.randn((1, latent_dim), requires_grad=True)
optimizer = torch.optim.Adam([z], lr=LR)
criterion = nn.MSELoss()

for step in range(OPT_STEPS):

    z_repeat = z.repeat(len(t_sparse_tensor), 1)

    pred = model(t_sparse_tensor, z_repeat)

    loss = criterion(pred, target_sparse)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 500 == 0:
        print(f"Step {step} | Loss: {loss.item():.6f}")

print("Latent inference complete.")


# =============================
# FULL TRAJECTORY PREDICTION
# =============================

t_full_tensor = torch.tensor(t_norm, dtype=torch.float32).unsqueeze(1)
z_full = z.repeat(len(t_full_tensor), 1)

pred_full = model(t_full_tensor, z_full).detach().numpy()

x_pred = pred_full[:, 0] * x_scale
y_pred = pred_full[:, 1] * y_scale


# =============================
# PLOT
# =============================

plt.figure(figsize=(8, 6))

plt.plot(x, y, label="True Full")
plt.plot(x_pred, y_pred, "--", label="Predicted Full")
plt.scatter(
    x[indices], y[indices],
    color="red",
    s=30,
    label="Sparse Observed"
)

plt.title("Stage E3.3 - Sparse Observation Reconstruction")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

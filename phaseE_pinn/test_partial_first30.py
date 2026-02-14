import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from pinn import TrajectoryPINN


def preprocess_traj(traj):
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

    return t_norm, x_norm, y_norm, T_scale, x_scale, y_scale


def main():

    device = torch.device("cpu")
    g = 9.81

    physics_weight = 0.01
    ic_weight = 1.0

    # ----------------------------
    # Load trained model
    # ----------------------------
    MODEL_PATH = os.path.join(
        os.path.dirname(__file__),
        "multitraj_latent_model_phase1.pt"
    )

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    latent_dim = checkpoint["latent_dim"]

    model = TrajectoryPINN(latent_dim).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    # ----------------------------
    # Load one unseen trajectory
    # ----------------------------
    DATA_PATH = os.path.join(
        os.path.dirname(__file__),
        "../phaseE_dataset/datasets/unseen_wind_ball.npy"
    )

    unseen = np.load(DATA_PATH, allow_pickle=True)
    traj = unseen[0]  # use first unseen trajectory

    t_norm, x_norm, y_norm, T_scale, x_scale, y_scale = preprocess_traj(traj)

    # ----------------------------
    # Extract first 30% slice
    # ----------------------------
    slice_len = int(0.3 * len(t_norm))

    t_slice = t_norm[:slice_len]
    x_slice = x_norm[:slice_len]
    y_slice = y_norm[:slice_len]

    t_tensor = torch.tensor(t_slice, dtype=torch.float32).unsqueeze(1).to(device)
    t_tensor.requires_grad_(True)

    target_slice = torch.tensor(
        np.stack([x_slice, y_slice], axis=1),
        dtype=torch.float32
    ).to(device)

    # ----------------------------
    # Latent initialization
    # ----------------------------
    z = torch.randn(1, latent_dim, requires_grad=True, device=device)

    optimizer = optim.Adam([z], lr=1e-2)
    criterion = nn.MSELoss()

    # ----------------------------
    # Optimize latent from partial slice
    # ----------------------------
    for step in range(2000):

        z_repeat = z.repeat(len(t_tensor), 1)
        pred = model(t_tensor, z_repeat)

        # Data loss (slice only)
        data_loss = criterion(pred, target_slice)

        # Physics loss (slice only)
        x_pred = pred[:, 0:1]
        y_pred = pred[:, 1:2]

        dx = torch.autograd.grad(x_pred, t_tensor, torch.ones_like(x_pred), create_graph=True)[0]
        dy = torch.autograd.grad(y_pred, t_tensor, torch.ones_like(y_pred), create_graph=True)[0]

        d2x = torch.autograd.grad(dx, t_tensor, torch.ones_like(dx), create_graph=True)[0]
        d2y = torch.autograd.grad(dy, t_tensor, torch.ones_like(dy), create_graph=True)[0]

        d2x_phys = d2x * (x_scale / (T_scale ** 2))
        d2y_phys = d2y * (y_scale / (T_scale ** 2))

        physics_loss = torch.mean(d2x_phys ** 2) + torch.mean((d2y_phys + g) ** 2)

        # IC loss if t=0 included
        ic_loss = torch.tensor(0.0, device=device)
        if t_slice[0] == 0.0:
            ic_loss = torch.mean((pred[0] - target_slice[0]) ** 2)

        loss = data_loss + physics_weight * physics_loss + ic_weight * ic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 500 == 0:
            print(f"Step {step} | Total: {loss.item():.6f} | Data: {data_loss.item():.6f}")

    print("Latent inference complete.")

    # ----------------------------
    # Reconstruct full trajectory
    # ----------------------------
    full_t_tensor = torch.tensor(t_norm, dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        z_repeat = z.repeat(len(full_t_tensor), 1)
        full_pred = model(full_t_tensor, z_repeat).cpu().numpy()

    x_pred_full = full_pred[:, 0] * x_scale
    y_pred_full = full_pred[:, 1] * y_scale

    # ----------------------------
    # Plot
    # ----------------------------
    plt.figure(figsize=(6, 5))

    # True full
    plt.plot(traj[:, 1], traj[:, 2], label="True Full")

    # Predicted full
    plt.plot(x_pred_full, y_pred_full, "--", label="Predicted Full")

    # Observed slice
    plt.scatter(traj[:slice_len, 1], traj[:slice_len, 2],
                color="red", s=20, label="Observed 30%")

    plt.title("Stage E3.1 - Partial First 30% Reconstruction")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()

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

    # Freeze model
    for p in model.parameters():
        p.requires_grad = False

    # ----------------------------
    # Load unseen dataset
    # ----------------------------
    DATA_PATH = os.path.join(
        os.path.dirname(__file__),
        "../phaseE_dataset/datasets/unseen_wind_ball.npy"
    )

    unseen = np.load(DATA_PATH, allow_pickle=True)

    print(f"Loaded {len(unseen)} unseen trajectories")

    # ----------------------------
    # Loop over unseen trajectories
    # ----------------------------
    for idx, traj in enumerate(unseen):

        print(f"\nOptimizing latent for unseen trajectory {idx}")

        t_norm, x_norm, y_norm, T_scale, x_scale, y_scale = preprocess_traj(traj)

        t_tensor = torch.tensor(t_norm, dtype=torch.float32).unsqueeze(1).to(device)
        target = torch.tensor(
            np.stack([x_norm, y_norm], axis=1),
            dtype=torch.float32
        ).to(device)

        # Create new latent vector
        z_new = torch.randn(1, latent_dim, requires_grad=True, device=device)

        optimizer = optim.Adam([z_new], lr=1e-2)
        criterion = nn.MSELoss()

        # Optimize latent only
        for step in range(2000):

            z_repeat = z_new.repeat(len(t_tensor), 1)
            pred = model(t_tensor, z_repeat)

            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 500 == 0:
                print(f"Step {step}, Loss: {loss.item():.6f}")

        # ----------------------------
        # Plot result
        # ----------------------------
        with torch.no_grad():
            z_repeat = z_new.repeat(len(t_tensor), 1)
            pred = model(t_tensor, z_repeat).cpu().numpy()

        x_pred = pred[:, 0] * x_scale
        y_pred = pred[:, 1] * y_scale

        plt.figure(figsize=(6, 5))
        plt.plot(traj[:, 1], traj[:, 2], label="True")
        plt.plot(x_pred, y_pred, "--", label="Predicted")
        plt.title(f"Unseen Trajectory {idx}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()

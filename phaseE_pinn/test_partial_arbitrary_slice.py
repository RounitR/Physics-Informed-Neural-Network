import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from pinn import TrajectoryPINN


device = torch.device("cpu")
g = 9.81


def load_model():
    MODEL_PATH = os.path.join(
        os.path.dirname(__file__),
        "multitraj_latent_model_phase1.pt"
    )

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    latent_dim = checkpoint["latent_dim"]
    model = TrajectoryPINN(latent_dim).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, latent_dim


def main():

    # ------------------------
    # Load unseen dataset
    # ------------------------
    DATA_PATH = os.path.join(
        os.path.dirname(__file__),
        "../phaseE_dataset/datasets/unseen_wind_ball.npy"
    )

    trajectories = np.load(DATA_PATH, allow_pickle=True)

    traj = trajectories[0]  # test on first unseen

    t = traj[:, 0]
    x = traj[:, 1]
    y = traj[:, 2]

    # Normalize
    t_min, t_max = t[0], t[-1]
    T_scale = t_max - t_min
    t_norm = (t - t_min) / T_scale

    x_scale = np.max(np.abs(x))
    y_scale = np.max(np.abs(y))

    x_norm = x / x_scale
    y_norm = y / y_scale

    # ------------------------
    # Select arbitrary slice
    # ------------------------
    N = len(t_norm)
    start_idx = int(0.35 * N)
    end_idx = int(0.45 * N)

    t_slice = t_norm[start_idx:end_idx]
    x_slice = x_norm[start_idx:end_idx]
    y_slice = y_norm[start_idx:end_idx]

    # ------------------------
    # Load model
    # ------------------------
    model, latent_dim = load_model()

    # ------------------------
    # Optimize latent z
    # ------------------------
    z = torch.randn(1, latent_dim, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([z], lr=1e-2)
    criterion = torch.nn.MSELoss()

    t_tensor_slice = torch.tensor(t_slice, dtype=torch.float32).unsqueeze(1).to(device)
    target_slice = torch.tensor(
        np.stack([x_slice, y_slice], axis=1),
        dtype=torch.float32
    ).to(device)

    STEPS = 2000

    for step in range(STEPS):

        z_rep = z.repeat(len(t_tensor_slice), 1)
        pred_slice = model(t_tensor_slice, z_rep)

        data_loss = criterion(pred_slice, target_slice)

        optimizer.zero_grad()
        data_loss.backward()
        optimizer.step()

        if step % 500 == 0:
            print(f"Step {step} | Loss: {data_loss.item():.6f}")

    print("Latent inference complete.")

    # ------------------------
    # Generate FULL trajectory
    # ------------------------
    t_tensor_full = torch.tensor(t_norm, dtype=torch.float32).unsqueeze(1).to(device)
    z_full = z.repeat(len(t_tensor_full), 1)

    with torch.no_grad():
        pred_full = model(t_tensor_full, z_full).cpu().numpy()

    x_pred = pred_full[:, 0] * x_scale
    y_pred = pred_full[:, 1] * y_scale

    # ------------------------
    # Plot
    # ------------------------
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label="True Full")
    plt.plot(x_pred, y_pred, "--", label="Predicted Full")
    plt.scatter(
        x[start_idx:end_idx],
        y[start_idx:end_idx],
        color="red",
        label="Observed Slice"
    )

    plt.title("Stage E3.2 - Arbitrary Slice Reconstruction")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()

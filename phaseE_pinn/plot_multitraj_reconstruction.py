import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pinn import TrajectoryPINN


def load_model():
    device = torch.device("cpu")

    MODEL_PATH = os.path.join(
        os.path.dirname(__file__),
        "multitraj_latent_model.pt"
    )

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    latent_dim = checkpoint["latent_dim"]

    model = TrajectoryPINN(latent_dim).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    embedding = torch.nn.Embedding.from_pretrained(
        checkpoint["embedding_state"]["weight"],
        freeze=True
    )

    return model, embedding, latent_dim


def preprocess(traj):
    t = traj[:, 0]
    x = traj[:, 1]
    y = traj[:, 2]

    t_min, t_max = t[0], t[-1]
    T_scale = t_max - t_min
    t_norm = (t - t_min) / T_scale

    x_scale = np.max(np.abs(x))
    y_scale = np.max(np.abs(y))

    return t_norm, x, y, T_scale, x_scale, y_scale


def plot_single(index, trajectories, model, embedding):

    device = torch.device("cpu")
    traj = trajectories[index]

    t_norm, x_true, y_true, T_scale, x_scale, y_scale = preprocess(traj)

    t_tensor = torch.tensor(t_norm, dtype=torch.float32).unsqueeze(1).to(device)

    z = embedding(torch.tensor([index]))
    z = z.repeat(len(t_tensor), 1)

    with torch.no_grad():
        pred = model(t_tensor, z).numpy()

    x_pred = pred[:, 0] * x_scale
    y_pred = pred[:, 1] * y_scale

    plt.figure(figsize=(6, 5))
    plt.plot(x_true, y_true, label="True", linewidth=2)
    plt.plot(x_pred, y_pred, '--', label="Predicted", linewidth=2)
    plt.title(f"Trajectory {index}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()


def main():

    DATA_PATH = os.path.join(
        os.path.dirname(__file__),
        "../phaseE_dataset/datasets/wind_ball.npy"
    )

    trajectories = np.load(DATA_PATH, allow_pickle=True)

    model, embedding, latent_dim = load_model()

    # ----------------------------
    # Plot original single test (index 0)
    # ----------------------------
    plot_single(0, trajectories, model, embedding)

    # ----------------------------
    # Plot 5 random seen trajectories
    # ----------------------------
    np.random.seed(42)
    random_indices = np.random.choice(len(trajectories), 5, replace=False)

    for idx in random_indices:
        plot_single(idx, trajectories, model, embedding)

    plt.show()  # Show all plots together


if __name__ == "__main__":
    main()

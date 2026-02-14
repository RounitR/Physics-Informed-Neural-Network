import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# -----------------------------
# Neural Network
# -----------------------------
class ForceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2),
        )

    def forward(self, t):
        return self.net(t)


# -----------------------------
# Main
# -----------------------------
def main():

    device = torch.device("cpu")

    # --------------------------------------------------
    # Load trajectory
    # --------------------------------------------------
    base_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(
        base_dir, "..", "phaseE_dataset", "datasets", "wind_ball.npy"
    )

    trajectories = np.load(dataset_path, allow_pickle=True)
    traj = trajectories[0]

    t = traj[:, 0]
    x = traj[:, 1]
    y = traj[:, 2]

    # normalize time
    t_norm = (t - t[0]) / (t[-1] - t[0])

    # compute accelerations
    vx = np.gradient(x, t)
    vy = np.gradient(y, t)

    ax = np.gradient(vx, t)
    ay = np.gradient(vy, t)

    g = 9.81

    fx_true = ax
    fy_true = ay + g

    # remove boundary artifacts
    mask = (t_norm > 0.02) & (t_norm < 0.98)

    t_train = torch.tensor(t_norm[mask], dtype=torch.float32).unsqueeze(1)
    f_train = torch.tensor(
        np.stack([fx_true[mask], fy_true[mask]], axis=1),
        dtype=torch.float32
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    model = ForceNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # --------------------------------------------------
    # Training
    # --------------------------------------------------
    epochs = 3000

    for epoch in range(epochs):
        optimizer.zero_grad()

        pred = model(t_train)
        loss = loss_fn(pred, f_train)

        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    print("Training complete.")

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------
    model.eval()
    with torch.no_grad():
        t_all = torch.tensor(t_norm, dtype=torch.float32).unsqueeze(1)
        pred_all = model(t_all).numpy()

    # --------------------------------------------------
    # Plot comparison
    # --------------------------------------------------
    plt.figure()
    plt.plot(t_norm, fx_true, label="True fx")
    plt.plot(t_norm, pred_all[:, 0], "--", label="Pred fx")
    plt.legend()
    plt.title("Force X")
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(t_norm, fy_true, label="True fy")
    plt.plot(t_norm, pred_all[:, 1], "--", label="Pred fy")
    plt.legend()
    plt.title("Force Y")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()

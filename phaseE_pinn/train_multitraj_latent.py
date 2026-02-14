import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pinn import TrajectoryPINN


def main():

    latent_dim = 8
    EPOCHS = 5000
    lr = 5e-4
    physics_weight = 0.01
    ic_weight = 5.0
    latent_reg_weight = 1e-4
    device = torch.device("cpu")
    g = 9.81

    # ----------------------------
    # Load dataset
    # ----------------------------
    DATA_PATH = os.path.join(
        os.path.dirname(__file__),
        "../phaseE_dataset/datasets/wind_ball.npy"
    )

    trajectories = np.load(DATA_PATH, allow_pickle=True)
    num_traj = len(trajectories)

    # ----------------------------
    # Model
    # ----------------------------
    model = TrajectoryPINN(latent_dim).to(device)
    embedding = nn.Embedding(num_traj, latent_dim).to(device)

    optimizer = optim.Adam(
        list(model.parameters()) + list(embedding.parameters()),
        lr=lr
    )

    criterion = nn.MSELoss()

    # ----------------------------
    # Preprocess
    # ----------------------------
    processed = []

    for traj in trajectories:
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

        processed.append((t_norm, x_norm, y_norm, T_scale, x_scale, y_scale))

    # ----------------------------
    # Training
    # ----------------------------
    best_loss = float("inf")
    best_state = None

    for epoch in range(EPOCHS):

        total_loss_epoch = 0
        data_loss_epoch = 0
        physics_loss_epoch = 0
        ic_loss_epoch = 0
        latent_loss_epoch = 0

        for i, (t_norm, x_norm, y_norm, T_scale, x_scale, y_scale) in enumerate(processed):

            t_tensor = torch.tensor(t_norm, dtype=torch.float32).unsqueeze(1).to(device)
            t_tensor.requires_grad_(True)

            target = torch.tensor(
                np.stack([x_norm, y_norm], axis=1),
                dtype=torch.float32
            ).to(device)

            z = embedding(torch.tensor([i]).to(device))
            z = z.repeat(len(t_tensor), 1)

            pred = model(t_tensor, z)

            # --------------------
            # Data loss
            # --------------------
            data_loss = criterion(pred, target)

            # --------------------
            # Physics residual
            # --------------------
            x_pred = pred[:, 0:1]
            y_pred = pred[:, 1:2]

            dx = torch.autograd.grad(x_pred, t_tensor, torch.ones_like(x_pred), create_graph=True)[0]
            dy = torch.autograd.grad(y_pred, t_tensor, torch.ones_like(y_pred), create_graph=True)[0]

            d2x = torch.autograd.grad(dx, t_tensor, torch.ones_like(dx), create_graph=True)[0]
            d2y = torch.autograd.grad(dy, t_tensor, torch.ones_like(dy), create_graph=True)[0]

            d2x_phys = d2x * (x_scale / (T_scale ** 2))
            d2y_phys = d2y * (y_scale / (T_scale ** 2))

            physics_loss = torch.mean(d2x_phys ** 2) + torch.mean((d2y_phys + g) ** 2)

            # --------------------
            # Initial condition anchoring
            # --------------------
            ic_loss = criterion(pred[0], target[0])

            # --------------------
            # Latent regularization
            # --------------------
            latent_reg = torch.mean(z ** 2)

            loss = (
                data_loss
                + physics_weight * physics_loss
                + ic_weight * ic_loss
                + latent_reg_weight * latent_reg
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_epoch += loss.item()
            data_loss_epoch += data_loss.item()
            physics_loss_epoch += physics_loss.item()
            ic_loss_epoch += ic_loss.item()
            latent_loss_epoch += latent_reg.item()

        # ----------------------------
        # Save best model automatically
        # ----------------------------
        if total_loss_epoch < best_loss:
            best_loss = total_loss_epoch
            best_state = {
                "model_state": model.state_dict(),
                "embedding_state": embedding.state_dict(),
                "latent_dim": latent_dim,
                "epoch": epoch,
                "loss": best_loss
            }

        if epoch % 500 == 0:
            print(
                f"Epoch {epoch} | "
                f"Total: {total_loss_epoch:.4f} | "
                f"Data: {data_loss_epoch:.4f} | "
                f"Phys: {physics_loss_epoch:.4f} | "
                f"IC: {ic_loss_epoch:.4f} | "
                f"Latent: {latent_loss_epoch:.4f}"
            )

    print("\nTraining complete.")
    print(f"Best model found at epoch {best_state['epoch']} with loss {best_state['loss']:.6f}")

    # ----------------------------
    # Save best model
    # ----------------------------
    SAVE_PATH = os.path.join(
        os.path.dirname(__file__),
        "multitraj_latent_model_phase1.pt"
    )

    torch.save(best_state, SAVE_PATH)
    print(f"Best model saved to: {SAVE_PATH}")


if __name__ == "__main__":
    main()

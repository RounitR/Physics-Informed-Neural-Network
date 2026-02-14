import os
import torch
import numpy as np

from phaseE_pinn.pinn import TrajectoryPINN


class PINNModelService:

    def __init__(self):
        self.device = torch.device("cpu")
        self.latent_dim = None
        self.model = None
        self._load_model()

    def _load_model(self):
        model_path = os.path.join(
            os.path.dirname(__file__),
            "../phaseE_pinn/multitraj_latent_model_phase1.pt"
        )

        checkpoint = torch.load(model_path, map_location=self.device)

        self.latent_dim = checkpoint["latent_dim"]
        self.model = TrajectoryPINN(self.latent_dim).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        print("PINN model loaded successfully.")

    # -------------------------------------------------------
    # Predict full trajectory from optimized latent
    # -------------------------------------------------------

    def predict_full(self, t_array, z):

        t_tensor = torch.tensor(t_array, dtype=torch.float32).unsqueeze(1)
        z_tensor = z.repeat(len(t_tensor), 1)

        with torch.no_grad():
            pred = self.model(t_tensor, z_tensor)

        pred = pred.numpy()
        return pred[:, 0], pred[:, 1]

    # -------------------------------------------------------
    # Multi-restart latent optimization
    # -------------------------------------------------------

    def infer_latent(self, t_obs, x_obs, y_obs,
                     steps=1500,
                     lr=1e-2,
                     restarts=3):

        print("\n🔍 Starting multi-restart latent optimization...")

        criterion = torch.nn.MSELoss()

        t_tensor = torch.tensor(t_obs, dtype=torch.float32).unsqueeze(1)
        target = torch.tensor(
            np.stack([x_obs, y_obs], axis=1),
            dtype=torch.float32
        )

        best_z = None
        best_loss = float("inf")

        for restart in range(restarts):

            print(f"\n--- Restart {restart+1}/{restarts} ---")

            # Random initialization like Stage E
            z = torch.randn((1, self.latent_dim), requires_grad=True)

            optimizer = torch.optim.Adam([z], lr=lr)

            for step in range(steps):

                z_expanded = z.repeat(len(t_tensor), 1)
                pred = self.model(t_tensor, z_expanded)

                loss = criterion(pred, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step % 500 == 0:
                    print(f"Restart {restart+1} | Step {step} | Loss: {loss.item():.6f}")

            final_loss = loss.item()
            print(f"Restart {restart+1} Final Loss: {final_loss:.6f}")

            if final_loss < best_loss:
                best_loss = final_loss
                best_z = z.detach().clone()

        print("\n✅ Best latent selected with loss:", best_loss)
        print("Best latent norm:", torch.norm(best_z).item())

        return best_z

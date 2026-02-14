import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    # --------------------------------------------------
    # Resolve dataset path relative to this file
    # --------------------------------------------------
    BASE_DIR = os.path.dirname(__file__)
    dataset_path = os.path.join(BASE_DIR, "datasets", "unseen_wind_ball.npy")

    trajectories = np.load(dataset_path, allow_pickle=True)

    print(f"Loaded {len(trajectories)} trajectories")
    print("Trajectory lengths:", [traj.shape[0] for traj in trajectories[:5]])

    # --------------------------------------------------
    # Plot full trajectories
    # --------------------------------------------------
    plt.figure(figsize=(8, 6))

    for traj in trajectories[:5]:
        x = traj[:, 1]
        y = traj[:, 2]
        plt.plot(x, y, alpha=0.8)

    plt.axhline(0, color="black", linewidth=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Sample Wind-Perturbed Ball Trajectories")
    plt.grid(True)

    plt.show()

    # --------------------------------------------------
    # Plot middle slice (important for Stage-E1)
    # --------------------------------------------------
    plt.figure(figsize=(8, 6))

    for traj in trajectories[:5]:
        T = traj.shape[0]
        start = int(0.4 * T)
        end = int(0.6 * T)

        x = traj[start:end, 1]
        y = traj[start:end, 2]

        plt.plot(x, y, alpha=0.8)

    plt.axhline(0, color="black", linewidth=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Middle Trajectory Slices (Identifiability Check)")
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()

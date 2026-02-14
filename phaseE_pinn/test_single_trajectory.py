import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------
    base_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(
        base_dir, "..", "phaseE_dataset", "datasets", "wind_ball.npy"
    )

    trajectories = np.load(dataset_path, allow_pickle=True)

    traj = trajectories[0]  # deterministic choice

    # Trajectory format: [t, x, y]
    t = traj[:, 0]
    x = traj[:, 1]
    y = traj[:, 2]

    # --------------------------------------------------
    # Normalize time to [0, 1]
    # --------------------------------------------------
    t0 = t[0]
    t1 = t[-1]
    t_norm = (t - t0) / (t1 - t0)

    print("Trajectory length:", len(t))
    print("Time range:", t0, "→", t1)

    # --------------------------------------------------
    # Numerical differentiation (velocity)
    # --------------------------------------------------
    vx = np.gradient(x, t)
    vy = np.gradient(y, t)

    # --------------------------------------------------
    # Plot positions
    # --------------------------------------------------
    plt.figure()
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Single Trajectory (Position)")
    plt.grid()
    plt.show()

    # --------------------------------------------------
    # Plot velocities
    # --------------------------------------------------
    plt.figure()
    plt.plot(t_norm, vx, label="vx")
    plt.plot(t_norm, vy, label="vy")
    plt.xlabel("t (normalized)")
    plt.ylabel("velocity")
    plt.title("Numerically Derived Velocities")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()

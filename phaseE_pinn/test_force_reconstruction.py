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
    # Normalize time
    # --------------------------------------------------
    t0, t1 = t[0], t[-1]
    t_norm = (t - t0) / (t1 - t0)

    # --------------------------------------------------
    # Numerical derivatives
    # --------------------------------------------------
    vx = np.gradient(x, t)
    vy = np.gradient(y, t)

    ax = np.gradient(vx, t)
    ay = np.gradient(vy, t)

    # --------------------------------------------------
    # Separate known physics
    # --------------------------------------------------
    g = 9.81

    fx = ax
    fy = ay + g

    # --------------------------------------------------
    # Plot accelerations
    # --------------------------------------------------
    plt.figure()
    plt.plot(t_norm, ax, label="ax")
    plt.plot(t_norm, ay, label="ay")
    plt.xlabel("t (normalized)")
    plt.ylabel("acceleration")
    plt.title("Raw Accelerations")
    plt.legend()
    plt.grid()
    plt.show()

    # --------------------------------------------------
    # Plot reconstructed forces
    # --------------------------------------------------
    plt.figure()
    plt.plot(t_norm, fx, label="fx (wind)")
    plt.plot(t_norm, fy, label="fy (wind)")
    plt.xlabel("t (normalized)")
    plt.ylabel("force")
    plt.title("Reconstructed Hidden Forces")
    plt.legend()
    plt.grid()
    plt.show()

    # --------------------------------------------------
    # Print force statistics
    # --------------------------------------------------
    print("Force statistics:")
    print("fx: min =", fx.min(), "max =", fx.max())
    print("fy: min =", fy.min(), "max =", fy.max())


if __name__ == "__main__":
    main()

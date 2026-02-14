import numpy as np
import matplotlib.pyplot as plt

from phaseE_controlled_physics.simulate import simulate_trajectory


def main():
    v0 = 30.0
    theta = np.deg2rad(45)
    h0 = 2.0

    gust = {
        "start": 1.0,
        "duration": 0.6,
        "magnitude": 4.0,
        "angle": np.deg2rad(20),
    }

    traj = simulate_trajectory(v0, theta, h0, gust)

    t = traj[:, 0]
    x = traj[:, 1]
    y = traj[:, 2]

    plt.figure(figsize=(7, 4))
    plt.plot(x, y, label="Trajectory with wind gust")
    plt.axhline(0, color="black", linewidth=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Wind-perturbed Ball Trajectory")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()

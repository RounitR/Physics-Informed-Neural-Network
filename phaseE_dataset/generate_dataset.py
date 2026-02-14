import os
import numpy as np

from phaseE_controlled_physics.simulate import simulate_trajectory


def main():
    np.random.seed(42)

    N = 50
    trajectories = []

    for i in range(N):
        v0 = np.random.uniform(15, 40)
        theta = np.random.uniform(np.deg2rad(20), np.deg2rad(70))
        h0 = np.random.uniform(0, 5)

        gust = {
            "start": np.random.uniform(0.5, 2.0),
            "duration": np.random.uniform(0.2, 0.8),
            "magnitude": np.random.uniform(2.0, 6.0),
            "angle": np.random.uniform(-np.pi / 3, np.pi / 3),
        }

        traj = simulate_trajectory(v0, theta, h0, gust)
        trajectories.append(traj)

    trajectories = np.array(trajectories, dtype=object)

    BASE_DIR = os.path.dirname(__file__)
    DATASET_DIR = os.path.join(BASE_DIR, "datasets")
    os.makedirs(DATASET_DIR, exist_ok=True)

    out_path = os.path.join(DATASET_DIR, "wind_ball.npy")
    np.save(out_path, trajectories, allow_pickle=True)

    print(f"Saved wind_ball.npy with {N} trajectories")
    print(f"Location: {out_path}")


if __name__ == "__main__":
    main()

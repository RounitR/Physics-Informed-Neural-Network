import os
import numpy as np
from phaseE_controlled_physics.simulate import simulate_trajectory


def main():

    unseen_trajectories = []

    # ----------------------------
    # 1. Higher velocity
    # ----------------------------
    gust1 = {
        "start": 1.0,
        "duration": 0.5,
        "magnitude": 4.0,
        "angle": 0.0,
    }

    traj1 = simulate_trajectory(
        v0=45,                         # outside [15, 40]
        theta=np.deg2rad(50),
        h0=2.0,
        gust=gust1
    )
    unseen_trajectories.append(traj1)

    # ----------------------------
    # 2. Lower angle
    # ----------------------------
    gust2 = gust1.copy()
    traj2 = simulate_trajectory(
        v0=30,
        theta=np.deg2rad(10),           # below 20°
        h0=1.0,
        gust=gust2
    )
    unseen_trajectories.append(traj2)

    # ----------------------------
    # 3. Higher angle
    # ----------------------------
    gust3 = gust1.copy()
    traj3 = simulate_trajectory(
        v0=30,
        theta=np.deg2rad(80),           # above 70°
        h0=1.0,
        gust=gust3
    )
    unseen_trajectories.append(traj3)

    # ----------------------------
    # 4. Stronger gust
    # ----------------------------
    gust4 = {
        "start": 1.0,
        "duration": 0.5,
        "magnitude": 8.0,              # > 6.0
        "angle": np.deg2rad(30),
    }

    traj4 = simulate_trajectory(
        v0=30,
        theta=np.deg2rad(45),
        h0=2.0,
        gust=gust4
    )
    unseen_trajectories.append(traj4)

    # ----------------------------
    # 5. Longer gust duration
    # ----------------------------
    gust5 = {
        "start": 1.0,
        "duration": 1.2,               # > 0.8
        "magnitude": 4.0,
        "angle": np.deg2rad(-30),
    }

    traj5 = simulate_trajectory(
        v0=30,
        theta=np.deg2rad(45),
        h0=2.0,
        gust=gust5
    )
    unseen_trajectories.append(traj5)

    unseen_trajectories = np.array(unseen_trajectories, dtype=object)

    BASE_DIR = os.path.dirname(__file__)
    DATASET_DIR = os.path.join(BASE_DIR, "datasets")
    os.makedirs(DATASET_DIR, exist_ok=True)

    out_path = os.path.join(DATASET_DIR, "unseen_wind_ball.npy")
    np.save(out_path, unseen_trajectories, allow_pickle=True)

    print("Saved unseen_wind_ball.npy with 5 trajectories")
    print("Location:", out_path)


if __name__ == "__main__":
    main()

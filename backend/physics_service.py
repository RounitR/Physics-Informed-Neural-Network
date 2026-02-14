import numpy as np
from phaseE_controlled_physics.simulate import simulate_trajectory


def generate_analytical_trajectory(
    v0,
    theta_deg,
    h0,
    gust_magnitude,
    gust_angle_deg,
    gust_start,
    gust_duration,
):

    theta = np.deg2rad(theta_deg)
    gust_angle = np.deg2rad(gust_angle_deg)

    gust = {
        "start": gust_start,
        "duration": gust_duration,
        "magnitude": gust_magnitude,
        "angle": gust_angle,
    }

    traj = simulate_trajectory(v0, theta, h0, gust)

    t = traj[:, 0]
    x = traj[:, 1]
    y = traj[:, 2]

    return t, x, y

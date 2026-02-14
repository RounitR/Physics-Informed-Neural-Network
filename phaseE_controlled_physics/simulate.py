import numpy as np
from phaseE_controlled_physics.physics import step
from phaseE_controlled_physics.wind_gust import wind_force


def simulate_trajectory(v0, theta, h0, gust, dt=0.01, t_max=10.0):
    vx0 = v0 * np.cos(theta)
    vy0 = v0 * np.sin(theta)

    state = np.array([0.0, h0, vx0, vy0])

    t = 0.0
    traj = []

    while state[1] >= 0 and t <= t_max:
        f = wind_force(t, gust)
        traj.append([t, state[0], state[1]])
        state = step(state, f, dt)
        t += dt

    return np.array(traj)

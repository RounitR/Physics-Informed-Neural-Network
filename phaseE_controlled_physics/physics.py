import numpy as np

G = 9.81

def step(state, force, dt):
    x, y, vx, vy = state
    fx, fy = force

    ax = fx
    ay = fy - G

    vx_new = vx + ax * dt
    vy_new = vy + ay * dt

    x_new = x + vx_new * dt
    y_new = y + vy_new * dt

    return np.array([x_new, y_new, vx_new, vy_new])

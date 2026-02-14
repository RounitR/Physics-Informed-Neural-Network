import numpy as np

def wind_force(t, gust):
    """
    gust = dict with keys:
      start, duration, magnitude, angle
    """
    if gust["start"] <= t <= gust["start"] + gust["duration"]:
        a = gust["magnitude"]
        phi = gust["angle"]
        return np.array([a * np.cos(phi), a * np.sin(phi)])
    return np.array([0.0, 0.0])

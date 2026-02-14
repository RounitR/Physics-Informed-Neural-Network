from pydantic import BaseModel
from typing import List


class ICRequest(BaseModel):
    v0: float
    theta_deg: float
    h0: float
    gust_magnitude: float
    gust_angle_deg: float
    gust_start: float
    gust_duration: float


class PartialRequest(BaseModel):
    # Include IC so backend can regenerate trajectory if needed
    v0: float
    theta_deg: float
    h0: float
    gust_magnitude: float
    gust_angle_deg: float
    gust_start: float
    gust_duration: float

    t_obs: List[float]
    x_obs: List[float]
    y_obs: List[float]


class TrajectoryResponse(BaseModel):
    t: List[float]
    true_x: List[float]
    true_y: List[float]
    pred_x: List[float]
    pred_y: List[float]

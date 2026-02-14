from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import numpy as np

from backend.schemas import ICRequest, PartialRequest, TrajectoryResponse
from backend.physics_service import generate_analytical_trajectory
from backend.model_service import PINNModelService


# -----------------------------------------------------
# App Initialization
# -----------------------------------------------------

app = FastAPI(root_path="")

# Load model once at startup
model_service = PINNModelService()


# -----------------------------------------------------
# Path Setup
# -----------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# -----------------------------------------------------
# Routes
# -----------------------------------------------------

@app.get("/")
def serve_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/reconstruction")
def serve_reconstruction():
    return FileResponse(os.path.join(STATIC_DIR, "reconstruction.html"))


@app.get("/ping")
def ping():
    return {"status": "ok"}


# -----------------------------------------------------
# IC-Based Generation
# -----------------------------------------------------

@app.post("/generate", response_model=TrajectoryResponse)
def generate_from_ic(request: ICRequest):

    # 1️⃣ Analytical trajectory
    t, true_x, true_y = generate_analytical_trajectory(
        request.v0,
        request.theta_deg,
        request.h0,
        request.gust_magnitude,
        request.gust_angle_deg,
        request.gust_start,
        request.gust_duration,
    )

    # 2️⃣ Compute FULL scales
    t0 = t[0]
    T_scale = t[-1] - t0

    x_scale = np.max(np.abs(true_x))
    y_scale = np.max(np.abs(true_y))

    # Normalize
    t_norm = (t - t0) / T_scale
    x_norm = true_x / x_scale
    y_norm = true_y / y_scale

    # 3️⃣ Infer latent
    z = model_service.infer_latent(t_norm, x_norm, y_norm)

    # 4️⃣ Predict
    pred_x_norm, pred_y_norm = model_service.predict_full(t_norm, z)

    # 5️⃣ De-normalize
    pred_x = pred_x_norm * x_scale
    pred_y = pred_y_norm * y_scale

    return {
        "t": t.tolist(),
        "true_x": true_x.tolist(),
        "true_y": true_y.tolist(),
        "pred_x": pred_x.tolist(),
        "pred_y": pred_y.tolist(),
    }


# -----------------------------------------------------
# Partial Reconstruction (FIXED PROPERLY)
# -----------------------------------------------------

@app.post("/reconstruct", response_model=TrajectoryResponse)
def reconstruct_from_partial(request: PartialRequest):

    # -------------------------------------------------
    # 1️⃣ Regenerate FULL analytical trajectory
    # (for correct scaling only)
    # -------------------------------------------------

    t_full_true, true_x_full, true_y_full = generate_analytical_trajectory(
        request.v0,
        request.theta_deg,
        request.h0,
        request.gust_magnitude,
        request.gust_angle_deg,
        request.gust_start,
        request.gust_duration,
    )

    # -------------------------------------------------
    # 2️⃣ Compute FULL trajectory scales
    # -------------------------------------------------

    t0 = t_full_true[0]
    T_scale = t_full_true[-1] - t0

    x_scale = np.max(np.abs(true_x_full))
    y_scale = np.max(np.abs(true_y_full))

    # -------------------------------------------------
    # 3️⃣ Normalize observed slice using FULL scale
    # -------------------------------------------------

    t_obs = np.array(request.t_obs)
    x_obs = np.array(request.x_obs)
    y_obs = np.array(request.y_obs)

    t_norm = (t_obs - t0) / T_scale
    x_norm = x_obs / x_scale
    y_norm = y_obs / y_scale

    # -------------------------------------------------
    # 4️⃣ Infer latent from slice
    # -------------------------------------------------

    z = model_service.infer_latent(t_norm, x_norm, y_norm)

    # -------------------------------------------------
    # 5️⃣ Predict full normalized trajectory
    # -------------------------------------------------

    t_full_norm = (t_full_true - t0) / T_scale

    pred_x_norm, pred_y_norm = model_service.predict_full(t_full_norm, z)

    # -------------------------------------------------
    # 6️⃣ De-normalize
    # -------------------------------------------------

    pred_x = pred_x_norm * x_scale
    pred_y = pred_y_norm * y_scale

    return {
        "t": t_full_true.tolist(),
        "true_x": true_x_full.tolist(),
        "true_y": true_y_full.tolist(),
        "pred_x": pred_x.tolist(),
        "pred_y": pred_y.tolist(),
    }

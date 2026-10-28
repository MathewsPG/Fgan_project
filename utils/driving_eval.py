import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

def evaluate_autonomous_driving_readiness(degraded_np, restored_np):
    """
    Evaluates road feature recoverability critical for autonomous vehicle perception:
    - Lane boundary edge strength (Sobel / Canny)
    - Road visual dynamic range / contrast
    - High-frequency edge preservation ratio
    """
    if HAS_CV2:
        deg_gray = cv2.cvtColor(degraded_np, cv2.COLOR_RGB2GRAY)
        res_gray = cv2.cvtColor(restored_np, cv2.COLOR_RGB2GRAY)
    else:
        # Standard luminance formula
        deg_gray = (0.299 * degraded_np[:, :, 0] + 0.587 * degraded_np[:, :, 1] + 0.114 * degraded_np[:, :, 2]).astype(np.uint8)
        res_gray = (0.299 * restored_np[:, :, 0] + 0.587 * restored_np[:, :, 1] + 0.114 * restored_np[:, :, 2]).astype(np.uint8)

    # Road Surface Region (lower half of dashcam frame)
    h, w = deg_gray.shape
    road_deg = deg_gray[int(h * 0.5):, :]
    road_res = res_gray[int(h * 0.5):, :]

    if HAS_CV2:
        edges_deg = cv2.Canny(road_deg, 50, 150)
        edges_res = cv2.Canny(road_res, 50, 150)
        edge_pixels_deg = np.count_nonzero(edges_deg)
        edge_pixels_res = np.count_nonzero(edges_res)
    else:
        # Simple gradient magnitude
        gy_deg, gx_deg = np.gradient(road_deg.astype(float))
        mag_deg = np.sqrt(gx_deg**2 + gy_deg**2)
        gy_res, gx_res = np.gradient(road_res.astype(float))
        mag_res = np.sqrt(gx_res**2 + gy_res**2)
        edge_pixels_deg = np.count_nonzero(mag_deg > 25.0)
        edge_pixels_res = np.count_nonzero(mag_res > 25.0)

    edge_gain = ((edge_pixels_res - edge_pixels_deg) / max(1, edge_pixels_deg)) * 100.0

    contrast_deg = float(np.std(deg_gray))
    contrast_res = float(np.std(res_gray))
    contrast_gain = ((contrast_res - contrast_deg) / max(1e-5, contrast_deg)) * 100.0

    readiness_score = min(100.0, max(0.0, 50.0 + (contrast_gain * 0.3) + (edge_gain * 0.2)))

    return {
        "deg_contrast": contrast_deg,
        "res_contrast": contrast_res,
        "contrast_gain_pct": contrast_gain,
        "edge_feature_gain_pct": edge_gain,
        "perception_readiness_score": readiness_score
    }

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# MARK: A4 canvas config (300 DPI)
DPI = 300
A4_WIDTH_PX = int(round(8.27 * DPI))  # 210 mm
A4_HEIGHT_PX = int(round(11.69 * DPI))  # 297 mm
MARGIN_PX = int(0.35 * DPI)  # ~9 mm safety

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

MARKER_SIZE = 280  # px (square)
BORDER_THICKNESS = 6

PATCH_ROWS = 7
PATCH_COLS = 9
PATCH_GAP = 10

GRID_TOP = MARGIN_PX + MARKER_SIZE + 80
GRID_BOTTOM = A4_HEIGHT_PX - (MARGIN_PX + MARKER_SIZE + 80)
GRID_LEFT = MARGIN_PX + MARKER_SIZE + 80
GRID_RIGHT = A4_WIDTH_PX - (MARGIN_PX + MARKER_SIZE + 80)

CHART_PNG = "calibration_chart_A4.png"
CHART_JSON = "calibration_chart_A4.json"
UV_MODEL_JSON = "uv_model.json"

INSET_FRAC_DEFAULT = 0.15
INSET_MIN_PX = 6


# MARK: INSET
def inset_rect(rect, frac=INSET_FRAC_DEFAULT, min_px=INSET_MIN_PX):
    x0, y0, x1, y1 = rect
    w = x1 - x0
    h = y1 - y0
    dx = max(int(round(w * frac)), min_px)
    dy = max(int(round(h * frac)), min_px)
    xi0, yi0, xi1, yi1 = x0 + dx, y0 + dy, x1 - dx, y1 - dy
    if xi1 <= xi0 + 2:
        m = (xi0 + xi1) // 2
        xi0, xi1 = m - 1, m + 1
    if yi1 <= yi0 + 2:
        m = (yi0 + yi1) // 2
        yi0, yi1 = m - 1, m + 1
    return [int(xi0), int(yi0), int(xi1), int(yi1)]


def _draw_sampling_visual(img_bgr, patches, out_path, color=(0, 255, 0)):
    vis = img_bgr.copy()
    for i, p in enumerate(patches, start=1):
        x0, y0, x1, y1 = p["rect"]
        cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)
        cv2.putText(vis, str(i), (x0 + 6, y0 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    cv2.imwrite(out_path, vis)


def _uv_metric_flatfield(img_bgr: np.ndarray, sigma: float = 45.0) -> np.ndarray:
    B = img_bgr[:, :, 0].astype(np.float32)
    G = img_bgr[:, :, 1].astype(np.float32)
    R = img_bgr[:, :, 2].astype(np.float32)
    uvm = B - 0.5 * (G + R)
    low = cv2.GaussianBlur(uvm, (0, 0), sigma)
    uv_flat = uvm - low
    uv_flat = (uv_flat - uv_flat.min()) / (uv_flat.max() - uv_flat.min() + 1e-9)
    return (uv_flat * 255.0).astype(np.float32)  # <-- scale to 0..255


# MARK: Calibration
# Chartless calibration:
#   - Warp both photos with ArUco if present (recommended).
#   - Build regression from a random subset of pixels over the interior.
#   - Learn UV ≈ a*R + b*G + c*B + d and write uv_model.json (1-channel).
def calibrate_free(normal_photo: str, uv_photo: str, aruco: bool = True, sample_px: int = 300_000):
    normal = cv2.imread(normal_photo, cv2.IMREAD_COLOR)
    uv = cv2.imread(uv_photo, cv2.IMREAD_COLOR)
    if normal is None or uv is None:
        raise RuntimeError("Failed to read one or both input photos.")

    if aruco:
        try:
            normal, _ = _detect_markers_and_warp(normal)
            uv, _ = _detect_markers_and_warp(uv)
            cv2.imwrite("rectified_normal.png", normal)
            cv2.imwrite("rectified_uv.png", uv)
        except Exception as e:
            print(f"[warn] ArUco warp failed ({e}); proceeding unwarped.")

    # Use inner margin to avoid borders
    h, w = normal.shape[:2]
    m = max(int(0.05 * min(h, w)), 30)
    roiN = normal[m : h - m, m : w - m]
    roiU = uv[m : h - m, m : w - m]

    # Build UV metric (0..1)
    uv_gray = _uv_metric_flatfield(roiU)

    # Flatten & random sample pixels
    R = roiN[:, :, 2].astype(np.float32).ravel()
    G = roiN[:, :, 1].astype(np.float32).ravel()
    B = roiN[:, :, 0].astype(np.float32).ravel()
    Y = (uv_gray * 255.0).ravel().astype(np.float32)

    N = R.size
    if sample_px < N:
        idx = np.random.default_rng(123).choice(N, size=sample_px, replace=False)
        R, G, B, Y = R[idx], G[idx], B[idx], Y[idx]

    X = np.stack([R, G, B, np.ones_like(R)], axis=1).astype(np.float64)
    y = Y.astype(np.float64)

    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b, c, d = theta.tolist()

    yhat = X @ theta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    print(f"[free] UV ≈ a*R + b*G + c*B + d | a={a:.6f} b={b:.6f} c={c:.6f} d={d:.6f} | R²={r2:.3f}")

    with open(UV_MODEL_JSON, "w") as f:
        json.dump({"type": "1ch", "uv_from_rgb": {"a": a, "b": b, "c": c, "d": d}, "note": "Chartless fit from pixel cloud; UV metric is blue-dominant with flat-field"}, f, indent=2)
    print(f"Wrote {UV_MODEL_JSON}")

    # Quick diagnostic images
    cv2.imwrite("free_calib_normal_crop.png", roiN)
    uv_vis = (uv_gray * 255).astype(np.uint8)
    cv2.imwrite("free_calib_uv_metric.png", uv_vis)


# MARK: ReCreate
# Recreate the exact patch rectangles the generator used,
# based on the global layout constants already in your script.
def _rebuild_chart_meta():
    grid_w = GRID_RIGHT - GRID_LEFT
    grid_h = GRID_BOTTOM - GRID_TOP
    pw = (grid_w - (PATCH_COLS - 1) * PATCH_GAP) // PATCH_COLS
    ph = (grid_h - (PATCH_ROWS - 1) * PATCH_GAP) // PATCH_ROWS

    patches = []
    idx = 0
    for r in range(PATCH_ROWS):
        for c in range(PATCH_COLS):
            x0 = GRID_LEFT + c * (pw + PATCH_GAP)
            y0 = GRID_TOP + r * (ph + PATCH_GAP)
            x1 = x0 + pw
            y1 = y0 + ph
            patches.append({"rect": [int(x0), int(y0), int(x1), int(y1)]})
            idx += 1

    return {
        "canvas_size": [A4_WIDTH_PX, A4_HEIGHT_PX],
        "margin_px": MARGIN_PX,
        "marker_size": MARKER_SIZE,
        "grid_rect": [GRID_LEFT, GRID_TOP, GRID_RIGHT, GRID_BOTTOM],
        "patch_rows": PATCH_ROWS,
        "patch_cols": PATCH_COLS,
        "patch_gap": PATCH_GAP,
        "patches": patches,
    }


# MARK: Json
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as _np

        if isinstance(obj, (_np.integer,)):
            return int(obj)
        if isinstance(obj, (_np.floating,)):
            return float(obj)
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
        return super().default(obj)


@dataclass
class Patch:
    x0: int
    y0: int
    x1: int
    y1: int
    rgb: Tuple[int, int, int]


# MARK: Chart generation
def _make_patch_colors(rows: int, cols: int) -> List[Tuple[int, int, int]]:
    colors = []
    base = [
        (0, 0, 0),
        (255, 255, 255),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (128, 0, 0),
        (0, 128, 0),
        (0, 0, 128),
        (128, 128, 0),
        (0, 128, 128),
        (128, 0, 128),
    ]
    for g in np.linspace(0, 255, 12, dtype=int):
        base.append((int(g), int(g), int(g)))
    for r in (32, 96, 160, 224):
        for g in (32, 96, 160, 224):
            for b in (32, 160):
                base.append((r, g, b))
    need = rows * cols
    if len(base) < need:
        rng = np.random.default_rng(42)
        while len(base) < need:
            base.append(tuple(int(x) for x in rng.integers(0, 256, size=3)))
    return base[:need]


def generate_chart():
    img = Image.new("RGB", (A4_WIDTH_PX, A4_HEIGHT_PX), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([MARGIN_PX, MARGIN_PX, A4_WIDTH_PX - MARGIN_PX, A4_HEIGHT_PX - MARGIN_PX], outline=(0, 0, 0), width=BORDER_THICKNESS)

    def paste_marker(marker_id, cx, cy):
        marker = cv2.aruco.generateImageMarker(ARUCO_DICT, marker_id, MARKER_SIZE)
        mk = Image.fromarray(marker).convert("L")
        mk = Image.merge("RGB", (mk, mk, mk))
        img.paste(mk, (cx - MARKER_SIZE // 2, cy - MARKER_SIZE // 2))

    c_tl = (MARGIN_PX + MARKER_SIZE // 2, MARGIN_PX + MARKER_SIZE // 2)
    c_tr = (A4_WIDTH_PX - MARGIN_PX - MARKER_SIZE // 2, MARGIN_PX + MARKER_SIZE // 2)
    c_bl = (MARGIN_PX + MARKER_SIZE // 2, A4_HEIGHT_PX - MARGIN_PX - MARKER_SIZE // 2)
    c_br = (A4_WIDTH_PX - MARGIN_PX - MARKER_SIZE // 2, A4_HEIGHT_PX - MARGIN_PX - MARKER_SIZE // 2)

    paste_marker(0, *c_tl)
    paste_marker(1, *c_tr)
    paste_marker(2, *c_bl)
    paste_marker(3, *c_br)

    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except:
        font = ImageFont.load_default()
    draw.text((GRID_LEFT, GRID_TOP - 50), "UV Calibration Chart (A4, 300 DPI)", fill=(0, 0, 0), font=font)

    grid_w = GRID_RIGHT - GRID_LEFT
    grid_h = GRID_BOTTOM - GRID_TOP
    pw = (grid_w - (PATCH_COLS - 1) * PATCH_GAP) // PATCH_COLS
    ph = (grid_h - (PATCH_ROWS - 1) * PATCH_GAP) // PATCH_ROWS

    colors = _make_patch_colors(PATCH_ROWS, PATCH_COLS)
    patches = []
    idx = 0
    for r in range(PATCH_ROWS):
        for c in range(PATCH_COLS):
            x0 = GRID_LEFT + c * (pw + PATCH_GAP)
            y0 = GRID_TOP + r * (ph + PATCH_GAP)
            x1 = x0 + pw
            y1 = y0 + ph
            col = colors[idx]
            draw.rectangle([x0, y0, x1, y1], fill=col, outline=(0, 0, 0), width=1)
            draw.text((x0 + 6, y0 + 6), f"{idx+1}", fill=(0, 0, 0), font=font)
            patches.append(Patch(int(x0), int(y0), int(x1), int(y1), (int(col[0]), int(col[1]), int(col[2]))))
            idx += 1

    img.save(CHART_PNG, dpi=(DPI, DPI))
    print(f"Wrote {CHART_PNG}")

    meta = {
        "canvas_size": [A4_WIDTH_PX, A4_HEIGHT_PX],
        "margin_px": MARGIN_PX,
        "marker_size": MARKER_SIZE,
        "grid_rect": [GRID_LEFT, GRID_TOP, GRID_RIGHT, GRID_BOTTOM],
        "patch_rows": PATCH_ROWS,
        "patch_cols": PATCH_COLS,
        "patch_gap": PATCH_GAP,
        "patches": [{"rect": [p.x0, p.y0, p.x1, p.y1], "rgb": [p.rgb[0], p.rgb[1], p.rgb[2]]} for p in patches],
    }
    with open(CHART_JSON, "w") as f:
        json.dump(meta, f, indent=2, cls=NpEncoder)
    print(f"Wrote {CHART_JSON}")


# MARK: Aruco
def _detect_markers_and_warp(image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(ARUCO_DICT, params)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) < 4:
        raise RuntimeError("Could not find all 4 ArUco markers (IDs 0,1,2,3).")

    id_to_center = {}
    for cs, i in zip(corners, ids.flatten()):
        pts = cs[0]
        center = pts.mean(axis=0)
        id_to_center[int(i)] = center

    for need in (0, 1, 2, 3):
        if need not in id_to_center:
            raise RuntimeError(f"Missing ArUco ID {need}.")

    dst = np.float32(
        [
            [MARGIN_PX + MARKER_SIZE // 2, MARGIN_PX + MARKER_SIZE // 2],
            [A4_WIDTH_PX - MARGIN_PX - MARKER_SIZE // 2, MARGIN_PX + MARKER_SIZE // 2],
            [MARGIN_PX + MARKER_SIZE // 2, A4_HEIGHT_PX - MARGIN_PX - MARKER_SIZE // 2],
            [A4_WIDTH_PX - MARGIN_PX - MARKER_SIZE // 2, A4_HEIGHT_PX - MARGIN_PX - MARKER_SIZE // 2],
        ]
    )
    src = np.float32([id_to_center[0], id_to_center[1], id_to_center[2], id_to_center[3]])

    H, _ = cv2.findHomography(src, dst, method=cv2.RANSAC)
    warped = cv2.warpPerspective(image_bgr, H, (A4_WIDTH_PX, A4_HEIGHT_PX))
    return warped, H


# MARK: Calibration
def _mean_rgb_normal(img_bgr: np.ndarray, rect):
    x0, y0, x1, y1 = rect
    roi = img_bgr[y0:y1, x0:x1, :]
    b, g, r = roi.reshape(-1, 3).mean(axis=0)
    return float(r), float(g), float(b)


def _draw_sampling_visual(img_bgr: np.ndarray, patches, out_path: str, color=(0, 255, 0)):
    """Draw rectangles (patch['rect']) and 1-based indices onto img_bgr and save."""
    vis = img_bgr.copy()
    for i, p in enumerate(patches, start=1):
        x0, y0, x1, y1 = p["rect"]
        cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)
        cv2.putText(vis, str(i), (x0 + 6, y0 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    cv2.imwrite(out_path, vis)


def calibrate(normal_photo: str, uv_photo: str):
    # Load patch layout (prefer cached JSON; else rebuild from constants)
    if os.path.exists(CHART_JSON):
        with open(CHART_JSON, "r") as f:
            meta = json.load(f)
    else:
        print(f"[warn] {CHART_JSON} not found; rebuilding patch layout from constants.")
        meta = _rebuild_chart_meta()

    # Read and rectify both photos
    normal = cv2.imread(normal_photo, cv2.IMREAD_COLOR)
    uv = cv2.imread(uv_photo, cv2.IMREAD_COLOR)
    if normal is None or uv is None:
        raise RuntimeError("Failed to read one or both input photos.")

    normal_warp, _ = _detect_markers_and_warp(normal)
    uv_warp, _ = _detect_markers_and_warp(uv)

    cv2.imwrite("rectified_normal.png", normal_warp)
    cv2.imwrite("rectified_uv.png", uv_warp)

    # UV target metric (blue-dominant + flat-field)
    uv_gray = _uv_metric_flatfield(uv_warp, sigma=45.0)  # float32 0..255

    # Build regression data using INSET rectangles
    X = []  # rows: [R,G,B,1]
    y = []  # uv metric per patch
    inset_patches = []  # for diagnostics overlay

    for p in meta["patches"]:
        rect = inset_rect(p["rect"])  # shrink toward center
        x0, y0, x1, y1 = rect

        r, g, b = _mean_rgb_normal(normal_warp, rect)  # from normal-light image
        uv_mean = float(uv_gray[y0:y1, x0:x1].mean())  # from UV metric

        X.append([r, g, b, 1.0])
        y.append(uv_mean)
        inset_patches.append({"rect": [x0, y0, x1, y1]})

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Solve least-squares: y ≈ [R,G,B,1] · [a,b,c,d]
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b, c, d = theta.tolist()

    # Goodness-of-fit
    y_pred = X @ theta
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    print(f"Model: UV ≈ a*R + b*G + c*B + d")
    print(f"a={a:.6f}  b={b:.6f}  c={c:.6f}  d={d:.6f}  |  R^2={r2:.3f}")

    # Save model
    with open(UV_MODEL_JSON, "w") as f:
        json.dump({"type": "1ch", "uv_from_rgb": {"a": a, "b": b, "c": c, "d": d}, "note": "UV ≈ a*R + b*G + c*B + d; UV metric is blue-dominant with flat-field"}, f, indent=2)
    print(f"Wrote {UV_MODEL_JSON}")

    # Per-patch CSV
    with open("calibration_patches.csv", "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["idx", "R", "G", "B", "UV_meas", "UV_pred", "err"])
        for i, (rowX, y_i, yhat) in enumerate(zip(X, y, y_pred), start=1):
            w.writerow([i, f"{rowX[0]:.1f}", f"{rowX[1]:.1f}", f"{rowX[2]:.1f}", f"{y_i:.3f}", f"{yhat:.3f}", f"{(y_i - yhat):.3f}"])
    print("Wrote calibration_patches.csv")

    # Diagnostics overlays: exactly where we sampled (inset rects)
    _draw_sampling_visual(normal_warp, inset_patches, "calibration_samples_normal.png", color=(0, 255, 0))
    uv_vis = cv2.cvtColor(uv_gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    _draw_sampling_visual(uv_vis, inset_patches, "calibration_samples_uv.png", color=(255, 0, 0))
    print("Wrote calibration_samples_normal.png and calibration_samples_uv.png")


# MARK: Embedding (stealth)
def _load_uv_model():
    if not os.path.exists(UV_MODEL_JSON):
        raise RuntimeError("Missing uv_model.json. Run calibrate or calibrate_free first.")
    with open(UV_MODEL_JSON, "r") as f:
        m = json.load(f)

    t = m.get("type")
    if t == "multi":
        models = {k: np.asarray(v, dtype=np.float64) for k, v in m["filters"].items()}  # label -> (4,3)
        return {"type": "multi", "filters": models}
    elif t == "1ch":
        u = m["uv_from_rgb"]
        a = float(u["a"])
        b = float(u["b"])
        c = float(u["c"])
        d = float(u["d"])
        return {"type": "1ch", "abcd": (a, b, c, d)}
    else:
        raise RuntimeError(f"Unsupported model type in {UV_MODEL_JSON}: {t}")


def embed(cover_path: str, secret_path: str, alpha: float, out_path: str, delta_max: float = 10.0, nir_filter: str = "1000", nir_vec: str = "1,-0.5,-0.5"):
    mdl = _load_uv_model()

    # Load cover/secret
    cover_bgr = cv2.imread(cover_path, cv2.IMREAD_COLOR)
    secret_g = cv2.imread(secret_path, cv2.IMREAD_GRAYSCALE)
    if cover_bgr is None or secret_g is None:
        raise RuntimeError("Failed to read cover or secret.")
    H, W = cover_bgr.shape[:2]
    secret_g = cv2.resize(secret_g, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
    cover_rgb = cv2.cvtColor(cover_bgr, cv2.COLOR_BGR2RGB).astype(np.float64)

    # Secret preprocessing
    S = secret_g / 255.0
    S -= S.mean()
    S_hp = S - cv2.GaussianBlur(S, (0, 0), 3)

    # Texture mask
    lum = (0.2126 * cover_rgb[:, :, 0] + 0.7152 * cover_rgb[:, :, 1] + 0.0722 * cover_rgb[:, :, 2]).astype(np.float32)
    high = cv2.absdiff(lum, cv2.GaussianBlur(lum, (0, 0), 1.5))
    gnorm = high / (high.max() + 1e-9)
    strength = 0.3 + 0.7 * gnorm.astype(np.float64)

    # Perceptual penalty
    Wm = np.diag([0.30, 0.59, 0.11])
    Winv2 = np.linalg.inv(Wm @ Wm)

    # Build target sensitivity vector m (shape 3,)
    if mdl["type"] == "multi":
        if nir_filter not in mdl["filters"]:
            raise RuntimeError(f"Filter '{nir_filter}' not in model. Available: {list(mdl['filters'].keys())}")
        Theta = mdl["filters"][nir_filter]  # (4,3) maps [R,G,B,1] -> [nirB,nirG,nirR]
        J = Theta[:3, :].T  # 3x3  d(nirBGR)/d(RGB)
        w = np.asarray([float(x) for x in nir_vec.split(",")], dtype=np.float64).reshape(3)
        mvec = J.T @ w
        # target scalar along w
        K = 255.0
        d_uv = (alpha * K * S_hp * strength).astype(np.float64)
    else:
        # 1-channel model
        a, b, c, d = mdl["abcd"]
        mvec = np.asarray([a, b, c], dtype=np.float64)
        K = 255.0
        d_uv = (alpha * K * S_hp * strength).astype(np.float64)

    denom = float(mvec.T @ Winv2 @ mvec)
    if denom < 1e-9:
        raise RuntimeError("Degenerate mapping; try different calibration or parameters.")

    direction = Winv2 @ mvec
    scale = (d_uv / denom)[..., None]
    delta = np.clip(scale * direction.reshape(1, 1, 3), -float(delta_max), float(delta_max))

    new_rgb = np.clip(cover_rgb + delta, 0, 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(new_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, out_bgr)
    print(f"Wrote {out_bgr.shape} -> {out_path}")

    # Previews
    if mdl["type"] == "multi":
        R = new_rgb[:, :, 0].astype(np.float64)
        G = new_rgb[:, :, 1].astype(np.float64)
        B = new_rgb[:, :, 2].astype(np.float64)
        X = np.dstack([R, G, B, np.ones_like(R)]).reshape(-1, 4)
        NIR_pred = (X @ Theta).reshape(H, W, 3)  # nirB,nirG,nirR ~ 0..1
        scalar = np.tensordot(NIR_pred, w, axes=([2], [0]))
        scalar_n = (scalar - scalar.min()) / (scalar.max() - scalar.min() + 1e-9)
        cv2.imwrite(os.path.splitext(out_path)[0] + "_pred_nir_scalar.png", (scalar_n * 255).astype(np.uint8))
        cv2.imwrite(os.path.splitext(out_path)[0] + "_pred_nir_color.png", (np.clip(NIR_pred, 0, 1) * 255).astype(np.uint8))
    else:
        # 1ch predicted UV map: UV ≈ aR + bG + cB + d
        a, b, c, d = mdl["abcd"]
        R2 = new_rgb[:, :, 0].astype(np.float64)
        G2 = new_rgb[:, :, 1].astype(np.float64)
        B2 = new_rgb[:, :, 2].astype(np.float64)
        uv_pred = a * R2 + b * G2 + c * B2 + d
        uv_pred_norm = (uv_pred - uv_pred.min()) / (uv_pred.max() - uv_pred.min() + 1e-9)
        cv2.imwrite(os.path.splitext(out_path)[0] + "_predicted_uv.png", (uv_pred_norm * 255).astype(np.uint8))


# MARK: Main/CLI
def main():
    p = argparse.ArgumentParser(description="UV watermark tool: chart, calibrate (with diagnostics), embed.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("generate_chart", help="Create the A4 calibration chart (PNG + JSON).")
    p1.set_defaults(func=lambda a: generate_chart())

    # calibrate
    p2 = sub.add_parser("calibrate", help="Patch-based calibration from chart photos (uses ArUco + inset sampling).")
    p2.add_argument("--normal", required=True, help="Chart photo under normal light")
    p2.add_argument("--uv", required=True, help="Chart photo under UV/black light")
    p2.set_defaults(func=lambda a: calibrate(a.normal, a.uv))

    # embed
    p3 = sub.add_parser("embed", help="Embed for a chosen NIR filter.")
    p3.add_argument("--cover", required=True)
    p3.add_argument("--secret", required=True)
    p3.add_argument("--alpha", type=float, default=0.6)
    p3.add_argument("--delta_max", type=float, default=10.0)
    p3.add_argument("--nir-filter", default="1000", help="Filter label used during calibrate (e.g. 570/715/1000)")
    p3.add_argument("--nir-vec", default="1,-0.5,-0.5", help='Target NIR contrast vector "b,g,r"')
    p3.add_argument("--out", default="embedded.png")
    p3.set_defaults(func=lambda a: embed(a.cover, a.secret, a.alpha, a.out, a.delta_max, a.nir_filter, a.nir_vec))

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
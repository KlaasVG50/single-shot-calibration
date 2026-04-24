import os
import cv2
import json
import numpy as np
from pathlib import Path


# =========================================================
# CENTRAAL RAPPORT
# =========================================================
REPORT_TXT_PATH = Path(r"C:\Users\klaas\OneDrive - Universiteit Antwerpen\Schakeljaar\bachelorproef 2\PythonProject\rapport_resultaten.txt")


def append_to_report(title: str, lines):
    REPORT_TXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_TXT_PATH, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write(title + "\n")
        f.write("=" * 80 + "\n")
        for line in lines:
            f.write(str(line) + "\n")


# =========================================================
# CONFIG
# =========================================================
DATASET_DIR = "dataset_one/images"
LEFT_NAME = "echte_left.png"
RIGHT_NAME = "echte_right.png"

PATTERN_SIZE = (13, 9)
OUT_DIR = "opencv_debug"

GP_DATASET_PATH = os.path.join(OUT_DIR, "gp_dataset_test.json")

SQUARE_SIZE = 0.02
BOARD_GAP = 0.02

FOCAL_MM = 35.0
SENSOR_WIDTH_MM = 36.0
SENSOR_HEIGHT_MM = 24.0

SHOW_WINDOWS = True

# =========================================================
# GP DATASET HELPERS
# =========================================================
def load_gp_dataset(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"data": []}


def save_gp_dataset(path, dataset):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

def load_blender_gt(path):
    with open(path, "r") as f:
        return json.load(f)

def extract_uv(boards):
    """
    X = alle (u,v) punten uit beide chessboards
    """
    all_pts = []
    for c in boards:
        pts = c.reshape(-1, 2)
        all_pts.append(pts)

    return np.vstack(all_pts).tolist()


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def build_K_blender_approx(w, h):
    fx = FOCAL_MM * w / SENSOR_WIDTH_MM
    fy = FOCAL_MM * h / SENSOR_HEIGHT_MM
    cx = w / 2.0
    cy = h / 2.0

    return np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)


def board_center(corners):
    pts = corners.reshape(-1, 2)
    return pts.mean(axis=0)


def corners_bbox(corners):
    pts = corners.reshape(-1, 2)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def bbox_iou(b1, b2):
    xa = max(b1[0], b2[0])
    ya = max(b1[1], b2[1])
    xb = min(b1[2], b2[2])
    yb = min(b1[3], b2[3])

    iw = max(0.0, xb - xa)
    ih = max(0.0, yb - ya)
    inter = iw * ih

    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = a1 + a2 - inter

    if union <= 1e-12:
        return 0.0
    return inter / union


def are_same_board(c1, c2, center_thresh=35.0, iou_thresh=0.55):
    d = np.linalg.norm(board_center(c1) - board_center(c2))
    iou = bbox_iou(corners_bbox(c1), corners_bbox(c2))
    return (d < center_thresh) or (iou > iou_thresh)


def preprocess_variants(gray):
    variants = [("raw", gray)]

    eq = cv2.equalizeHist(gray)
    variants.append(("equalized", eq))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(gray)
    variants.append(("clahe", cl))

    variants.append(("blur_eq", cv2.GaussianBlur(eq, (3, 3), 0)))
    variants.append(("blur_clahe", cv2.GaussianBlur(cl, (3, 3), 0)))
    return variants


def detect_board_sb_single(gray, pattern_size):
    flags = (
        cv2.CALIB_CB_NORMALIZE_IMAGE |
        cv2.CALIB_CB_EXHAUSTIVE |
        cv2.CALIB_CB_ACCURACY
    )
    ok, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags)
    if ok and corners is not None:
        return True, corners.astype(np.float32)
    return False, None


def upscale_image(gray, scale):
    if scale == 1.0:
        return gray
    h, w = gray.shape[:2]
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_CUBIC)


def downscale_corners(corners, scale):
    if scale == 1.0:
        return corners
    c = corners.copy().astype(np.float32)
    c[:, 0, 0] /= scale
    c[:, 0, 1] /= scale
    return c


def detect_board_in_roi(img_bgr, roi, pattern_size):
    x0, y0, x1, y1 = roi
    crop = img_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return False, None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    for _, gray_var in preprocess_variants(gray):
        ok, corners = detect_board_sb_single(gray_var, pattern_size)
        if ok:
            corners[:, 0, 0] += x0
            corners[:, 0, 1] += y0
            return True, corners

        up = upscale_image(gray_var, 2.0)
        ok, corners = detect_board_sb_single(up, pattern_size)
        if ok:
            corners = downscale_corners(corners, 2.0)
            corners[:, 0, 0] += x0
            corners[:, 0, 1] += y0
            return True, corners

    return False, None


def add_unique_board(found, corners):
    for existing in found:
        if are_same_board(existing, corners):
            return False
    found.append(corners)
    return True


def detect_with_roi_list(img_bgr, pattern_size, rois):
    found = []
    for roi in rois:
        ok, corners = detect_board_in_roi(img_bgr, roi, pattern_size)
        if ok:
            add_unique_board(found, corners)
        if len(found) >= 2:
            break

    found.sort(key=lambda c: float(c.reshape(-1, 2)[:, 0].mean()))
    return found


def method_1_fixed_split(img_bgr, pattern_size):
    h, w = img_bgr.shape[:2]
    rois = [(0, 0, w // 2, h), (w // 2, 0, w, h)]
    return detect_with_roi_list(img_bgr, pattern_size, rois)


def method_2_overlap_split(img_bgr, pattern_size):
    h, w = img_bgr.shape[:2]
    mid = w // 2
    ov = max(24, w // 10)
    rois = [(0, 0, min(w, mid + ov), h), (max(0, mid - ov), 0, w, h)]
    return detect_with_roi_list(img_bgr, pattern_size, rois)


def method_3_multi_roi(img_bgr, pattern_size):
    h, w = img_bgr.shape[:2]
    roi_candidates = [
        (0, 0, int(0.50 * w), h), (int(0.50 * w), 0, w, h),
        (0, 0, int(0.55 * w), h), (int(0.45 * w), 0, w, h),
        (0, 0, int(0.60 * w), h), (int(0.40 * w), 0, w, h),
        (0, 0, int(0.65 * w), h), (int(0.35 * w), 0, w, h),
        (0, 0, int(0.70 * w), h), (int(0.30 * w), 0, w, h),
        (int(0.05 * w), 0, int(0.95 * w), h),
        (int(0.10 * w), 0, int(0.90 * w), h),
        (int(0.15 * w), 0, int(0.85 * w), h),
        (0, int(0.00 * h), w, int(0.90 * h)),
        (0, int(0.05 * h), w, int(0.95 * h)),
        (0, int(0.10 * h), w, h),
        (0, int(0.05 * h), int(0.60 * w), h),
        (int(0.40 * w), int(0.05 * h), w, h),
        (0, 0, int(0.60 * w), int(0.95 * h)),
        (int(0.40 * w), 0, w, int(0.95 * h)),
    ]

    unique = []
    seen = set()
    for r in roi_candidates:
        rr = tuple(map(int, r))
        if rr not in seen and rr[2] > rr[0] and rr[3] > rr[1]:
            unique.append(rr)
            seen.add(rr)

    return detect_with_roi_list(img_bgr, pattern_size, unique)


def method_4_dense_scan(img_bgr, pattern_size):
    h, w = img_bgr.shape[:2]
    found = []

    win_ws = [0.45, 0.55, 0.65, 0.75]
    win_hs = [0.80, 0.90, 1.00]
    step_x = max(16, w // 16)
    step_y = max(16, h // 10)

    for fw in win_ws:
        for fh in win_hs:
            ww = int(round(w * fw))
            hh = int(round(h * fh))
            if ww <= 0 or hh <= 0 or ww > w or hh > h:
                continue

            max_x = w - ww
            max_y = h - hh

            for y0 in range(0, max_y + 1, step_y):
                for x0 in range(0, max_x + 1, step_x):
                    roi = (x0, y0, x0 + ww, y0 + hh)
                    ok, corners = detect_board_in_roi(img_bgr, roi, pattern_size)
                    if ok:
                        add_unique_board(found, corners)

                    if len(found) >= 2:
                        found.sort(key=lambda c: float(c.reshape(-1, 2)[:, 0].mean()))
                        return found

    found.sort(key=lambda c: float(c.reshape(-1, 2)[:, 0].mean()))
    return found


def detect_two_boards_cascade(img_bgr, pattern_size):
    methods = [
        ("fixed_split", method_1_fixed_split),
        ("overlap_split", method_2_overlap_split),
        ("multi_roi", method_3_multi_roi),
        ("dense_scan", method_4_dense_scan),
    ]

    best_boards = []
    best_method = None

    for name, fn in methods:
        boards = fn(img_bgr, pattern_size)
        print(f"{name}: found {len(boards)} board(s)")
        if len(boards) > len(best_boards):
            best_boards = boards
            best_method = name
        if len(boards) >= 2:
            return boards, name

    return best_boards, best_method


def draw_blue_points_only(img_bgr, boards):
    vis = img_bgr.copy()
    for corners in boards:
        pts = corners.reshape(-1, 2)
        for (x, y) in pts:
            cv2.circle(vis, (int(round(x)), int(round(y))), 3, (255, 0, 0), -1)
    return vis


def reprojection_error(obj_pts, img_pts, rvec, tvec, K, dist):
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    img_pts = img_pts.reshape(-1, 2)
    err = np.linalg.norm(proj - img_pts, axis=1)
    return float(np.mean(err))


def make_board_object_points(pattern_size, square_size, board_name="A", gap=0.0):
    cols, rows = pattern_size
    pts = []

    if board_name == "A":
        for j in range(rows):
            for i in range(cols):
                pts.append([gap / 2.0 + i * square_size, 0.0, j * square_size])
    elif board_name == "B":
        for j in range(rows):
            for i in range(cols):
                pts.append([0.0, gap / 2.0 + i * square_size, j * square_size])
    else:
        raise ValueError("board_name must be 'A' or 'B'")

    return np.array(pts, dtype=np.float32)


def corners_variant(corners, pattern_size, variant_id):
    cols, rows = pattern_size
    grid = corners.reshape(rows, cols, 2)

    if variant_id == 0:
        g = grid
    elif variant_id == 1:
        g = grid[:, ::-1, :]
    elif variant_id == 2:
        g = grid[::-1, :, :]
    elif variant_id == 3:
        g = grid[::-1, ::-1, :]
    else:
        raise ValueError("variant_id must be 0..3")

    return g.reshape(-1, 1, 2).astype(np.float32)


def choose_best_pose_for_image(boards_img, objA, objB, K, dist, pattern_size):
    best = None

    detected0 = boards_img[0]
    detected1 = boards_img[1]

    assignments = [
        ("A_left_B_right", detected0, objA, detected1, objB),
        ("B_left_A_right", detected0, objB, detected1, objA),
    ]

    for assign_name, det_first, obj_first, det_second, obj_second in assignments:
        for v0 in range(4):
            for v1 in range(4):
                img0 = corners_variant(det_first, pattern_size, v0)
                img1 = corners_variant(det_second, pattern_size, v1)

                obj_pts = np.vstack([obj_first, obj_second]).astype(np.float32)
                img_pts = np.vstack([img0, img1]).reshape(-1, 2).astype(np.float32)

                ok, rvec, tvec = cv2.solvePnP(
                    obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
                )
                if not ok:
                    continue

                err = reprojection_error(obj_pts, img_pts, rvec, tvec, K, dist)

                if best is None or err < best["err"]:
                    best = {
                        "err": err,
                        "rvec": rvec,
                        "tvec": tvec,
                        "assignment": assign_name,
                        "variant_first": v0,
                        "variant_second": v1,
                    }

    if best is None:
        raise RuntimeError("Kon geen geldige solvePnP-oplossing vinden.")

    return best


def rvec_tvec_to_R_t(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3, 1)
    return R, t


def relative_pose_from_world_poses(R1, t1, R2, t2):
    R_rel = R2 @ R1.T
    t_rel = t2 - R_rel @ t1
    return R_rel, t_rel


def rotation_matrix_to_euler_xyz_deg(R):
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0.0

    return np.degrees([x, y, z]).astype(np.float64)


def build_pose_output(R_rel, t_rel):
    t_vec = np.asarray(t_rel, dtype=np.float64).reshape(3)
    tx, ty, tz = t_vec.tolist()

    rvec, _ = cv2.Rodrigues(np.asarray(R_rel, dtype=np.float64))
    rvec = rvec.reshape(3)
    rvec_x, rvec_y, rvec_z = rvec.tolist()

    roll_deg, pitch_deg, yaw_deg = rotation_matrix_to_euler_xyz_deg(np.asarray(R_rel, dtype=np.float64)).tolist()
    baseline = float(np.linalg.norm(t_vec))

    pose6_rvec = [float(tx), float(ty), float(tz), float(rvec_x), float(rvec_y), float(rvec_z)]
    pose6_euler_deg = [float(tx), float(ty), float(tz), float(roll_deg), float(pitch_deg), float(yaw_deg)]

    return {
        "R": np.asarray(R_rel, dtype=np.float64).tolist(),
        "t": t_vec.tolist(),
        "baseline_m": baseline,
        "tx": float(tx),
        "ty": float(ty),
        "tz": float(tz),
        "rvec": rvec.tolist(),
        "rvec_x": float(rvec_x),
        "rvec_y": float(rvec_y),
        "rvec_z": float(rvec_z),
        "euler_xyz_deg": [float(roll_deg), float(pitch_deg), float(yaw_deg)],
        "roll_deg": float(roll_deg),
        "pitch_deg": float(pitch_deg),
        "yaw_deg": float(yaw_deg),
        "pose6_rvec": pose6_rvec,
        "pose6_euler_deg": pose6_euler_deg,
    }


def save_pose_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_fail_debug(path, img):
    cv2.imwrite(path, img)
    print("Saved debug:", path)


def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    left_path = os.path.join(project_dir, DATASET_DIR, LEFT_NAME)
    right_path = os.path.join(project_dir, DATASET_DIR, RIGHT_NAME)

    out_dir = os.path.join(project_dir, OUT_DIR)
    ensure_dir(out_dir)

    imgL = cv2.imread(left_path)
    imgR = cv2.imread(right_path)
    if imgL is None:
        raise FileNotFoundError(left_path)
    if imgR is None:
        raise FileNotFoundError(right_path)

    h, w = imgL.shape[:2]
    K = build_K_blender_approx(w, h)
    dist = np.zeros((4, 1), dtype=np.float64)

    boardsL, methodL = detect_two_boards_cascade(imgL, PATTERN_SIZE)
    boardsR, methodR = detect_two_boards_cascade(imgR, PATTERN_SIZE)

    blender_gt = load_blender_gt("dataset_one/blender_gt.json")

    print(f"\nLeft image : {len(boardsL)} board(s) via {methodL}")
    print(f"Right image: {len(boardsR)} board(s) via {methodR}")

    if len(boardsL) < 2 or len(boardsR) < 2:
        visL = draw_blue_points_only(imgL, boardsL)
        visR = draw_blue_points_only(imgR, boardsR)
        save_fail_debug(os.path.join(out_dir, "left_failed_debug.png"), visL)
        save_fail_debug(os.path.join(out_dir, "right_failed_debug.png"), visR)
        raise RuntimeError(
            f"Niet beide borden gevonden. "
            f"L={len(boardsL)} via {methodL}, R={len(boardsR)} via {methodR}"
        )
    print("UV checksum:", np.sum(extract_uv(boardsL)))
    expected = PATTERN_SIZE[0] * PATTERN_SIZE[1]
    if len(boardsL[0]) != expected or len(boardsL[1]) != expected:
        raise RuntimeError("Linker corners count klopt niet.")
    if len(boardsR[0]) != expected or len(boardsR[1]) != expected:
        raise RuntimeError("Rechter corners count klopt niet.")

    objA = make_board_object_points(PATTERN_SIZE, SQUARE_SIZE, board_name="A", gap=BOARD_GAP)
    objB = make_board_object_points(PATTERN_SIZE, SQUARE_SIZE, board_name="B", gap=BOARD_GAP)

    bestL = choose_best_pose_for_image(boardsL, objA, objB, K, dist, PATTERN_SIZE)
    bestR = choose_best_pose_for_image(boardsR, objA, objB, K, dist, PATTERN_SIZE)

    RL, tL = rvec_tvec_to_R_t(bestL["rvec"], bestL["tvec"])
    RR, tR = rvec_tvec_to_R_t(bestR["rvec"], bestR["tvec"])

    R_rel, t_rel = relative_pose_from_world_poses(RL, tL, RR, tR)
    pose_rel = build_pose_output(R_rel, t_rel)

    dataset = load_gp_dataset(GP_DATASET_PATH)

    X_uv = {
        "left": extract_uv(boardsL),
        "right": extract_uv(boardsR)
}

    Y_gt = [
        pose_rel["tx"],
        pose_rel["ty"],
        pose_rel["tz"],
        pose_rel["roll_deg"],
        pose_rel["pitch_deg"],
        pose_rel["yaw_deg"],
    ]

    entry = {
        "index": len(dataset["data"]),
        "uv": X_uv,  # X voor GP
        "target": Y_gt # Y = ground truth pose
    }

    dataset = {"data": []}  # altijd reset
    dataset["data"].append(entry)
    save_gp_dataset(GP_DATASET_PATH, dataset)

    print("\n✅ GP dataset entry saved")

    print("\n=== RESULT: relative pose camL -> camR ===")
    print("R_rel:\n", np.asarray(pose_rel["R"]))
    print("t_rel:\n", np.asarray(pose_rel["t"]))
    print("baseline_m:", pose_rel["baseline_m"])

    print(f"\nLeft reprojection error : {bestL['err']:.4f} px")
    print(f"Right reprojection error: {bestR['err']:.4f} px")

    visL = draw_blue_points_only(imgL, boardsL)
    visR = draw_blue_points_only(imgR, boardsR)

    cv2.imwrite(os.path.join(out_dir, "left_detected.png"), visL)
    cv2.imwrite(os.path.join(out_dir, "right_detected.png"), visR)

    pose_payload = {
        "K": K.tolist(),
        "distCoeffs": dist.ravel().tolist(),
        "pattern_size": {"cols": int(PATTERN_SIZE[0]), "rows": int(PATTERN_SIZE[1])},
        "square_size_m": float(SQUARE_SIZE),
        "board_gap_m": float(BOARD_GAP),
        "left_debug": {
            "method_used": methodL,
            "reprojection_error_px": float(bestL["err"]),
            "assignment": bestL["assignment"],
            "variant_first": int(bestL["variant_first"]),
            "variant_second": int(bestL["variant_second"]),
        },
        "right_debug": {
            "method_used": methodR,
            "reprojection_error_px": float(bestR["err"]),
            "assignment": bestR["assignment"],
            "variant_first": int(bestR["variant_first"]),
            "variant_second": int(bestR["variant_second"]),
        },
        "relative_pose_camL_to_camR": pose_rel
    }

    out_json_path = os.path.join(out_dir, "estimated_pose_pnp.json")
    save_pose_json(out_json_path, pose_payload)
    print("\n✅ Saved:", out_json_path)

    append_to_report(
        title="OPENCV - BEREKENDE POSE",
        lines=[
            f"Saved JSON: {out_json_path}",
            f"baseline_m: {pose_rel['baseline_m']:.12f}",
            f"tx: {pose_rel['tx']:.12f}",
            f"ty: {pose_rel['ty']:.12f}",
            f"tz: {pose_rel['tz']:.12f}",
            f"rvec_x: {pose_rel['rvec_x']:.12f}",
            f"rvec_y: {pose_rel['rvec_y']:.12f}",
            f"rvec_z: {pose_rel['rvec_z']:.12f}",
            f"roll_deg: {pose_rel['roll_deg']:.12f}",
            f"pitch_deg: {pose_rel['pitch_deg']:.12f}",
            f"yaw_deg: {pose_rel['yaw_deg']:.12f}",
            "Translation vector t:",
            pose_rel["t"],
            "Rotation matrix R:",
            pose_rel["R"][0],
            pose_rel["R"][1],
            pose_rel["R"][2],
            f"pose6_rvec: {pose_rel['pose6_rvec']}",
            f"pose6_euler_deg: {pose_rel['pose6_euler_deg']}",
            f"Left reprojection error px: {bestL['err']:.12f}",
            f"Right reprojection error px: {bestR['err']:.12f}",
            f"Left method: {methodL}",
            f"Right method: {methodR}",
            f"Left assignment: {bestL['assignment']}",
            f"Right assignment: {bestR['assignment']}",
        ]
    )

    if SHOW_WINDOWS:
        cv2.imshow("LEFT detected", visL)
        cv2.imshow("RIGHT detected", visR)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # =====================================================
    # REPORT
    # =====================================================
    append_to_report(
        title="GP DATA + POSE",
        lines=[
            f"GP dataset file: {GP_DATASET_PATH}",
            f"Samples in dataset: {len(dataset['data'])}",
            f"Pose baseline: {pose_rel['baseline_m']}",
        ]
    )

    print("\n=== DONE ===")

if __name__ == "__main__":
    main()
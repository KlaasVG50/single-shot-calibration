import os
import json
import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np


# =========================================================
# CONFIG
# =========================================================
DATASET_DIR = Path(r"C:\Users\klaas\OneDrive - Universiteit Antwerpen\Schakeljaar\bachelorproef 2\PythonProject\dataset_one")
IMAGES_SUBDIR = "images"
META_SUBDIR = "metadata"
RENDER_META_FILENAME = "render_metadata_50pairs.json"
POSE_META_FILENAME = "estimated_pose_pnp_50pairs.json"
OPENCV_DEBUG_DIRNAME = "opencv_debug"

N_PAIRS = 50

# OpenCV / board config (volgens script 3)
PATTERN_SIZE = (13, 9)          # inner corners: cols, rows
SQUARE_SIZE_M = 0.02            # 2 cm
BOARD_GAP_M = 0.02              # 2 cm

FOCAL_MM = 35.0
SENSOR_WIDTH_MM = 36.0
SENSOR_HEIGHT_MM = 24.0

SHOW_WINDOWS = False

BLUE_BGR = (255, 0, 0)
RED_BGR = (0, 0, 255)
YELLOW_BGR = (0, 255, 255)


# =========================================================
# BLENDER SCRIPT
# - juiste scene uit script 2
# - baseline jitter tussen 10 cm en 50 cm
# - 50 paren renderen
# =========================================================
BLENDER_SCRIPT = r'''
import bpy
import os
import math
import json
import random
import bmesh
from mathutils import Vector, Matrix


# =========================================================
# UNIT CONVERSION
# Alles hier in cm ingeven, Blender gebruikt meter.
# =========================================================
def cm(value_cm: float) -> float:
    return value_cm / 100.0

def cm_vec(xyz_cm):
    return Vector((
        xyz_cm[0] / 100.0,
        xyz_cm[1] / 100.0,
        xyz_cm[2] / 100.0
    ))

def m_to_cm(value_m: float) -> float:
    return value_m * 100.0

def vec_distance(a, b) -> float:
    return (Vector(a) - Vector(b)).length


# =========================================================
# SETTINGS
# =========================================================
RENDER_RES = (640, 360)
ENGINE = "BLENDER_EEVEE"
FOCAL_MM = 35.0

OUT_DIR = r"C:\Users\klaas\OneDrive - Universiteit Antwerpen\Schakeljaar\bachelorproef 2\PythonProject\dataset_one"
IMAGES_SUBDIR = "images"
META_SUBDIR = "metadata"
META_FILENAME = "render_metadata_50pairs.json"

N_PAIRS = 50

# -------------------------
# Camera settings (in cm)
# -------------------------
CAM_L_LOC_CM = (80.0, 60.0, 78.0)
CAM_L_ROT_DEG = (81.0, 0.0, 124.0)

# baseline jitter range
BASELINE_MIN_CM = 10.0
BASELINE_MAX_CM = 50.0

RIGHT_CAMERA_SIDE_SIGN = 1

# -------------------------
# Checkerboard settings (in cm)
# -------------------------
BOARD_W_CM = 28.0
BOARD_H_CM = 20.0
HINGE_ANGLE_DEG = 90.0
THICKNESS_CM = 0.5
SQUARES_X = 14
SQUARES_Y = 10
HINGE_POINT_CM = (0.0, 0.0, 60.0)

BOARD_GAP_CM = 2.0
HALF_GAP_CM = BOARD_GAP_CM / 2.0

HINGE_POINT_A_CM = (
    HINGE_POINT_CM[0] + HALF_GAP_CM,
    HINGE_POINT_CM[1],
    HINGE_POINT_CM[2]
)

HINGE_POINT_B_CM = (
    HINGE_POINT_CM[0],
    HINGE_POINT_CM[1] + HALF_GAP_CM,
    HINGE_POINT_CM[2]
)

FLOOR_SIZE_CM = 300.0
MARKER_RADIUS_CM = 1.0
CAMERA_DISPLAY_SIZE_CM = 5.0


# =========================================================
# Scene helpers
# =========================================================
def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    for block in list(bpy.data.meshes):
        bpy.data.meshes.remove(block, do_unlink=True)

    for block in list(bpy.data.materials):
        bpy.data.materials.remove(block, do_unlink=True)

    for block in list(bpy.data.cameras):
        bpy.data.cameras.remove(block, do_unlink=True)

    for block in list(bpy.data.lights):
        bpy.data.lights.remove(block, do_unlink=True)


def setup_render():
    s = bpy.context.scene
    s.render.engine = ENGINE
    s.render.resolution_x = RENDER_RES[0]
    s.render.resolution_y = RENDER_RES[1]
    s.render.image_settings.file_format = "PNG"
    s.render.use_motion_blur = False

    s.unit_settings.system = 'METRIC'
    s.unit_settings.length_unit = 'METERS'
    s.unit_settings.scale_length = 1.0

    if hasattr(s, "eevee") and hasattr(s.eevee, "taa_render_samples"):
        s.eevee.taa_render_samples = 16


def world_white():
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True

    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    out = nodes.new(type="ShaderNodeOutputWorld")
    bg = nodes.new(type="ShaderNodeBackground")
    bg.inputs["Color"].default_value = (1, 1, 1, 1)
    bg.inputs["Strength"].default_value = 1.0
    links.new(bg.outputs["Background"], out.inputs["Surface"])


def add_floor():
    bpy.ops.mesh.primitive_plane_add(
        size=cm(FLOOR_SIZE_CM),
        location=(0, 0, 0)
    )
    bpy.context.active_object.name = "Floor"


def add_marker(location_m: Vector, name="Marker", radius_cm=MARKER_RADIUS_CM):
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=cm(radius_cm),
        location=location_m
    )
    bpy.context.active_object.name = name


def add_solidify(obj, thickness_m):
    mod = obj.modifiers.new(name="Solidify", type='SOLIDIFY')
    mod.thickness = thickness_m
    mod.offset = 0.0
    mod.use_even_offset = True


def make_procedural_uv_checker_material(squares_x: int, squares_y: int, phase_u: int = 0, phase_v: int = 0):
    mat = bpy.data.materials.new(f"CheckerUV_pu{phase_u}_pv{phase_v}")
    mat.use_nodes = True
    mat.use_backface_culling = False

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    for n in list(nodes):
        nodes.remove(n)

    out = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 1.0

    texcoord = nodes.new("ShaderNodeTexCoord")
    separate = nodes.new("ShaderNodeSeparateXYZ")

    mul_u = nodes.new("ShaderNodeMath")
    mul_u.operation = 'MULTIPLY'
    mul_u.inputs[1].default_value = float(squares_x)

    mul_v = nodes.new("ShaderNodeMath")
    mul_v.operation = 'MULTIPLY'
    mul_v.inputs[1].default_value = float(squares_y)

    floor_u = nodes.new("ShaderNodeMath")
    floor_u.operation = 'FLOOR'

    floor_v = nodes.new("ShaderNodeMath")
    floor_v.operation = 'FLOOR'

    add_phase_u = nodes.new("ShaderNodeMath")
    add_phase_u.operation = 'ADD'
    add_phase_u.inputs[1].default_value = float(phase_u)

    add_phase_v = nodes.new("ShaderNodeMath")
    add_phase_v.operation = 'ADD'
    add_phase_v.inputs[1].default_value = float(phase_v)

    add_uv = nodes.new("ShaderNodeMath")
    add_uv.operation = 'ADD'

    mod2 = nodes.new("ShaderNodeMath")
    mod2.operation = 'MODULO'
    mod2.inputs[1].default_value = 2.0

    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (0.02, 0.02, 0.02, 1.0)
    ramp.color_ramp.elements[1].position = 1.0
    ramp.color_ramp.elements[1].color = (0.98, 0.98, 0.98, 1.0)

    links.new(texcoord.outputs["UV"], separate.inputs["Vector"])
    links.new(separate.outputs["X"], mul_u.inputs[0])
    links.new(separate.outputs["Y"], mul_v.inputs[0])

    links.new(mul_u.outputs[0], floor_u.inputs[0])
    links.new(mul_v.outputs[0], floor_v.inputs[0])

    links.new(floor_u.outputs[0], add_phase_u.inputs[0])
    links.new(floor_v.outputs[0], add_phase_v.inputs[0])

    links.new(add_phase_u.outputs[0], add_uv.inputs[0])
    links.new(add_phase_v.outputs[0], add_uv.inputs[1])

    links.new(add_uv.outputs[0], mod2.inputs[0])
    links.new(mod2.outputs[0], ramp.inputs["Fac"])

    links.new(ramp.outputs["Color"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], out.inputs["Surface"])

    return mat


def create_hinged_panel_uv(name: str, width_cm: float, height_cm: float, hinge_location_cm, mat, thickness_cm=0.5):
    width_m = cm(width_cm)
    height_m = cm(height_cm)
    hinge_location_m = cm_vec(hinge_location_cm)
    thickness_m = cm(thickness_cm)

    mesh = bpy.data.meshes.new(name + "_mesh")
    bm = bmesh.new()

    v0 = bm.verts.new((0.0, 0.0, 0.0))
    v1 = bm.verts.new((0.0, 0.0, height_m))
    v2 = bm.verts.new((width_m, 0.0, height_m))
    v3 = bm.verts.new((width_m, 0.0, 0.0))
    face = bm.faces.new((v0, v1, v2, v3))

    uv_layer = bm.loops.layers.uv.new("UVMap")
    uvs = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]

    for loop, uv in zip(face.loops, uvs):
        loop[uv_layer].uv = uv

    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.location = hinge_location_m

    obj.data.materials.append(mat)
    add_solidify(obj, thickness_m)
    return obj


def create_camera_fixed(name: str, location_cm, rotation_deg, focal_mm=35.0):
    location_m = cm_vec(location_cm)

    bpy.ops.object.camera_add(location=location_m)
    cam_obj = bpy.context.active_object
    cam_obj.name = name
    cam_obj.data.lens = focal_mm
    cam_obj.data.clip_start = 0.01
    cam_obj.data.clip_end = 100.0
    cam_obj.data.display_size = cm(CAMERA_DISPLAY_SIZE_CM)

    rx, ry, rz = rotation_deg
    cam_obj.rotation_euler = (
        math.radians(rx),
        math.radians(ry),
        math.radians(rz)
    )
    return cam_obj


def get_right_camera_location_equal_distance_cm(left_loc_cm, hinge_point_cm, baseline_cm, side_sign=1):
    left = Vector(left_loc_cm)
    hinge = Vector(hinge_point_cm)

    rel = left - hinge
    rel_xy = Vector((rel.x, rel.y))
    radius_xy = rel_xy.length

    if radius_xy == 0:
        raise ValueError("Linkercamera ligt exact op Z-as boven hinge; rotatie rond Z-as is dan niet gedefinieerd.")

    max_baseline = 2.0 * radius_xy
    if baseline_cm > max_baseline:
        raise ValueError(
            f"Gevraagde baseline ({baseline_cm:.3f} cm) is te groot. "
            f"Maximaal mogelijk bij deze geometrie is {max_baseline:.3f} cm."
        )

    theta = 2.0 * math.asin(baseline_cm / (2.0 * radius_xy))
    theta *= side_sign

    rot_z = Matrix.Rotation(theta, 3, 'Z')
    new_rel = rot_z @ rel
    new_right = hinge + new_rel

    return tuple(new_right), math.degrees(theta), max_baseline


def point_camera_to_target(cam_obj, target_m: Vector):
    direction = target_m - cam_obj.location
    direction.normalize()
    quat = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_euler = quat.to_euler()


def render_cam(cam_obj, filepath):
    scene = bpy.context.scene
    scene.camera = cam_obj
    scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)


def main():
    random.seed()

    out_images = os.path.join(OUT_DIR, IMAGES_SUBDIR)
    out_meta = os.path.join(OUT_DIR, META_SUBDIR)

    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_meta, exist_ok=True)

    clear_scene()
    setup_render()
    world_white()
    add_floor()

    add_marker(cm_vec(HINGE_POINT_CM), name="HingeMarker")

    matA = make_procedural_uv_checker_material(SQUARES_X, SQUARES_Y, phase_u=0, phase_v=0)
    matB = make_procedural_uv_checker_material(SQUARES_X, SQUARES_Y, phase_u=1, phase_v=0)

    A = create_hinged_panel_uv(
        "Checker_A",
        width_cm=BOARD_W_CM,
        height_cm=BOARD_H_CM,
        hinge_location_cm=HINGE_POINT_A_CM,
        mat=matA,
        thickness_cm=THICKNESS_CM
    )

    B = create_hinged_panel_uv(
        "Checker_B",
        width_cm=BOARD_W_CM,
        height_cm=BOARD_H_CM,
        hinge_location_cm=HINGE_POINT_B_CM,
        mat=matB,
        thickness_cm=THICKNESS_CM
    )
    B.rotation_euler = (0.0, 0.0, math.radians(HINGE_ANGLE_DEG))

    camL = create_camera_fixed("Cam_L", CAM_L_LOC_CM, CAM_L_ROT_DEG, focal_mm=FOCAL_MM)

    meta = {
        "render_res": {"w": int(RENDER_RES[0]), "h": int(RENDER_RES[1])},
        "engine": ENGINE,
        "focal_mm": float(FOCAL_MM),
        "camera_left": {
            "location_cm": list(CAM_L_LOC_CM),
            "rotation_deg": list(CAM_L_ROT_DEG),
        },
        "board": {
            "board_w_cm": float(BOARD_W_CM),
            "board_h_cm": float(BOARD_H_CM),
            "thickness_cm": float(THICKNESS_CM),
            "hinge_angle_deg": float(HINGE_ANGLE_DEG),
            "hinge_point_cm": list(HINGE_POINT_CM),
            "board_gap_cm": float(BOARD_GAP_CM),
            "checker_a_hinge_cm": list(HINGE_POINT_A_CM),
            "checker_b_hinge_cm": list(HINGE_POINT_B_CM),
            "squares_x": int(SQUARES_X),
            "squares_y": int(SQUARES_Y),
            "pattern_size_internal_corners": [int(SQUARES_X - 1), int(SQUARES_Y - 1)],
            "square_size_cm": float(BOARD_W_CM / SQUARES_X),
        },
        "baseline_range_cm": [float(BASELINE_MIN_CM), float(BASELINE_MAX_CM)],
        "pairs": []
    }

    blend_path = os.path.join(OUT_DIR, "scene_50pairs.blend")

    for i in range(1, N_PAIRS + 1):
        baseline_cm = random.uniform(BASELINE_MIN_CM, BASELINE_MAX_CM)

        cam_r_loc_cm, rotation_angle_deg, max_baseline_cm = get_right_camera_location_equal_distance_cm(
            CAM_L_LOC_CM,
            HINGE_POINT_CM,
            baseline_cm,
            RIGHT_CAMERA_SIDE_SIGN
        )

        if "Cam_R" in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects["Cam_R"], do_unlink=True)

        camR = create_camera_fixed("Cam_R", cam_r_loc_cm, CAM_L_ROT_DEG, focal_mm=FOCAL_MM)
        point_camera_to_target(camR, cm_vec(HINGE_POINT_CM))
        bpy.context.view_layer.update()

        stem = f"{i:06d}"
        left_path = os.path.join(out_images, f"{stem}_left.png")
        right_path = os.path.join(out_images, f"{stem}_right.png")

        render_cam(camL, left_path)
        render_cam(camR, right_path)

        measured_baseline_m = vec_distance(camL.location, camR.location)
        measured_baseline_cm = m_to_cm(measured_baseline_m)

        meta["pairs"].append({
            "index": int(i),
            "left_image": os.path.basename(left_path),
            "right_image": os.path.basename(right_path),
            "requested_baseline_cm": float(baseline_cm),
            "measured_baseline_cm": float(measured_baseline_cm),
            "measured_baseline_m": float(measured_baseline_m),
            "cam_r_location_cm": [float(v) for v in cam_r_loc_cm],
            "cam_r_rotation_deg": [float(math.degrees(a)) for a in camR.rotation_euler],
            "rotation_around_hinge_deg": float(rotation_angle_deg),
            "max_possible_baseline_cm": float(max_baseline_cm),
            "right_camera_side_sign": int(RIGHT_CAMERA_SIDE_SIGN),
        })

        print(f"[{i:02d}/{N_PAIRS}] baseline={baseline_cm:.3f} cm | saved {os.path.basename(left_path)} / {os.path.basename(right_path)}")

    bpy.ops.wm.save_as_mainfile(filepath=blend_path)

    meta_path = os.path.join(out_meta, META_FILENAME)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\\n✅ Blender stage klaar.")
    print("Images:", out_images)
    print("Metadata:", meta_path)
    print("Blend:", blend_path)


if __name__ == "__main__":
    main()
'''


# =========================================================
# OpenCV helpers
# - juiste aanpak uit script 3
# =========================================================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


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
    import cv2

    variants = []
    variants.append(("raw", gray))

    eq = cv2.equalizeHist(gray)
    variants.append(("equalized", eq))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(gray)
    variants.append(("clahe", cl))

    variants.append(("blur_eq", cv2.GaussianBlur(eq, (3, 3), 0)))
    variants.append(("blur_clahe", cv2.GaussianBlur(cl, (3, 3), 0)))

    return variants


def detect_board_sb_single(gray, pattern_size):
    import cv2

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
    import cv2

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
    import cv2

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
    rois = [
        (0, 0, w // 2, h),
        (w // 2, 0, w, h),
    ]
    return detect_with_roi_list(img_bgr, pattern_size, rois)


def method_2_overlap_split(img_bgr, pattern_size):
    h, w = img_bgr.shape[:2]
    mid = w // 2
    ov = max(24, w // 10)

    rois = [
        (0, 0, min(w, mid + ov), h),
        (max(0, mid - ov), 0, w, h),
    ]
    return detect_with_roi_list(img_bgr, pattern_size, rois)


def method_3_multi_roi(img_bgr, pattern_size):
    h, w = img_bgr.shape[:2]

    roi_candidates = [
        (0, 0, int(0.50 * w), h),
        (int(0.50 * w), 0, w, h),

        (0, 0, int(0.55 * w), h),
        (int(0.45 * w), 0, w, h),

        (0, 0, int(0.60 * w), h),
        (int(0.40 * w), 0, w, h),

        (0, 0, int(0.65 * w), h),
        (int(0.35 * w), 0, w, h),

        (0, 0, int(0.70 * w), h),
        (int(0.30 * w), 0, w, h),

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
    import cv2

    vis = img_bgr.copy()
    for corners in boards:
        pts = corners.reshape(-1, 2)
        for (x, y) in pts:
            cv2.circle(vis, (int(round(x)), int(round(y))), 3, BLUE_BGR, -1)
    return vis


def reprojection_error(obj_pts, img_pts, rvec, tvec, K, dist):
    import cv2

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
                x = gap / 2.0 + i * square_size
                y = 0.0
                z = j * square_size
                pts.append([x, y, z])

    elif board_name == "B":
        for j in range(rows):
            for i in range(cols):
                x = 0.0
                y = gap / 2.0 + i * square_size
                z = j * square_size
                pts.append([x, y, z])
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
    import cv2

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

                proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
                proj = proj.reshape(-1, 2)
                err = np.linalg.norm(proj - img_pts, axis=1).mean()

                if best is None or err < best["err"]:
                    best = {
                        "err": float(err),
                        "rvec": rvec,
                        "tvec": tvec,
                        "assignment": assign_name,
                        "variant_first": v0,
                        "variant_second": v1,
                        "img_points_first": img0,
                        "img_points_second": img1,
                    }

    if best is None:
        raise RuntimeError("Kon geen geldige solvePnP-oplossing vinden.")

    return best


def rvec_tvec_to_R_t(rvec, tvec):
    import cv2

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3, 1)
    return R, t


def relative_pose_from_world_poses(R1, t1, R2, t2):
    R_rel = R2 @ R1.T
    t_rel = t2 - R_rel @ t1
    return R_rel, t_rel


# =========================================================
# OpenCV stage for all 50 pairs
# =========================================================
def stage_pose_estimation_for_50pairs(dataset_dir: Path):
    import cv2

    images_dir = dataset_dir / IMAGES_SUBDIR
    meta_dir = dataset_dir / META_SUBDIR
    render_meta_path = meta_dir / RENDER_META_FILENAME
    debug_dir = dataset_dir / OPENCV_DEBUG_DIRNAME

    ensure_dir(debug_dir)
    ensure_dir(meta_dir)

    if not render_meta_path.exists():
        raise FileNotFoundError(f"Render metadata niet gevonden: {render_meta_path}")

    with open(render_meta_path, "r", encoding="utf-8") as f:
        render_meta = json.load(f)

    pairs = render_meta.get("pairs", [])
    if not pairs:
        raise RuntimeError("Geen pairs gevonden in render metadata.")

    # 👉 GP DATASET (CLEAN)
    gp_data = []

    objA = make_board_object_points(PATTERN_SIZE, SQUARE_SIZE_M, board_name="A", gap=BOARD_GAP_M)
    objB = make_board_object_points(PATTERN_SIZE, SQUARE_SIZE_M, board_name="B", gap=BOARD_GAP_M)

    expected = PATTERN_SIZE[0] * PATTERN_SIZE[1]

    for p in pairs:
        idx = int(p["index"])
        left_path = images_dir / p["left_image"]
        right_path = images_dir / p["right_image"]

        imgL = cv2.imread(str(left_path))
        imgR = cv2.imread(str(right_path))

        if imgL is None or imgR is None:
            continue

        h, w = imgL.shape[:2]

        K = build_K_blender_approx(w, h)
        dist = np.zeros((4, 1), dtype=np.float64)

        print(f"\n========== Pair {idx:06d} ==========")

        boardsL, _ = detect_two_boards_cascade(imgL, PATTERN_SIZE)
        boardsR, _ = detect_two_boards_cascade(imgR, PATTERN_SIZE)

        if len(boardsL) < 2 or len(boardsR) < 2:
            print("❌ Boards niet gevonden")
            continue

        if len(boardsL[0]) != expected or len(boardsL[1]) != expected:
            continue

        if len(boardsR[0]) != expected or len(boardsR[1]) != expected:
            continue

        try:
            bestL = choose_best_pose_for_image(boardsL, objA, objB, K, dist, PATTERN_SIZE)
            bestR = choose_best_pose_for_image(boardsR, objA, objB, K, dist, PATTERN_SIZE)

            RL, tL = rvec_tvec_to_R_t(bestL["rvec"], bestL["tvec"])
            RR, tR = rvec_tvec_to_R_t(bestR["rvec"], bestR["tvec"])

            R_rel, t_rel = relative_pose_from_world_poses(RL, tL, RR, tR)

            # =========================
            # 🔵 INPUT = UV FEATURES
            # =========================
            # Gebruik dezelfde corners als solvePnP!
            imgL_A = bestL["img_points_first"]
            imgL_B = bestL["img_points_second"]

            imgR_A = bestR["img_points_first"]
            imgR_B = bestR["img_points_second"]

            uv = np.concatenate([
                imgL_A.reshape(-1),
                imgL_B.reshape(-1),
                imgR_A.reshape(-1),
                imgR_B.reshape(-1),
            ]).astype(np.float32)

            # normalisatie (BELANGRIJK)
            uv[0::2] /= w
            uv[1::2] /= h

            # =========================
            # 🔵 OUTPUT = (rvec + t)
            # =========================
            rvec_rel, _ = cv2.Rodrigues(R_rel)

            target = np.concatenate([
                rvec_rel.reshape(-1),
                t_rel.reshape(-1)
            ]).astype(np.float32)

            # =========================
            # OPSLAAN
            # =========================
            gp_data.append({
                "index": idx,
                "uv": uv.tolist(),
                "target": target.tolist()
            })

            print("✅ toegevoegd aan GP dataset")

        except Exception as e:
            print(f"❌ Pose fail: {e}")
            continue

    # =========================
    # SAVE CLEAN DATASET
    # =========================
    gp_out = meta_dir / "gp_dataset_clean.json"

    with open(gp_out, "w", encoding="utf-8") as f:
        json.dump({
            "description": "Clean dataset WITHOUT Blender ground truth",
            "input": "normalized uv coordinates (all checkerboard corners, both cameras)",
            "output": "relative pose (Rodrigues 3 + translation 3)",
            "data": gp_data
        }, f, indent=2)

    print("\n✅ CLEAN GP dataset opgeslagen:")
    print(gp_out)
# =========================================================
# Blender stage runner
# =========================================================
def run_blender_stage(blender_exe: str):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tf:
        tf.write(BLENDER_SCRIPT)
        script_path = tf.name

    try:
        cmd = [
            blender_exe,
            "--background",
            "--factory-startup",
            "--python",
            script_path,
        ]
        print("Running Blender:\n ", " ".join(cmd))
        subprocess.run(cmd, check=True)
    finally:
        try:
            os.remove(script_path)
        except OSError:
            pass


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--blender",
        default=r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe",
        help="Path naar blender.exe"
    )
    ap.add_argument(
        "--skip_render",
        action="store_true",
        help="Sla Blender rendering over en voer enkel OpenCV stage uit"
    )
    args = ap.parse_args()

    ensure_dir(DATASET_DIR)
    ensure_dir(DATASET_DIR / IMAGES_SUBDIR)
    ensure_dir(DATASET_DIR / META_SUBDIR)
    ensure_dir(DATASET_DIR / OPENCV_DEBUG_DIRNAME)

    if not args.skip_render:
        run_blender_stage(args.blender)

    stage_pose_estimation_for_50pairs(DATASET_DIR)


if __name__ == "__main__":
    main()
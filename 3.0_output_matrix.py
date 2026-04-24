import json
from pathlib import Path
import numpy as np
import torch
import gpytorch
from scipy.spatial.transform import Rotation as Rot

RESULT_PATH = "pose_result.txt"
# =========================================================
# MODEL (MOET IDENTIEK ZIJN AAN TRAINING)
# =========================================================
class PoseGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, input_dim, num_tasks):
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(),
            num_tasks=num_tasks
        )

        base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=input_dim)

        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.ScaleKernel(base_kernel),
            num_tasks=num_tasks,
            rank=6
        )

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean, covar)


# =========================================================
# LOAD TEST DATA
# =========================================================
def load_blender_gt(path):
    with open(path, "r") as f:
        return json.load(f)

def relative_pose(RL, tL, RR, tR):
    R_rel = RR @ RL.T
    t_rel = tR - R_rel @ tL
    return R_rel, t_rel
def load_test_dataset(path):
    with open(path, "r") as f:
        data = json.load(f)["data"]

    X, GT = [], []

    for item in data:
        uv = np.array(
            item["uv"]["left"] + item["uv"]["right"],
            dtype=np.float64
        ).reshape(-1)
        gt = np.array(item["target"], dtype=np.float64)

        X.append(uv)
        GT.append(gt)

    return np.array(X), np.array(GT)


# =========================================================
# LOAD MODEL
# =========================================================
def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    stats = ckpt["stats"]

    x_mean = np.array(stats["x_mean"])
    x_std = np.array(stats["x_std"])
    y_mean = np.array(stats["y_mean"])
    y_std = np.array(stats["y_std"])

    input_dim = len(x_mean)
    output_dim = len(y_mean)

    train_x = torch.zeros((1, input_dim), dtype=torch.float64)
    train_y = torch.zeros((1, output_dim), dtype=torch.float64)

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=output_dim
    ).double()

    model = PoseGP(train_x, train_y, likelihood, input_dim, output_dim).double()

    model.load_state_dict(ckpt["model_state_dict"])
    likelihood.load_state_dict(ckpt["likelihood_state_dict"])

    model.eval()
    likelihood.eval()

    return model, likelihood, x_mean, x_std, y_mean, y_std


# =========================================================
# PREDICT
# =========================================================
def predict(model, likelihood, x, x_mean, x_std, y_mean, y_std):

    x_norm = (x - x_mean) / x_std
    x_tensor = torch.tensor(x_norm, dtype=torch.float64).unsqueeze(0)

    with torch.no_grad():
        pred = likelihood(model(x_tensor))

    mean = pred.mean.numpy()       # shape (1,6)
    y = mean[0] * y_std + y_mean   # shape (6,)

    t = y[:3]
    rvec = y[3:6]

    R = Rot.from_rotvec(rvec).as_matrix()

    return R, t

def compute_baseline(t):
    return np.linalg.norm(t)

def format_matrix(mat):
    return "\n".join(["  " + str(row.tolist()) for row in mat])

def rotation_error(R_est, R_gt):
    R_rel = R_est.T @ R_gt
    trace = np.trace(R_rel)

    cos_theta = (trace - 1.0) / 2.0

    # clamp voor numerische stabiliteit
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    return np.arccos(cos_theta)  # in radialen

def translation_error(t_est, t_gt):
    return np.linalg.norm(t_est - t_gt)

def pose_loss(t_est, t_gt, R_est, R_gt, lambda_rot=0.1):
    e_t = np.linalg.norm(t_est - t_gt)
    e_r = rotation_error(R_est, R_gt)

    L = (e_t ** 2) + lambda_rot * e_r

    return L, e_t, e_r
# =========================================================
# MAIN
# =========================================================
def main():

    model_path = "gp_clean_pose.pt"
    from pathlib import Path

    SCRIPT_DIR = Path(__file__).resolve().parent
    test_path = SCRIPT_DIR / "opencv_debug" / "gp_dataset_test.json"

    model, likelihood, x_mean, x_std, y_mean, y_std = load_model(model_path)

    X, GT = load_test_dataset(test_path)


    lambda_rot = 0.1  # zelf tunen!

    blender_gt_path = SCRIPT_DIR / "dataset_one" / "blender_gt.json"
    blender_gt = load_blender_gt(blender_gt_path)

    R_L = np.array(blender_gt["camL"]["R"])
    t_L = np.array(blender_gt["camL"]["t"]).reshape(3, 1)

    R_R = np.array(blender_gt["camR"]["R"])
    t_R = np.array(blender_gt["camR"]["t"]).reshape(3, 1)

    R_gt_rel, t_gt_rel = relative_pose(R_L, t_L, R_R, t_R)
    t_gt_rel = t_gt_rel.reshape(3)

    rvec_gt = Rot.from_matrix(R_gt_rel).as_rotvec()

    with open(RESULT_PATH, "w") as f:

        for i in range(len(X)):
            R_pred, t_pred = predict(model, likelihood, X[i], x_mean, x_std, y_mean, y_std)

            t_gt = t_gt_rel
            R_gt = R_gt_rel

            L, e_t, e_r = pose_loss(t_pred, t_gt, R_pred, R_gt, lambda_rot)

            baseline_pred = compute_baseline(t_pred)

            def camera_center(R, t):
                return -R.T @ t

            C_L = camera_center(R_L, t_L)
            C_R = camera_center(R_R, t_R)

            baseline_gt = np.linalg.norm(C_R - C_L)


            f.write("\n" + "=" * 80 + "\n")
            f.write(f"SAMPLE {i}\n")
            f.write("=" * 80 + "\n")

            # ---------------- PREDICTED ----------------
            f.write("\n[PREDICTED]\n")
            f.write(f"t_pred: {t_pred.tolist()}\n")
            f.write(f"baseline_pred: {baseline_pred}\n")
            f.write("R_pred:\n")
            for row in R_pred:
                f.write(f"{row.tolist()}\n")

            # ---------------- GROUND TRUTH ----------------
            f.write("\n[GROUND TRUTH]\n")
            f.write(f"t_gt: {t_gt.tolist()}\n")
            f.write(f"baseline_gt: {baseline_gt}\n")
            f.write("R_gt:\n")
            for row in R_gt:
                f.write(f"{row.tolist()}\n")

            f.write(f"baseline_gt: {baseline_gt}\n")
            f.write(f"baseline_pred (norm t): {baseline_pred}\n")

            # ---------------- ERRORS ----------------
            f.write("\n[ERRORS]\n")
            f.write(f"translation_error: {e_t}\n")
            f.write(f"rotation_error_rad: {e_r}\n")
            f.write(f"rotation_error_deg: {np.degrees(e_r)}\n")
            f.write(f"weigthed_loss: {L}\n")

    print(f"Resultaten opgeslagen in: {RESULT_PATH}")
    print("baseline from your code:", baseline_gt)
    print("GT baseline:", baseline_gt)
    print("Pred baseline:", baseline_pred)

if __name__ == "__main__":
    main()
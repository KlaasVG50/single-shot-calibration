import json
import torch
import gpytorch
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot

# ========================
# CONFIG
# ========================
DATASET_DIR = "dataset_one"
META_SUBDIR = "metadata"
META_FILENAME = "stereo_50pairs_simple_rt.json"
DEVICE = 'cpu'  # 'cuda' als je GPU wilt gebruiken
torch.manual_seed(0)

# ========================
# HELPER FUNCTIES
# ========================
def posevec_from_R_T(R_3x3, T_3):
    """Convert (R,T) to 6D pose vector [tx,ty,tz, roll,pitch,yaw] in degrees."""
    Rm = np.array(R_3x3, dtype=float)
    t = np.array(T_3, dtype=float).reshape(3)
    euler = Rot.from_matrix(Rm).as_euler("xyz", degrees=True)
    return np.hstack([t, euler])

def load_pairs(meta_path):
    """Load stereo pairs from JSON."""
    if not meta_path.exists():
        raise FileNotFoundError(f"JSON file niet gevonden: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    pairs = meta.get("pairs", [])
    if not pairs:
        raise RuntimeError("Geen pairs gevonden in JSON.")
    return pairs

# ========================
# GPyTorch MODEL
# ========================
class MultitaskGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, num_tasks, num_inducing=50):
        # Variational distribution voor multitask GP
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=num_inducing, batch_shape=torch.Size([num_tasks])
        )
        variational_strategy = gpytorch.variational.MultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self,
                inducing_points=torch.linspace(0, num_inducing - 1, num_inducing).unsqueeze(-1),
                variational_distribution=variational_distribution,
                learn_inducing_locations=True
            ),
            num_tasks=num_tasks
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_tasks]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_tasks])),
            batch_shape=torch.Size([num_tasks])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# ========================
# TRAIN GP
# ========================
def train_gp_gpytrch():
    meta_path = Path(DATASET_DIR) / META_SUBDIR / META_FILENAME
    pairs = load_pairs(meta_path)

    # Build dataset
    Y = []
    baselines = []
    for p in pairs:
        stereo = p.get("stereo_right_wrt_left_in_left_coords", None)
        if stereo is None:
            continue
        R_3x3 = stereo["R_3x3"]
        T_3 = stereo["T_3"]
        Y.append(posevec_from_R_T(R_3x3, T_3))
        baselines.append(float(stereo.get("baseline_m", np.linalg.norm(T_3))))

    Y = np.array(Y, dtype=float)  # shape [N, 6]
    X = np.arange(len(Y)).reshape(-1, 1).astype(float)

    X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    Y_tensor = torch.tensor(Y, dtype=torch.float32).to(DEVICE)  # shape [N, 6] ✅

    # Setup GP
    num_tasks = Y_tensor.shape[1]  # 6D pose
    model = MultitaskGPModel(num_tasks).to(DEVICE)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks).to(DEVICE)

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=0.05)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=X_tensor.size(0))

    print("Training GP...")
    for i in range(250):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = -mll(output, Y_tensor)
        loss.backward()
        if i % 50 == 0:
            print(f"Iter {i}/250 - Loss: {loss.item():.3f}")
        optimizer.step()

    print("✅ Training klaar")
    return model, likelihood, X_tensor, Y_tensor, baselines

# ========================
# SINGLE SHOT PREDICT
# ========================
def predict_single_shot(model, likelihood, index):
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        X_test = torch.tensor([[index]], dtype=torch.float32).to(DEVICE)
        pred = likelihood(model(X_test))
        mean = pred.mean.cpu().numpy().flatten()
        std = pred.variance.sqrt().cpu().numpy().flatten()
        print(f"Predicted pose for pair index {index}:")
        print(f"Translation (m): {mean[:3]}")
        print(f"Rotation (deg): {mean[3:]}")
        print(f"Std dev: {std}")
        return mean, std

# ========================
# MAIN
# ========================
if __name__ == "__main__":
    model, likelihood, X_train, Y_train, baselines = train_gp_gpytrch()

    # Single-shot predict voor pair index 0
    predict_single_shot(model, likelihood, index=0)

# Na het trainen van je GP (uit je trainingsscript)
torch.save({
    "model_state_dict": model.state_dict(),
    "likelihood_state_dict": likelihood.state_dict(),
    "num_tasks": 6
}, "pose_prior_gp_torch.pt")
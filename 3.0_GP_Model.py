import json
import argparse
from pathlib import Path

import numpy as np
import torch
import gpytorch



# =========================================================
# CONFIG
# =========================================================
DEVICE = "cpu"

# 🔥 float64 blijft belangrijk voor stabiliteit
DTYPE = torch.float64

torch.manual_seed(0)
np.random.seed(0)


# =========================================================
# DATA LOADING
# =========================================================
def load_clean_dataset(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)["data"]

    X, Y, idx = [], [], []

    for item in data:
        X.append(np.asarray(item["uv"], dtype=np.float64))
        Y.append(np.asarray(item["target"], dtype=np.float64))
        idx.append(item["index"])

    return np.asarray(X), np.asarray(Y), idx


# =========================================================
# GP MODEL
# =========================================================
class PoseGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, input_dim, num_tasks):
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(),
            num_tasks=num_tasks
        )

        # 🔥 RBF kernel (zoals gevraagd)
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
# TRAINING
# =========================================================
def train_gp(clean_path: Path, iters=400, lr=0.05):

    X_raw, Y_raw, indices = load_clean_dataset(clean_path)

    print(f"Samples: {len(X_raw)}")

    # -------------------------
    # NORMALIZATION
    # -------------------------
    x_mean = X_raw.mean(axis=0)
    x_std = X_raw.std(axis=0)
    x_std[x_std < 1e-12] = 1.0

    y_mean = Y_raw.mean(axis=0)
    y_std = Y_raw.std(axis=0)
    y_std[y_std < 1e-12] = 1.0

    Xn = (X_raw - x_mean) / x_std
    Yn = (Y_raw - y_mean) / y_std

    train_x = torch.from_numpy(Xn).to(device=DEVICE, dtype=torch.float64)
    train_y = torch.from_numpy(Yn).to(device=DEVICE, dtype=torch.float64)

    input_dim = train_x.shape[1]
    num_tasks = train_y.shape[1]

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=num_tasks,
        noise_constraint=gpytorch.constraints.GreaterThan(1e-3)
    ).to(DEVICE).double()

    model = PoseGP(
        train_x,
        train_y,
        likelihood,
        input_dim=input_dim,
        num_tasks=num_tasks
    ).to(DEVICE).double()

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    print("Training GP (RBF)")

    for i in range(iters):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

        if i % 50 == 0 or i == iters - 1:
            print(f"Iter {i:03d} | Loss {loss.item():.6f}")

    print("✅ Training done")

    return model, likelihood, {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "indices": indices,
        "input_dim": input_dim,
        "num_tasks": num_tasks
    }


# =========================================================
# SAVE
# =========================================================
def save_model(path, model, likelihood, stats):

    torch.save({
        "model_state_dict": model.state_dict(),
        "likelihood_state_dict": likelihood.state_dict(),
        "stats": {
            "x_mean": stats["x_mean"].tolist(),
            "x_std": stats["x_std"].tolist(),
            "y_mean": stats["y_mean"].tolist(),
            "y_std": stats["y_std"].tolist(),
            "indices": stats["indices"],
            "input_dim": stats["input_dim"],
            "num_tasks": stats["num_tasks"],
        }
    }, path)

    print(f"✅ Saved: {path}")


# =========================================================
# MAIN
# =========================================================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset_dir", default="dataset_one")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--out", default="gp_clean_pose.pt")

    args = ap.parse_args()

    clean_path = Path(args.dataset_dir) / "metadata" / "gp_dataset_clean.json"

    model, likelihood, stats = train_gp(
        clean_path,
        iters=args.iters,
        lr=args.lr
    )

    save_model(args.out, model, likelihood, stats)


if __name__ == "__main__":
    main()
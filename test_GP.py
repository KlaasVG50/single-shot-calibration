import json
import numpy as np
import torch
import gpytorch
from pathlib import Path

# ---------------------------
# CONFIG
# ---------------------------
DATASET_JSON = "dataset_one/opencv_debug_annotated/triangulation_uv_50pairs.json"
GP_MODEL_PATH = "gp_models.pth"
DEVICE = "cpu"  # of "cuda" als GPU beschikbaar

EPOCHS = 200
LR = 0.1

# ---------------------------
# Dataset class
# ---------------------------
class StereoCheckerDataset(torch.utils.data.Dataset):
    def __init__(self, json_file: str):
        with open(json_file, "r") as f:
            data = json.load(f)

        pairs = data.get("pairs", [])
        Xs, Ys_T, Ys_R = [], [], []

        for p in pairs:
            if not p.get("ok", False):
                continue

            H_L_A = np.array(p["homography_uv_to_px"]["left"]["board_A"], dtype=np.float32)
            H_L_B = np.array(p["homography_uv_to_px"]["left"]["board_B"], dtype=np.float32)
            H_R_A = np.array(p["homography_uv_to_px"]["right"]["board_A"], dtype=np.float32)
            H_R_B = np.array(p["homography_uv_to_px"]["right"]["board_B"], dtype=np.float32)

            feat = np.concatenate([H_L_A.flatten(), H_L_B.flatten(), H_R_A.flatten(), H_R_B.flatten()])
            Xs.append(feat)

            T = np.array(p["stereo_pose"]["T_m"], dtype=np.float32)      # 3-dim translation
            R = np.array(p["stereo_pose"]["R"], dtype=np.float32).flatten()  # 9-dim rotation
            Ys_T.append(T)
            Ys_R.append(R)

        self.X = torch.tensor(np.stack(Xs), dtype=torch.float32)
        self.Y_T = torch.tensor(np.stack(Ys_T), dtype=torch.float32)
        self.Y_R = torch.tensor(np.stack(Ys_R), dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y_T[idx], self.Y_R[idx]

# ---------------------------
# GP model
# ---------------------------
class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# ---------------------------
# Training helper
# ---------------------------
def train_gp(X, Y, epochs=EPOCHS, lr=LR):
    models = []
    likelihoods = []

    for dim in range(Y.shape[1]):
        y_dim = Y[:, dim]
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GPModel(X, y_dim, likelihood)
        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for i in range(epochs):
            optimizer.zero_grad()
            output = model(X)
            loss = -mll(output, y_dim)
            loss.backward()
            optimizer.step()
            if (i+1) % 50 == 0:
                print(f"Dim {dim+1}/{Y.shape[1]} Iter {i+1}/{epochs} - Loss: {loss.item():.4f}")

        models.append(model)
        likelihoods.append(likelihood)

    return models, likelihoods

# ---------------------------
# Main
# ---------------------------
def main():
    # laad dataset
    dataset = StereoCheckerDataset(DATASET_JSON)
    X = dataset.X
    Y_T = dataset.Y_T
    Y_R = dataset.Y_R

    print(f"Dataset loaded: {len(dataset)} pairs")
    print(f"Input dim: {X.shape[1]}, T dim: {Y_T.shape[1]}, R dim: {Y_R.shape[1]}")

    # train GP voor translation
    print("\n=== Training GP voor T ===")
    models_T, likelihoods_T = train_gp(X, Y_T)

    # train GP voor rotation
    print("\n=== Training GP voor R ===")
    models_R, likelihoods_R = train_gp(X, Y_R)

    # opslaan
    torch.save({
        "train_X": X,
        "train_Y_T": Y_T,
        "train_Y_R": Y_R,
        "models_T": [m.state_dict() for m in models_T],
        "likelihoods_T": [l.state_dict() for l in likelihoods_T],
        "models_R": [m.state_dict() for m in models_R],
        "likelihoods_R": [l.state_dict() for l in likelihoods_R]
    }, GP_MODEL_PATH)

    print(f"\n✅ GP models opgeslagen in {GP_MODEL_PATH}")

if __name__ == "__main__":
    main()
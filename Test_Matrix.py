import torch
import gpytorch
import numpy as np
from scipy.spatial.transform import Rotation as Rot

# ========================
# CONFIG
# ========================
DEVICE = 'cpu'  # of 'cuda' als je GPU wilt gebruiken
TORCH_GP_FILE = "pose_prior_gp_torch.pt"  # model opgeslagen via torch.save

# ========================
# GPytorch model (moet matchen met trainingsscript)
# ========================
class MultitaskGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, num_tasks):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=50, batch_shape=torch.Size([num_tasks])
        )
        variational_strategy = gpytorch.variational.MultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, torch.linspace(0, 49, 50).unsqueeze(-1), variational_distribution, learn_inducing_locations=True
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
# LAAD MODEL
# ========================
def load_gp_model(file_path):
    checkpoint = torch.load(file_path, map_location=DEVICE)
    num_tasks = checkpoint["num_tasks"]
    model = MultitaskGPModel(num_tasks).to(DEVICE)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    likelihood.load_state_dict(checkpoint["likelihood_state_dict"])
    model.eval()
    likelihood.eval()
    return model, likelihood

# ========================
# PREDICTIE
# ========================
def predict_pose(model, likelihood, index):
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        X_test = torch.tensor([[index]], dtype=torch.float32).to(DEVICE)
        pred = likelihood(model(X_test))
        mean = pred.mean.cpu().numpy().flatten()
        std = pred.variance.sqrt().cpu().numpy().flatten()
    # splitsen in translation en rotatie
    t = mean[:3]
    r_deg = mean[3:]
    # Euler naar rotatiematrix
    R = Rot.from_euler("xyz", r_deg, degrees=True).as_matrix()
    return R, t, std

# ========================
# VOORBEELD GEBRUIK
# ========================
if __name__ == "__main__":
    # 1️⃣ Laad GP
    model, likelihood = load_gp_model(TORCH_GP_FILE)
    print("✅ GP model geladen")

    # 2️⃣ Voorspel pose voor nieuw beeld (bijvoorbeeld index 0)
    R_pred, t_pred, std_pred = predict_pose(model, likelihood, index=0)

    print("\nPredicted pose for new image:")
    print("Translation T (m):", t_pred)
    print("Rotation R (3x3):\n", R_pred)
    print("Std dev:", std_pred)
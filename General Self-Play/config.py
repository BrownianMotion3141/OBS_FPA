# Inputs
import torch

## General
KMIN, KMAX, K0 = 2.0, 2000.0, 101.0          
M0 = 0.5
N = 2                          

## Optimiser
LR0, LR1 = 1e-3, 0
STEPS = 8000
WARM = 7000

## Estimators
M_STEP = 1 << 14
K_CDF = 512
N_FIELD = 1 << 17
H_DIFF = 1e-4

## Utility
GAMMA = 0.0
def u(x):
    return torch.where(x>0, x.clamp_min(1e-6).pow(1 - GAMMA) / (1 - GAMMA), torch.zeros_like(x))
def du(x):
    return torch.where(x>0, x.clamp_min(1e-6).pow(-GAMMA), torch.zeros_like(x))

DEVICE = torch.device(
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available() else "cpu"
)

def seed_everything(seed=0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def report(tag, eu, mu, kappa):
    print(f"{tag:>10s} | payoff {eu:.4f} | mu {mu:.4f} | kappa {kappa:6.2f}")

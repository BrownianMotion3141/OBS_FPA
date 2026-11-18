# The policy network
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg


class BetaPolicy(nn.Module):
    def __init__(self, h=10):
        super().__init__()
        self.fc1, self.fc2, self.head = nn.Linear(2, h), nn.Linear(h, h), nn.Linear(h, 2)
        for m in (self.fc1, self.fc2):                      
            nn.init.normal_(m.weight, 0.0, 1.0 / math.sqrt(m.in_features))
            nn.init.zeros_(m.bias)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, obs):                                   
        z = F.selu(self.fc2(F.selu(self.fc1(obs))))
        z = self.head(z)
        mu    = torch.sigmoid(z[..., 0])
        kappa = cfg.KMIN + (cfg.KMAX - cfg.KMIN) * torch.sigmoid(z[..., 1])
        return mu, kappa
    

class FixedShare:
    #Test double for BetaPolicy: always bids b = c * v.
    def __init__(self, c, kappa=2000.0):
        self.c, self.kappa = c, kappa

    def __call__(self, obs):
        shape = obs.shape[:-1]                      # strip the 2-feature axis
        return (torch.full(shape, self.c,     device=obs.device),
                torch.full(shape, self.kappa, device=obs.device))
        

def make_policy():
    ## Build the policy and set its initial (mu, kappa) via the head bias.
    policy = BetaPolicy().to(cfg.DEVICE)
    policy.head.bias.data = torch.logit(torch.tensor(
        [cfg.M0, (cfg.K0 - cfg.KMIN) / (cfg.KMAX - cfg.KMIN)], device=cfg.DEVICE))
    return policy

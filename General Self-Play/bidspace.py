# observations, bid sampling, auction clearing, policy update rules

import torch

import config as cfg


def play(bids):                       
    top, _ = bids.max(dim=1, keepdim=True)
    share = (bids == top).float()    
    return share / share.sum(dim=1, keepdim=True).clamp_min(1.0)


def obs(v):  return torch.stack([v, torch.full_like(v, 1.0 / cfg.N)], -1)


def sample_bids(policy, v):
    mu, kappa = policy(obs(v))
    a, b = mu * kappa, (1 - mu) * kappa
    S = torch.distributions.Beta(a, b).rsample()     
    return v * S, S, mu, kappa


## Reinforce Update Rule

def selfplay_step(policy, opt, M=1 << 15):
    v = torch.rand(M, cfg.N, device=cfg.DEVICE)
    bids, S, mu, kappa = sample_bids(policy, v)
    a = (mu * kappa).clamp_min(1.0)                   
    b = ((1 - mu) * kappa).clamp_min(1.0)   # min of 1 keeps the log_prob finite
    logp = torch.distributions.Beta(a, b).log_prob(S)

    payoff = play(bids) * (v - bids)                  
    adv = (payoff - payoff.mean()).detach()         
    loss = -(logp * adv).mean()

    opt.zero_grad(); loss.backward(); opt.step()
    return payoff.mean().item(), mu.mean().item(), kappa.mean().item()


## Pseudo Gradient Update Rule

# building an empirical CDF of rival bids
@torch.no_grad()
def field_cdf(policy, K=cfg.K_CDF, n=cfg.N_FIELD):
    b, _, _, _ = sample_bids(policy, torch.rand(n, device=cfg.DEVICE))
    grid = torch.cat([b.new_zeros(1), torch.quantile(b, torch.linspace(0, 1, K, device=cfg.DEVICE)), b.new_ones(1)])
    qs   = torch.cat([b.new_zeros(1), torch.linspace(0, 1, K, device=cfg.DEVICE), b.new_ones(1)])
    return grid, qs

# perform linear interpolation for a given bid
def interp_F(x, grid, qs):
    x = x.clamp(grid[0], grid[-1] - 1e-9)
    i = torch.searchsorted(grid, x).clamp(1, len(grid) - 1)
    w = (x - grid[i - 1]) / (grid[i] - grid[i - 1]).clamp_min(1e-12)
    return (qs[i - 1] + w * (qs[i] - qs[i - 1])).clamp(1e-6, 1 - 1e-6)

# optimises toward the analytical solution
def pseudo_step(policy, opt, M=cfg.M_STEP, h=cfg.H_DIFF):
    grid, qs = field_cdf(policy)
    v = torch.rand(M, cfg.N, device=cfg.DEVICE)
    bid, _, mu, kappa = sample_bids(policy, v)

    Fb = interp_F(bid, grid, qs)
    fb = (interp_F(bid + h, grid, qs) - interp_F(bid - h, grid, qs)) / (2 * h)
    win, d_win = Fb ** (cfg.N - 1), (cfg.N - 1) * Fb ** (cfg.N - 2) * fb
    
    s = v - bid
    U, dU = cfg.u(s), cfg.du(s)
    d_pay = d_win * U - win * dU
    eu = (win * U).mean().item()
    mu_bar    = mu.mean().item()
    kappa_bar = kappa.mean().item()                  

    loss = -(d_pay.detach() * bid).mean()            
    opt.zero_grad(); loss.backward(); opt.step()
    return eu, mu_bar, kappa_bar
# Exploitability, revenue, effciency, pre-run validations
import torch

import config as cfg
import bidspace as bs
import network1 as net


@torch.no_grad()
def exploitability(policy, T=201):
    grid, qs = bs.field_cdf(policy)
    v = torch.linspace(0.01, 1, T, device=cfg.DEVICE)
    cand = grid.unsqueeze(0).expand(T, -1)
    eu = (bs.interp_F(cand, grid, qs) ** (cfg.N - 1) * cfg.u(v.unsqueeze(1) - cand)).clamp_min(0)
    br = eu.max(dim=1).values
    mu, _ = policy(bs.obs(v)); played = v * mu
    eu_played = bs.interp_F(played, grid, qs) ** (cfg.N - 1) * cfg.u(v - played)
    return (br - eu_played).mean().item(), br.mean().item()


@torch.no_grad()
def relative_utility_loss(policy, T=201):
    D, br = exploitability(policy)
    return D / br


@torch.no_grad()
def curve_distance(policy, T=201, norm="l2"):
    v  = torch.linspace(0.01, 1, T, device=cfg.DEVICE)
    mu, _ = policy(bs.obs(v))
    b, bs1 = v * mu, v * (cfg.N - 1) / (cfg.N - cfg.GAMMA)
    if norm == "l2":
        return ((b - bs1).pow(2).mean().sqrt() / bs1.pow(2).mean().sqrt()).item()
    return (b - bs1).abs().max().item()


@torch.no_grad()
def revenue_mc(policy, M=1 << 16):
    v = torch.rand(M, cfg.N, device=cfg.DEVICE)
    bids, _, _, _ = bs.sample_bids(policy, v)
    return (bs.play(bids) * bids).sum(dim=1).mean().item()


@torch.no_grad()
def revenue_cdf(policy, K=512, n=1 << 16):
    grid, qs = bs.field_cdf(policy, K=K, n=n)
    b_mid = 0.5 * (grid[1:] + grid[:-1])
    F_mid = 0.5 * (qs[1:] + qs[:-1])
    dF    = (qs[1:] - qs[:-1]).clamp_min(0.0)
    return (b_mid * cfg.N * F_mid ** (cfg.N - 1) * dF).sum().item()


@torch.no_grad()
def efficiencyR(policy, **kw):
    return revenue_mc(policy, **kw) / (cfg.N / (cfg.N + 1))


@torch.no_grad()
def efficiencyA(policy, M=1 << 20):
    v = torch.rand(M, cfg.N, device=cfg.DEVICE)
    bids, _, _, _ = bs.sample_bids(policy, v)
    return (bs.play(bids) * v).sum(dim=1).mean().item() / (cfg.N / (cfg.N + 1))


def check_play(M=1 << 16):
    v = torch.rand(M, cfg.N, device=cfg.DEVICE); c = (cfg.N - 1) / (cfg.N - cfg.GAMMA)
    bids = c * v
    rev = (bs.play(bids) * bids).sum(1).mean()
    print(rev.item(), (cfg.N - 1) / (cfg.N + 1))   # comparing calculated result to the analytical result
    
    
def check_revenue(tol=5e-3, M=1 << 16):
    c = (cfg.N - 1) / (cfg.N - cfg.GAMMA)
    bench = c * cfg.N / (cfg.N + 1)
    rule = net.FixedShare(c)

    err = max(abs(revenue_mc(rule, M=M) - bench),
              abs(revenue_cdf(rule)     - bench),
              abs(efficiencyR(rule, M=M) - c))

    print(f"revenue err {err:.1e}  tol {tol:.0e}  {'ok' if err <= tol else 'FAIL'}")
    assert err <= tol, "revenue validation failed -- check play(), sample_bids(), field_cdf()"
    # print(revenue_mc(rule, cfg.N, M=M), revenue_cdf(rule, cfg.N), efficiencyR(rule, cfg.N, M=M))   # uncomment if revenue check fails
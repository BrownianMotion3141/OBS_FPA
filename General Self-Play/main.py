# Algorithm 1: Neural Self-Play
import torch

import config as cfg
import network1 as net
import bidspace as bs
import metrics as mt


def main():
    cfg.seed_everything(0)

    policy = net.make_policy()
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.LR0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.STEPS, eta_min=cfg.LR1)

    mt.check_play()
    mt.check_revenue()
    mu_chk, kappa_chk = policy(bs.obs(torch.tensor([0.5], device=cfg.DEVICE)))
    print(f"init: mu {mu_chk.item():.4f}  kappa {kappa_chk.item():.1f}")

    tail = [torch.zeros_like(p) for p in policy.parameters()]
    n = 0
    for step in range(1, cfg.STEPS + 1):
        eu, mu, kappa = bs.pseudo_step(policy, opt)   # or bs.selfplay_step(...)
        sched.step()
        if step > cfg.WARM:
            for t, p in zip(tail, policy.parameters()):
                t.add_(p.detach())
            n += 1
        if step % 1000 == 0:
            cfg.report(f"step {step}", eu, mu, kappa)

    assert n > 0, "nothing averaged - WARM too large"
    with torch.no_grad():
        for p, t in zip(policy.parameters(), tail):
            p.copy_(t / n)

    vg = torch.linspace(0.0, 1.0, 101, device=cfg.DEVICE)
    mu_avg, kappa_avg = policy(bs.obs(vg))
    print(f"avg policy: mu {mu_avg.mean().item():.4f}  kappa {kappa_avg.mean().item():.1f}")
    print(f"exploitability: {mt.relative_utility_loss(policy):.4f}")
    print(f"Expected revenue: {mt.revenue_mc(policy):.4f}")
    print(f"Revenue Efficiency: {mt.efficiencyR(policy):.4f}")
    print(f"Allocative Efficiency: {mt.efficiencyA(policy):.4f}")
    print(f"Curve Distance: {mt.curve_distance(policy):.4f}")


if __name__ == "__main__":
    main()
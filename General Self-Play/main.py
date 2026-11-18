# Algorithm 1: Neural Self-Play
import itertools
import json
import statistics as st
import time

import torch

import config as cfg
import network1 as net
import bidspace as bs
import metrics as mt

_BASE = dict(N=cfg.N, GAMMA=cfg.GAMMA, M0=cfg.M0)

# update: pseudo -> direct pseudogradient
# update: reinforce -> score function
# field: grid, qs for an exogenous opponent field; None = policy's own field
def train(seed=0, update="pseudo", field=None, N=None, GAMMA=None, M0=None, steps=None, warm=None): 
    
    cfg.N     = _BASE["N"]     if N     is None else N
    cfg.GAMMA = _BASE["GAMMA"] if GAMMA is None else GAMMA
    cfg.M0    = _BASE["M0"]    if M0    is None else M0
    steps = cfg.STEPS if steps is None else steps
    warm  = cfg.WARM  if warm  is None else warm
    
    cfg.seed_everything(seed)
    policy = net.make_policy()
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.LR0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.STEPS, eta_min=cfg.LR1)
    
    mt.check_play()
    mt.check_revenue()
    mu_chk, kappa_chk = policy(bs.obs(torch.tensor([0.5], device=cfg.DEVICE)))
    print(f"init: mu {mu_chk.item():.4f}  kappa {kappa_chk.item():.1f}")
    
    cfg.seed_everything(seed)
    tail = [torch.zeros_like(p) for p in policy.parameters()]
    n, t0 = 0, time.time()
    
    for step in range(1, steps + 1):
        if update == "pseudo":
            eu, mu, kappa = bs.pseudo_step(policy, opt, field=field)
        else:
            eu, mu, kappa = bs.selfplay_step(policy, opt) 
        sched.step()
        if step > warm:
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
    D, br = mt.deviation_gain(policy, field=field)
    
    out = dict(seed=seed, N=cfg.N, GAMMA=cfg.GAMMA, M0=cfg.M0, update=update,
               mu=mu_avg.mean().item(), kappa=kappa_avg.mean().item(),
               D=D, rul=D / br, payoff=mt.played_payoff(policy, field=field),
               R=mt.revenue_mc(policy), rho=mt.efficiencyR(policy),
               effA=mt.efficiencyA(policy),
               # curve distance is only defined against the symmetric benchmark c*v
               cd_l2=None if field else mt.curve_distance(policy),
               cd_inf=None if field else mt.curve_distance(policy, norm="inf"),
               secs=round(time.time() - t0, 1))
    
    print(f"avg policy: mu {out['mu']:.4f}  kappa {out['kappa']:.1f}")
    print(f"D {out['D']:.2e} | RUL {out['rul']:.4f} | payoff {out['payoff']:.4f} | "
          f"R {out['R']:.4f} | rho {out['rho']:.4f} | effA {out['effA']:.4f}")

    return out

def sweep(Ns=(2, 3, 5), GAMMAs=(0.0, 0.25, 0.5, 0.75, 0.9), seeds=(0, 1), **kw):
    rows = []
    for N, GAMMA, seed in itertools.product(Ns, GAMMAs, seeds):
        print(f"\n=== N={N}  gamma={GAMMA}  seed={seed} ===")
        rows.append(train(seed=seed, N=N, GAMMA=GAMMA,
                          M0=(N - 1) / (N - GAMMA), **kw))
    return rows       

def table(rows):
    hdr = (f"{'N':>2} {'gamma':>6} {'mu':>8} {'sd':>7} {'target':>7} "
           f"{'kappa':>8} {'R':>7} {'rho':>7} {'effA':>6} {'RUL':>7}")
    print("\n" + hdr)
    print("-" * len(hdr))
    cells = sorted({(r["N"], r["GAMMA"]) for r in rows})
    for N, GAMMA in cells:
        c = [r for r in rows if r["N"] == N and r["GAMMA"] == GAMMA]
        mus = [r["mu"] for r in c]
        m = lambda k: sum(r[k] for r in c) / len(c)
        print(f"{N:>2} {GAMMA:>6.2f} {m('mu'):>8.4f} "
              f"{(st.stdev(mus) if len(mus) > 1 else 0.0):>7.4f} "
              f"{(N - 1) / (N - GAMMA):>7.4f} {m('kappa'):>8.1f} "
              f"{m('R'):>7.4f} {m('rho'):>7.4f} {m('effA'):>6.3f} {m('rul'):>7.4f}")
        
if __name__ == "__main__":
    rows = sweep()
    table(rows)
    json.dump(rows, open("results.json", "w"), indent=2)

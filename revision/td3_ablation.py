"""TD3 ablation and training diagnostics (paper Section 5.11, Table 5).

Same simulator, TD3 networks/hyper-parameters and terminal one-step formulation as the paper.
Variants (N=128):
  steps      : 1200 (paper protocol, 240 updates), 5000 and 20000 (1 critic update per step)
  obs        : 'base' (paper Eq. 15) or 'eve' (adds normalised eavesdropper large-scale statistics)
  action     : 'raw'   (paper Eqs. 17-19, phase = pi*a)
               '2pi'   (phase = 2*pi*a)
               'phase' (actor outputs only N phases; BS beam from the same surrogate beamformer
                        used by AO-/Random-RIS -> isolates the beamforming part of the action)
Diagnostics: reward mean/SD/zero-fraction, critic loss, critic and actor gradient norms,
             held-out secrecy on the paper's 120 validation snapshots every 500 transitions.
Final evaluation: the exact 1,000 nominal channels of Table 2.
"""
import sys, json, time, argparse
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / 'src'
sys.path.insert(0, str(SRC))
from physical_umi import (Config, channel_snapshot, observation, physical_action, secrecy,
                          TD3, Replay, secrecy_surrogate_beamformer, eve_covariance, effective_row)
from run_sweeps import sample_training_config

OUT = REPO / 'results' / 'ablation'
OUT.mkdir(parents=True, exist_ok=True)


def obs_fn(cfg, s, mode):
    o = observation(cfg, s)
    if mode == 'eve':
        extra = np.array([(10 * np.log10(s['beta_be']) + 100) / 50,
                          (10 * np.log10(s['beta_re']) + 100) / 50], dtype=np.float32)
        o = np.concatenate([o, extra])
    return o


def act_dim(cfg, mode):
    return cfg.n if mode == 'phase' else 2 * cfg.m + cfg.n


def to_physical(cfg, s, a, mode):
    if mode == 'raw':
        return physical_action(cfg, a)
    if mode == '2pi':
        w, _ = physical_action(cfg, a)
        return w, np.exp(1j * 2 * np.pi * np.asarray(a[2 * cfg.m:], dtype=float))
    # 'phase': actor sets phases, beam from surrogate beamformer (same as Random-RIS / AO beam step)
    phi = np.exp(1j * np.pi * np.asarray(a, dtype=float))
    row = effective_row(s['hbu_obs'], s['hru_obs'], s['Hbr_obs'], phi)
    return secrecy_surrogate_beamformer(cfg, row, eve_covariance(cfg, s, True)), phi


def grad_norm(params):
    t = [p.grad.detach().norm() ** 2 for p in params if p.grad is not None]
    return float(torch.sqrt(torch.stack(t).sum())) if t else 0.0


def update(agent, batch, diag):
    s, a, r, s2, d = [torch.as_tensor(x, dtype=torch.float32) for x in batch]
    r = r.view(-1, 1); d = d.view(-1, 1)
    with torch.no_grad():
        n = torch.clamp(torch.randn_like(a) * .2, -.5, .5)
        a2 = torch.clamp(agent.actor_target(s2) + n, -1., 1.)
        q = torch.minimum(agent.q1t(torch.cat([s2, a2], -1)), agent.q2t(torch.cat([s2, a2], -1)))
        y = r + .99 * (1. - d) * q
    xin = torch.cat([s, a], 1)
    q1 = agent.q1(xin)
    loss = F.mse_loss(q1, y) + F.mse_loss(agent.q2(xin), y)
    agent.q_opt.zero_grad(); loss.backward()
    diag['critic_grad'].append(grad_norm(list(agent.q1.parameters()) + list(agent.q2.parameters())))
    agent.q_opt.step()
    diag['critic_loss'].append(float(loss)); diag['q_mean'].append(float(q1.mean()))
    agent.step += 1
    if agent.step % 2 == 0:
        al = -agent.q1(torch.cat([s, agent.actor(s)], -1)).mean()
        agent.actor_opt.zero_grad(); al.backward()
        diag['actor_grad'].append(grad_norm(agent.actor.parameters()))
        agent.actor_opt.step()
        with torch.no_grad():
            for net, tgt in ((agent.actor, agent.actor_target), (agent.q1, agent.q1t), (agent.q2, agent.q2t)):
                for p, pt in zip(net.parameters(), tgt.parameters()):
                    pt.data.mul_(.995).add_(.005 * p.data)


def evaluate(cfg, agent, chans, omode, amode):
    with torch.no_grad():
        acts = agent.actor(torch.tensor(np.stack([obs_fn(cfg, s, omode) for s in chans]))).numpy()
    return np.array([secrecy(cfg, s, *to_physical(cfg, s, a, amode))[0] for s, a in zip(chans, acts)])


def nominal_channels(cfg, n_mc=1000):
    rng = np.random.default_rng(880021 + int(128 * 1000) + len('n') * 90)
    out = []
    for _ in range(n_mc):
        out.append(channel_snapshot(cfg, rng)); rng.uniform(-np.pi, np.pi, cfg.n)
    return out


def train(cfg, seed, steps, upd_per_step, omode, amode, val, paper_schedule=False):
    rng = np.random.default_rng(seed); torch.manual_seed(seed); torch.set_num_threads(1)
    s0 = channel_snapshot(cfg, rng)
    sd = len(obs_fn(cfg, s0, omode)); ad = act_dim(cfg, amode)
    agent = TD3(sd, ad, seed); mem = Replay(max(steps + 1, 2000), sd, ad)
    diag = {k: [] for k in ('critic_loss', 'critic_grad', 'actor_grad', 'q_mean')}
    rewards = []; curve = [(0, float(evaluate(cfg, agent, val, omode, amode).mean()))]
    warm = max(150, steps // 5) if paper_schedule else max(240, steps // 10)
    for t in range(steps):
        ct = sample_training_config(cfg, rng)
        s = channel_snapshot(ct, rng); ob = obs_fn(ct, s, omode)
        a = rng.uniform(-1., 1., ad) if t < warm else np.clip(agent.act(ob) + rng.normal(0, .1, ad), -1, 1)
        r = secrecy(ct, s, *to_physical(ct, s, a, amode))[0]
        rewards.append(r); mem.add(ob, a, r, ob, 1.)
        if t >= warm:
            if paper_schedule:
                if (t - warm) % max(1, (steps - warm) // 240) == 0:
                    update(agent, mem.batch(rng, min(128, mem.count)), diag)
            else:
                for _ in range(upd_per_step):
                    update(agent, mem.batch(rng, min(128, mem.count)), diag)
        if (t + 1) % (200 if steps <= 1200 else 500) == 0:
            curve.append((t + 1, float(evaluate(cfg, agent, val, omode, amode).mean())))
    rw = np.array(rewards)
    summ = {'reward_mean_random_phase': float(rw[:warm].mean()), 'reward_mean_policy_phase': float(rw[warm:].mean()) if steps > warm else None,
            'reward_sd': float(rw.std()), 'reward_zero_fraction': float((rw <= 1e-12).mean()),
            'updates': len(diag['critic_loss'])}
    for k, v in diag.items():
        v = np.array(v)
        if len(v):
            q = max(1, len(v) // 10)
            summ[k + '_first10pct'] = float(v[:q].mean()); summ[k + '_last10pct'] = float(v[-q:].mean())
    return agent, curve, summ


VARIANTS = {
    'paper_1200':      dict(steps=1200, ups=1, obs='base', act='raw', paper=True),
    'steps_5000':      dict(steps=5000, ups=1, obs='base', act='raw', paper=False),
    'steps_20000':     dict(steps=20000, ups=1, obs='base', act='raw', paper=False),
    'eve_obs_20000':   dict(steps=20000, ups=1, obs='eve', act='raw', paper=False),
    'phase2pi_20000':  dict(steps=20000, ups=1, obs='base', act='2pi', paper=False),
    'phase_only_5000': dict(steps=5000, ups=1, obs='base', act='phase', paper=False),
}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--variant', required=True)
    ap.add_argument('--seeds', type=int, default=5); ap.add_argument('--only-seed', type=int, default=None)
    args = ap.parse_args()
    v = VARIANTS[args.variant]
    cfg = Config()
    vr = np.random.default_rng(440011 + cfg.n)
    val = [channel_snapshot(cfg, vr) for _ in range(120)]  # paper's validation set
    chans = nominal_channels(cfg)
    res = {'variant': args.variant, 'spec': v, 'seeds': []}
    per = []
    t0 = time.perf_counter()
    seed_list = [args.only_seed] if args.only_seed is not None else range(args.seeds)
    tag = '' if args.only_seed is None else f'_s{args.only_seed}'
    for i in seed_list:
        seed = 19542 + 101 * i + 13 * cfg.n
        ag, curve, summ = train(cfg, seed, v['steps'], v['ups'], v['obs'], v['act'], val, v['paper'])
        ev = evaluate(cfg, ag, chans, v['obs'], v['act']); per.append(ev)
        res['seeds'].append({'seed_idx': i, 'curve': curve, 'nominal_mean': float(ev.mean()), **summ})
        print(args.variant, 'seed', i, round(ev.mean(), 4), round(time.perf_counter() - t0, 1), flush=True)
    per = np.array(per)
    sm = per.mean(axis=1)
    res['nominal_mean_over_seeds'] = float(sm.mean()); res['between_seed_sd'] = float(sm.std(ddof=1)) if len(sm) > 1 else 0.0
    res['train_seconds_total'] = time.perf_counter() - t0
    np.save(OUT / f'{args.variant}{tag}_per_channel.npy', per)
    (OUT / f'{args.variant}{tag}.json').write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ('variant', 'nominal_mean_over_seeds', 'between_seed_sd', 'train_seconds_total')}))


if __name__ == '__main__':
    main()

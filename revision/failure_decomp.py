"""TD3 failure decomposition on the exact Table-2 nominal channels (paper Section 5.11).

Compares the trained paper actors (N=128, five seeds) with:
  - uniformly random raw actions through the same mapping (Eqs. 17-19)  -> 'no-learning' level
  - TD3 beamformer + all-one phases, and TD3 phases + surrogate beamformer -> which part fails
  - actor output statistics (how far actions move from the initial policy)
"""
import sys, json
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / 'src'
sys.path.insert(0, str(SRC))
from physical_umi import (Config, channel_snapshot, observation, physical_action, secrecy, MLP,
                          secrecy_surrogate_beamformer, eve_covariance, effective_row)

REPO = Path(__file__).resolve().parents[1]
W = REPO


def nominal_channels(cfg, n_mc=1000):
    rng = np.random.default_rng(880021 + int(128 * 1000) + len('n') * 90)
    out = []
    for _ in range(n_mc):
        out.append(channel_snapshot(cfg, rng)); rng.uniform(-np.pi, np.pi, cfg.n)
    return out


def main():
    cfg = Config()
    chans = nominal_channels(cfg)
    obs = torch.tensor(np.stack([observation(cfg, s) for s in chans]))
    sd = obs.shape[1]; ad = 2 * cfg.m + cfg.n
    rng = np.random.default_rng(5150)
    res = {}
    # random raw actions
    rnd = [secrecy(cfg, s, *physical_action(cfg, rng.uniform(-1, 1, ad)))[0] for s in chans]
    res['uniform_random_action'] = float(np.mean(rnd))
    td_full, td_beam_only, td_phase_only, mrt_dir = [], [], [], []
    act_stats = []
    for k in range(5):
        actor = MLP(sd, ad, True)
        actor.load_state_dict(torch.load(W / 'artifacts' / 'actors' / f'actor_N128_seed{k}.pt'))
        # untrained actor of the same seed for drift measurement
        torch.manual_seed(19542 + 101 * k + 13 * 128); init = MLP(sd, ad, True)
        with torch.no_grad():
            A = actor(obs).numpy(); A0 = init(obs).numpy()
        act_stats.append({'mean_abs_action': float(np.abs(A).mean()),
                          'fraction_saturated_|a|>0.99': float((np.abs(A) > .99).mean()),
                          'mean_abs_change_from_init': float(np.abs(A - A0).mean()),
                          'action_sd_across_channels': float(A.std(axis=0).mean())})
        f, b, p, m = [], [], [], []
        for s, a in zip(chans, A):
            w, phi = physical_action(cfg, a)
            f.append(secrecy(cfg, s, w, phi)[0])
            b.append(secrecy(cfg, s, w, np.ones(cfg.n, dtype=complex))[0])
            row = effective_row(s['hbu_obs'], s['hru_obs'], s['Hbr_obs'], phi)
            wb = secrecy_surrogate_beamformer(cfg, row, eve_covariance(cfg, s, True))
            p.append(secrecy(cfg, s, wb, phi)[0])
            # alignment of TD3 beam with the surrogate-optimal beam for its own phases
            m.append(abs(np.vdot(wb, w)) / (np.linalg.norm(wb) * np.linalg.norm(w)))
        td_full.append(np.mean(f)); td_beam_only.append(np.mean(b)); td_phase_only.append(np.mean(p)); mrt_dir.append(np.mean(m))
    res['td3_full_action'] = float(np.mean(td_full))
    res['td3_beam_with_all_one_phases'] = float(np.mean(td_beam_only))
    res['td3_phases_with_surrogate_beam'] = float(np.mean(td_phase_only))
    res['mean_|cos|_td3_beam_vs_surrogate_beam'] = float(np.mean(mrt_dir))
    res['random_beam_expected_|cos|_M8'] = float(np.mean([abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
                                                       for a, b in ((rng.standard_normal(8) + 1j * rng.standard_normal(8),
                                                                     rng.standard_normal(8) + 1j * rng.standard_normal(8)) for _ in range(5000))]))
    res['actor_output_stats'] = act_stats
    (W / 'results' / 'ablation' / 'failure_decomposition.json').write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()

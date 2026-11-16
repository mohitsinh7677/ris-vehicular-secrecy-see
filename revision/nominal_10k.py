"""Nominal point repeated with 10,000 new channel realizations (paper Section 5.10).
Requires the trained actors written by revision/sweeps_ext.py to artifacts/actors/."""
import sys, json
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src')); sys.path.insert(0, str(REPO / 'revision'))
from physical_umi import Config, MLP, observation, channel_snapshot
from sweeps_ext import matched_ext


class _Agent:
    pass


def main():
    torch.set_num_threads(1)
    cfg = Config()
    sd = len(observation(cfg, channel_snapshot(cfg, np.random.default_rng(1)))); ad = 2 * cfg.m + cfg.n
    agents = []
    for k in range(5):
        a = _Agent(); a.actor = MLP(sd, ad, True)
        a.actor.load_state_dict(torch.load(REPO / 'artifacts' / 'actors' / f'actor_N128_seed{k}.pt')); agents.append(a)
    r = matched_ext(cfg, agents, channel_seed=99991, n_mc=10000)
    P = {'AO-RIS': cfg.total_power_w(True), 'Random-RIS': cfg.total_power_w(True),
         'No-RIS': cfg.total_power_w(False), 'TD3-RIS': cfg.total_power_w(True)}
    r['TD3-RIS'] = r.pop('TD3-seeds').mean(1)
    out = {m: {'mean': float(v.mean()), 'ci95': float(1.96 * v.std(ddof=1) / np.sqrt(len(v))), 'SEE': float(v.mean() / P[m])}
           for m, v in r.items()}
    d = r['AO-RIS'] - r['TD3-RIS']
    out['AO_minus_TD3'] = [float(d.mean()), float(1.96 * d.std(ddof=1) / np.sqrt(len(d)))]
    P0 = cfg.total_power_w(False)
    out['break_even_W'] = float(P0 * (out['AO-RIS']['mean'] / out['No-RIS']['mean'] - 1))
    (REPO / 'results' / 'verification' / 'nominal_10k.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()

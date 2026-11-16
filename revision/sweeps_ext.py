"""Re-execution of the published sweeps with per-channel, per-seed records saved.

Identical seeds, channels and training as run_sweeps.main(); the only addition is that each
TD3 seed's secrecy on each channel is stored (not just the five-seed average), and the trained
actors are saved. Used for reproducibility check, significance tests and seed-inclusive CIs.
"""
import sys, json, time
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / 'src'
sys.path.insert(0, str(SRC))
from physical_umi import (Config, channel_snapshot, observation, physical_action, secrecy,
                          ao_action, random_action, secrecy_surrogate_beamformer, eve_covariance)
from run_sweeps import N_VALUES, P_VALUES, SIGMAS, SPEEDS, DELAYS, train_short

OUT = REPO / 'results' / 'verification'
ACT = REPO / 'artifacts' / 'actors'
OUT.mkdir(parents=True, exist_ok=True); ACT.mkdir(parents=True, exist_ok=True)


def matched_ext(cfg, agents, channel_seed, n_mc=1000):
    rng = np.random.default_rng(channel_seed)
    K = len(agents)
    ao = np.empty(n_mc); rd = np.empty(n_mc); nr = np.empty(n_mc); td = np.empty((n_mc, K))
    for i in range(n_mc):
        s = channel_snapshot(cfg, rng)
        w, p = ao_action(cfg, s); ao[i] = secrecy(cfg, s, w, p)[0]
        w, p = random_action(cfg, s, rng); rd[i] = secrecy(cfg, s, w, p)[0]
        w = secrecy_surrogate_beamformer(cfg, s['hbu_obs'].conj(), eve_covariance(cfg, s, False))
        nr[i] = secrecy(cfg, s, w)[0]
        ob = torch.from_numpy(observation(cfg, s))[None, :]
        with torch.no_grad():
            acts = torch.stack([a.actor(ob)[0] for a in agents]).numpy()
        for k, a in enumerate(acts):
            w, p = physical_action(cfg, a); td[i, k] = secrecy(cfg, s, w, p)[0]
    return {'AO-RIS': ao, 'Random-RIS': rd, 'No-RIS': nr, 'TD3-seeds': td}


def main():
    cfg = Config()
    t0 = time.perf_counter()
    agents = {}
    for n in N_VALUES:
        agents[n] = []
        for i in range(5):
            ag, curve, nupd = train_short(replace(cfg, n=n), seed=19542 + 101 * i + 13 * n,
                                          steps=1200, updates=240)
            agents[n].append(ag)
            torch.save(ag.actor.state_dict(), ACT / f'actor_N{n}_seed{i}.pt')
        print('trained', n, round(time.perf_counter() - t0, 1), flush=True)
    sweeps = [('n', N_VALUES, 20.), ('pt_dbm', P_VALUES, 15.), ('n', N_VALUES, 15.),
              ('csi_std', SIGMAS, 15.), ('v_kmh', SPEEDS, 15.), ('delay_ms', DELAYS, 15.)]
    store = {}
    for sweep, values, pt in sweeps:
        c = replace(cfg, pt_dbm=pt)
        for x in values:
            cc = replace(c, **{sweep: float(x) if sweep != 'n' else int(x)})
            aa = agents[int(x)] if sweep == 'n' else agents[128]
            seed = 880021 + int(x * 1000) + len(sweep) * 90
            r = matched_ext(cc, aa, seed, 1000)
            key = f'{sweep}|{x}|{pt}'
            for m, v in r.items():
                store[f'{key}|{m}'] = v
        print('sweep', sweep, pt, round(time.perf_counter() - t0, 1), flush=True)
    np.savez_compressed(OUT / 'per_channel_records.npz', **store)
    (OUT / 'runtime.json').write_text(json.dumps({'seconds': time.perf_counter() - t0}))
    print('done', round(time.perf_counter() - t0, 1))


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()

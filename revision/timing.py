"""Per-decision CPU time of each method on the 1,000 nominal channels (paper Table 4).
Timings depend on the machine; the paper values were measured single-threaded on an Intel Xeon 2.1 GHz."""
import sys, time, json
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src')); sys.path.insert(0, str(REPO / 'revision'))
from physical_umi import (Config, MLP, observation, physical_action, ao_action, random_action,
                          secrecy_surrogate_beamformer, eve_covariance)
from ao_study import nominal_channels


def tm(chans, f):
    t = time.perf_counter()
    for s in chans:
        f(s)
    return 1e3 * (time.perf_counter() - t) / len(chans)


def main():
    torch.set_num_threads(1)
    cfg = Config(); ch = nominal_channels(cfg)
    sd = len(observation(cfg, ch[0])); ad = 2 * cfg.m + cfg.n
    actor = MLP(sd, ad, True); actor.load_state_dict(torch.load(REPO / 'artifacts' / 'actors' / 'actor_N128_seed0.pt')); actor.eval()
    rng = np.random.default_rng(3)

    def td3(s):
        with torch.no_grad():
            a = actor(torch.from_numpy(observation(cfg, s))[None])[0].numpy()
        physical_action(cfg, a)

    res = {}
    for _ in range(2):  # second (warm) pass is reported
        res['TD3 (one actor), ms'] = tm(ch, td3)
        res['AO-RIS 12 iterations, ms'] = tm(ch, lambda s: ao_action(replace(cfg, ao_iterations=12), s))
        res['AO-RIS 100 iterations, ms'] = tm(ch, lambda s: ao_action(replace(cfg, ao_iterations=100), s))
        res['Random-RIS, ms'] = tm(ch, lambda s: random_action(cfg, s, rng))
        res['No-RIS, ms'] = tm(ch, lambda s: secrecy_surrogate_beamformer(cfg, s['hbu_obs'].conj(), eve_covariance(cfg, s, False)))
    res['TD3 actor multiply-accumulate operations'] = sd * 256 + 256 * 256 + 256 * ad
    (REPO / 'results' / 'verification' / 'timing.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()

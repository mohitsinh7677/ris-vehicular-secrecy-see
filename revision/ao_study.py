"""AO-RIS convergence, multi-start and quantized-phase study (paper Section 5.10, Table 3).

Uses the exact nominal channel set of Table 2 (N=128, 15 dBm, 60 km/h, CSI 0.05, 0.1 ms):
channel_seed = 880021 + 128*1000 + len('n')*90, with the Random-RIS phase draw interleaved
exactly as in run_sweeps.matched(), so channels are identical to the published table.
"""
import sys, json, time, csv
from pathlib import Path
from dataclasses import replace
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / 'src'
sys.path.insert(0, str(SRC))
from physical_umi import (Config, channel_snapshot, effective_row, secrecy,
                          eve_covariance, secrecy_surrogate_beamformer)

OUT = REPO / 'results' / 'verification'
OUT.mkdir(parents=True, exist_ok=True)
MAXIT = 100
CHECK = [0, 1, 2, 3, 5, 8, 12, 20, 30, 50, 100]


def surrogate(cfg, s, w, phi, C):
    row = effective_row(s['hbu_obs'], s['hru_obs'], s['Hbr_obs'], phi)
    gu = abs(row @ w) ** 2 / cfg.noise_w()
    ge = float(np.real(np.vdot(w, C @ w))) / cfg.noise_w()
    return float(np.log2(1 + gu) - np.log2(1 + ge))


def ao_trace(cfg, s, phi0, maxit=MAXIT, bits=None):
    """Same update rule as physical_umi.ao_action; records surrogate and true secrecy
    after the beamformer step of every iteration (iteration k = k phase updates)."""
    hb, hr, H = s['hbu_obs'], s['hru_obs'], s['Hbr_obs']
    C = eve_covariance(cfg, s, True)
    phi = phi0.copy()
    sur, sec = [], []
    for k in range(maxit + 1):
        w = secrecy_surrogate_beamformer(cfg, effective_row(hb, hr, H, phi), C)
        sur.append(surrogate(cfg, s, w, phi, C))
        sec.append(secrecy(cfg, s, w, phi)[0])
        if k == maxit:
            break
        q = np.conj(hr) * (H @ w)
        direct = np.vdot(hb, w)
        phase0 = np.angle(direct) if abs(direct) > 1e-22 else 0.
        ang = phase0 - np.angle(q)
        if bits is not None:
            L = 2 ** bits
            ang = np.round(ang / (2 * np.pi / L)) * (2 * np.pi / L)
        phi = np.exp(1j * ang)
    return np.array(sur), np.array(sec), w, phi


def nominal_channels(cfg, n_mc=1000):
    rng = np.random.default_rng(880021 + int(128 * 1000) + len('n') * 90)
    out = []
    for _ in range(n_mc):
        s = channel_snapshot(cfg, rng)
        rng.uniform(-np.pi, np.pi, cfg.n)  # Random-RIS draw consumed in matched()
        out.append(s)
    return out


def main():
    cfg = Config()  # nominal: N=128, 15 dBm, 60 km/h, 0.05, 0.1 ms
    chans = nominal_channels(cfg)
    L = len(chans)
    start_rng = np.random.default_rng(777)
    n_starts = 10
    sur_all = np.zeros((L, MAXIT + 1)); sec_all = np.zeros((L, MAXIT + 1))
    ms_sec_sur = np.zeros(L)      # multi-start, chosen by surrogate (deployable)
    ms_sec_best = np.zeros(L)     # multi-start oracle (chosen by true secrecy; upper bound only)
    single12 = np.zeros(L); single100 = np.zeros(L)
    t_ao12 = 0.; t_ao100 = 0.; t_ms = 0.
    q_res = {b: np.zeros(L) for b in (1, 2, 3)}
    for i, s in enumerate(chans):
        phi1 = np.ones(cfg.n, dtype=complex)
        t = time.perf_counter(); ao_trace(cfg, s, phi1, maxit=12); t_ao12 += time.perf_counter() - t
        t = time.perf_counter(); sur, sec, *_ = ao_trace(cfg, s, phi1); t_ao100 += time.perf_counter() - t
        sur_all[i] = sur; sec_all[i] = sec
        single12[i] = sec[12]; single100[i] = sec[MAXIT]
        # multi-start: all-one start + 9 random starts, each run 100 iterations
        t = time.perf_counter()
        cands = [(sur[-1], sec[-1])]
        for _ in range(n_starts - 1):
            p0 = np.exp(1j * start_rng.uniform(-np.pi, np.pi, cfg.n))
            su, se, *_ = ao_trace(cfg, s, p0)
            cands.append((su[-1], se[-1]))
        t_ms += time.perf_counter() - t
        ms_sec_sur[i] = max(cands, key=lambda z: z[0])[1]
        ms_sec_best[i] = max(z[1] for z in cands)
        for b in q_res:
            _, se, *_ = ao_trace(cfg, s, phi1, maxit=12, bits=b)
            q_res[b][i] = se[-1]
        if (i + 1) % 200 == 0:
            print('channels', i + 1, flush=True)

    def ci(v):
        return float(1.96 * v.std(ddof=1) / np.sqrt(len(v)))

    conv = []
    for k in CHECK:
        conv.append({'iterations': k, 'mean_true_secrecy': float(sec_all[:, k].mean()),
                     'ci95_halfwidth': ci(sec_all[:, k]),
                     'mean_surrogate': float(sur_all[:, k].mean())})
    # surrogate monotonicity check
    d = np.diff(sur_all, axis=1)
    mono_viol = float((d < -1e-9).mean())
    # relative surrogate change between iterations 11->12
    rel12 = float(np.median(np.abs(sur_all[:, 12] - sur_all[:, 11]) / np.maximum(1e-12, np.abs(sur_all[:, 12]))))
    diff_100_12 = single100 - single12
    diff_ms_12 = ms_sec_sur - single12
    res = {
        'channels': L,
        'convergence': conv,
        'surrogate_nonincrease_fraction_of_steps': mono_viol,
        'median_relative_surrogate_change_iter11_to_12': rel12,
        'AO12_mean': float(single12.mean()), 'AO12_ci': ci(single12),
        'AO100_mean': float(single100.mean()), 'AO100_minus_AO12_mean': float(diff_100_12.mean()),
        'AO100_minus_AO12_paired_ci': ci(diff_100_12),
        'multistart10_by_surrogate_mean': float(ms_sec_sur.mean()),
        'multistart_minus_AO12_mean': float(diff_ms_12.mean()),
        'multistart_minus_AO12_paired_ci': ci(diff_ms_12),
        'multistart10_oracle_true_best_mean_UPPER_BOUND_NOT_DEPLOYABLE': float(ms_sec_best.mean()),
        'quantized_AO12_mean': {f'{b}-bit': float(v.mean()) for b, v in q_res.items()},
        'quantized_AO12_ci': {f'{b}-bit': ci(v) for b, v in q_res.items()},
        'cpu_ms_per_channel': {'AO_12it': 1e3 * t_ao12 / L, 'AO_100it': 1e3 * t_ao100 / L,
                               'AO_10start_x100it_extra9': 1e3 * t_ms / L},
    }
    (OUT / 'ao_study.json').write_text(json.dumps(res, indent=2))
    with open(OUT / 'ao_convergence.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=conv[0].keys()); w.writeheader(); w.writerows(conv)
    np.savez_compressed(OUT / 'ao_traces.npz', sur=sur_all, sec=sec_all, ms_sur=ms_sec_sur,
                        ms_best=ms_sec_best, **{f'q{b}': v for b, v in q_res.items()})
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()

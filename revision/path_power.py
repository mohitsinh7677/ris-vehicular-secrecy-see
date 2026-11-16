"""Direct versus reflected path power at the nominal point (paper Section 6.1)."""
import sys, json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src')); sys.path.insert(0, str(REPO / 'revision'))
from physical_umi import Config, ao_action
from ao_study import nominal_channels

cfg = Config(); ch = nominal_channels(cfg)[:500]; rng = np.random.default_rng(1)
f = lambda x: float(10 * np.log10(np.mean(x)))
refl = lambda ph, hr, H, w: np.einsum('n,n,nm->m', hr.conj(), ph, H) @ w
d, rr, ra, ed, er = [], [], [], [], []
for s in ch:
    w, p = ao_action(cfg, s)
    phr = np.exp(1j * rng.uniform(-np.pi, np.pi, cfg.n))
    d.append(abs(s['hbu_true'].conj() @ w) ** 2)
    rr.append(abs(refl(phr, s['hru_true'], s['Hbr_true'], w)) ** 2)
    ra.append(abs(refl(p, s['hru_true'], s['Hbr_true'], w)) ** 2)
    ed.append(abs(s['hbe_true'].conj() @ w) ** 2)
    er.append(abs(refl(p, s['hre_true'], s['Hbr_true'], w)) ** 2)
res = {'user_direct_dB': f(d), 'user_reflected_random_phases_dB': f(rr), 'user_reflected_AO_phases_dB': f(ra),
       'eve_direct_dB': f(ed), 'eve_reflected_AO_phases_dB': f(er)}
(REPO / 'results' / 'verification' / 'path_power.json').write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))

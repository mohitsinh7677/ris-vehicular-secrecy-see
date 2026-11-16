"""Reproducibility check, significance tests and seed-inclusive intervals (paper Sections 5.9-5.10)."""
import json, csv
from pathlib import Path
import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
W = REPO
REPRO = W / 'results' / 'main'
D = np.load(W / 'results' / 'verification' / 'per_channel_records.npz')
keys = sorted({k.rsplit('|', 1)[0] for k in D.files})


def get(point):
    return {m: D[f'{point}|{m}'] for m in ('AO-RIS', 'Random-RIS', 'No-RIS', 'TD3-seeds')}


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1., run)
    return adj


# 1. reproducibility against published CSV
orig = list(csv.DictReader(open(REPRO / 'measured_sweep_summary.csv')))
maxdiff = {}
for row in orig:
    pt = f"{row['sweep']}|{row['x']}|{float(row['pt_dbm_nominal'])}"
    r = get(pt)
    new = r['TD3-seeds'].mean(axis=1) if row['method'] == 'TD3-RIS' else r[row['method']]
    d = abs(new.mean() - float(row['mean_secrecy']))
    maxdiff[row['method']] = max(maxdiff.get(row['method'], 0), d)

# 2. significance tests at every sweep point (paired over the same 1,000 channels)
pairs = [('AO-RIS', 'No-RIS'), ('AO-RIS', 'Random-RIS'), ('Random-RIS', 'No-RIS'),
         ('AO-RIS', 'TD3-RIS'), ('No-RIS', 'TD3-RIS'), ('Random-RIS', 'TD3-RIS')]
rows = []
for pt in keys:
    r = get(pt); r['TD3-RIS'] = r['TD3-seeds'].mean(axis=1)
    for a, b in pairs:
        d = r[a] - r[b]
        t = stats.ttest_rel(r[a], r[b])
        nz = d[np.abs(d) > 1e-12]
        wil = stats.wilcoxon(r[a], r[b], zero_method='wilcox') if len(nz) > 0 else None
        rows.append({'point': pt, 'A': a, 'B': b, 'mean_A_minus_B': float(d.mean()),
                     'ci95_paired': float(1.96 * d.std(ddof=1) / np.sqrt(len(d))),
                     'cohen_dz': float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else 0.0,
                     't_p': float(t.pvalue), 'wilcoxon_p': float(wil.pvalue) if wil else 1.0})
adj_t = holm([r['t_p'] for r in rows]); adj_w = holm([r['wilcoxon_p'] for r in rows])
for r, a, b in zip(rows, adj_t, adj_w):
    r['t_p_holm'] = float(a); r['wilcoxon_p_holm'] = float(b)
with open(W / 'results' / 'verification' / 'significance_all_points.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)


def summary_pair(a, b):
    sel = [r for r in rows if r['A'] == a and r['B'] == b]
    sig = sum(r['wilcoxon_p_holm'] < .05 and r['t_p_holm'] < .05 for r in sel)
    pos = sum(r['mean_A_minus_B'] > 0 for r in sel)
    return {'points': len(sel), 'significant_both_tests_holm05': sig, 'A_better_points': pos}


# 3. seed-inclusive intervals at nominal (N=128, 15 dBm): hierarchical bootstrap (seeds, then channels)
nom = get('n|128|15.0')
td = nom['TD3-seeds']; ao = nom['AO-RIS']; nr = nom['No-RIS']; rd = nom['Random-RIS']
rng = np.random.default_rng(2026)
B = 20000; L, K = td.shape
bt_td = np.empty(B); bt_diff = np.empty(B); bt_ao = np.empty(B)
for i in range(B):
    ks = rng.integers(0, K, K); ls = rng.integers(0, L, L)
    m = td[np.ix_(ls, ks)].mean(); bt_td[i] = m; bt_ao[i] = ao[ls].mean(); bt_diff[i] = bt_ao[i] - m
seed_means = td.mean(axis=0)
tcrit = stats.t.ppf(.975, K - 1)
nominal = {
    'means': {'AO-RIS': float(ao.mean()), 'Random-RIS': float(rd.mean()), 'No-RIS': float(nr.mean()), 'TD3-RIS': float(td.mean())},
    'TD3_seed_means': seed_means.tolist(),
    'TD3_t_interval_over_seeds_df4': [float(seed_means.mean() - tcrit * seed_means.std(ddof=1) / np.sqrt(K)),
                                      float(seed_means.mean() + tcrit * seed_means.std(ddof=1) / np.sqrt(K))],
    'TD3_hier_bootstrap95': [float(np.percentile(bt_td, 2.5)), float(np.percentile(bt_td, 97.5))],
    'TD3_channel_only95': [float(td.mean(1).mean() - 1.96 * td.mean(1).std(ddof=1) / np.sqrt(L)),
                           float(td.mean(1).mean() + 1.96 * td.mean(1).std(ddof=1) / np.sqrt(L))],
    'AO_minus_TD3_hier_bootstrap95': [float(np.percentile(bt_diff, 2.5)), float(np.percentile(bt_diff, 97.5))],
    'AO_minus_TD3_channel_only95_published': [0.3421, 0.3652],
    'Random_minus_NoRIS': {r2['A'] + '-' + r2['B']: r2 for r2 in rows if r2['point'] == 'n|128|15.0' and r2['A'] == 'Random-RIS' and r2['B'] == 'No-RIS'},
    'AO_minus_NoRIS': [r2 for r2 in rows if r2['point'] == 'n|128|15.0' and r2['A'] == 'AO-RIS' and r2['B'] == 'No-RIS'][0],
}
# SEE significance at nominal (power constants fixed => same test on R/P per channel)
P_ris, P_no = 0.8183507902905252, 0.5903507902905252
see_t = stats.ttest_rel(nr / P_no, ao / P_ris); see_w = stats.wilcoxon(nr / P_no, ao / P_ris)
nominal['SEE_NoRIS_minus_AO'] = {'mean': float((nr / P_no - ao / P_ris).mean()), 't_p': float(see_t.pvalue), 'wilcoxon_p': float(see_w.pvalue)}
out = {'reproduction_max_abs_diff_vs_published_mean_secrecy': maxdiff,
       'pair_summaries': {f'{a} vs {b}': summary_pair(a, b) for a, b in pairs},
       'nominal': nominal, 'tests': len(rows)}
(W / 'results' / 'verification' / 'stats_summary.json').write_text(json.dumps(out, indent=2, default=float))
print(json.dumps(out, indent=2, default=float))

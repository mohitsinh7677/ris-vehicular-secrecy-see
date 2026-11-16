"""Regenerate Figs. 1-7 from the published CSV records (values unchanged).
Printed width 5.75 in (as embedded in the manuscript), 600 dpi, 9 pt text, no banners."""
import csv, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / 'results' / 'main'
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader(open(SRC / 'measured_sweep_summary.csv')))
ck = list(csv.DictReader(open(SRC / 'five_seed_training_checkpoints.csv')))

plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
                     'legend.fontsize': 8.5, 'font.family': 'DejaVu Sans', 'axes.linewidth': 0.8,
                     'axes.edgecolor': '#555555', 'xtick.color': '#333333', 'ytick.color': '#333333'})
W, H = 5.75, 3.4
STY = {'TD3-RIS':    dict(c='#0072B2', marker='o', ls='-',  mfc='#0072B2', label='TD3 baseline'),
       'AO-RIS':     dict(c='#D55E00', marker='s', ls='--', mfc='#D55E00', label='AO-RIS'),
       'Random-RIS': dict(c='#009E73', marker='^', ls='-.', mfc='white',   label='Random-RIS'),
       'No-RIS':     dict(c='#CC79A7', marker='D', ls=':',  mfc='#CC79A7', label='No-RIS')}
ORDER = ['AO-RIS', 'Random-RIS', 'No-RIS', 'TD3-RIS']
NAMES = {1: 'R_Fig1_Secrecy_vs_RIS_Size.png', 2: 'R_Fig2_SEE_vs_Transmit_Power.png',
         3: 'R_Fig3_SEE_vs_RIS_Size.png', 4: 'R_Fig4_Secrecy_vs_CSI_Error.png',
         5: 'R_Fig5_Secrecy_vs_Vehicle_Speed.png', 6: 'R_Fig6_Secrecy_vs_Delay.png',
         7: 'R_Fig7_TD3_Training_Behavior.png'}


def series(sweep, pt, metric):
    out = {}
    for m in ORDER:
        d = sorted([(float(r['x']), float(r[metric])) for r in rows
                    if r['sweep'] == sweep and float(r['pt_dbm_nominal']) == pt and r['method'] == m])
        out[m] = np.array(d)
    return out


def finish(fig, ax, name, xlabel, ylabel, xticks=None):
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if xticks is not None: ax.set_xticks(xticks)
    ax.grid(True, color='#e3e3e3', lw=0.6); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, ncol=4, loc='lower center', bbox_to_anchor=(0.5, 1.0), handlelength=2.6, columnspacing=1.2)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / name, dpi=600); plt.close(fig)


def plot_sweep(num, sweep, pt, metric, xlabel, ylabel, xticks=None, shade=None):
    fig, ax = plt.subplots(figsize=(W, H))
    if shade:
        ax.axvspan(*shade, color='#f0f0f0', lw=0)
        ax.text(sum(shade) / 2, 0.97, 'not used\nin training', transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=8, color='#555555')
    for m in ORDER:
        d = series(sweep, pt, metric)[m]; s = STY[m]
        ax.plot(d[:, 0], d[:, 1], color=s['c'], marker=s['marker'], ls=s['ls'], mfc=s['mfc'],
                mec=s['c'], lw=1.6, ms=6, mew=1.2, label=s['label'])
    finish(fig, ax, NAMES[num], xlabel, ylabel, xticks)


SR = 'Average secrecy rate (bit/s/Hz)'; SE = 'Secrecy energy efficiency (bit/J/Hz)'
plot_sweep(1, 'n', 20., 'mean_secrecy', 'Number of RIS elements $N$', SR, [32, 64, 128, 256])
plot_sweep(2, 'pt_dbm', 15., 'mean_see', 'Radiated transmit power (dBm)', SE, [0, 5, 10, 15, 20, 25, 30])
plot_sweep(3, 'n', 15., 'mean_see', 'Number of RIS elements $N$', SE, [32, 64, 128, 256])
plot_sweep(4, 'csi_std', 15., 'mean_secrecy', 'Relative CSI-error standard deviation $\\sigma_e$', SR, [0, .02, .05, .1, .15, .2])
plot_sweep(5, 'v_kmh', 15., 'mean_secrecy', 'Vehicle speed (km/h)', SR, [30, 45, 60, 75, 90, 105, 120], shade=(97.5, 127.5))
plot_sweep(6, 'delay_ms', 15., 'mean_secrecy', 'CSI feedback/control delay (ms)', SR, [0, .1, .25, .5, .75, 1.0])

# Fig. 7: training behaviour
xs = sorted({int(r['transitions']) for r in ck})
M = np.array([[float(r['heldout_secrecy']) for r in ck if int(r['seed_idx']) == k] for k in range(5)])
mean, sd = M.mean(0), M.std(0, ddof=1)
ao = [float(r['mean_secrecy']) for r in rows if r['sweep'] == 'n' and r['x'] == '128'
      and float(r['pt_dbm_nominal']) == 15. and r['method'] == 'AO-RIS'][0]
fig, ax = plt.subplots(figsize=(W, H))
ax.fill_between(xs, mean - sd, mean + sd, color='#0072B2', alpha=0.18, lw=0, label='TD3 baseline, $\\pm$1 SD (5 seeds)')
ax.plot(xs, mean, color='#0072B2', marker='o', ms=6, lw=1.6, label='TD3 baseline, mean')
ax.axhline(ao, color='#D55E00', ls='--', lw=1.6, label='AO-RIS (nominal)')
ax.set_xlim(-40, 1240); ax.set_ylim(0, 0.45)
finish(fig, ax, NAMES[7], 'Training transitions (independent one-step)', 'Held-out secrecy rate (bit/s/Hz)', xs)
print('ok')

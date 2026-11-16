"""Recompute paper's two added numerical findings from archived executed CSVs.
This is postprocessing, not a new simulation/training run.
"""
from pathlib import Path
import csv
base=Path(__file__).resolve().parents[1]/'results'/'main'
with (base/'measured_sweep_summary.csv').open(newline='') as fh:
    rows=list(csv.DictReader(fh))
q={row['method']:row for row in rows if row['sweep']=='n' and row['x']=='128' and row['pt_dbm_nominal']=='15.0'}
assert set(q)=={'AO-RIS','No-RIS','Random-RIS','TD3-RIS'}
P0=float(q['No-RIS']['power_w'])
R_AO=float(q['AO-RIS']['mean_secrecy'])
R_No=float(q['No-RIS']['mean_secrecy'])
R_TD3=float(q['TD3-RIS']['mean_secrecy'])
P_RIS_actual=float(q['AO-RIS']['power_w'])-P0
P_RIS_break_even=P0*(R_AO/R_No-1)
with (base/'paired_td3_ao_differences.csv').open(newline='') as fh:
    pairs=list(csv.DictReader(fh))
row=next(x for x in pairs if x['sweep']=='n' and x['x']=='128' and x['pt_dbm_nominal']=='15.0')
difference=-float(row['td3_minus_ao_mean'])
halfwidth=float(row['paired_ci95_halfwidth'])
assert abs(R_AO-R_TD3-difference)<1e-10
print(f'Nominal P0: {P0:.9f} W')
print(f'Nominal actual RIS overhead: {P_RIS_actual:.9f} W')
print(f'Conditional break-even RIS overhead: {P_RIS_break_even:.9f} W')
print(f'Paired AO-minus-TD3 difference: {difference:.9f} bit/s/Hz')
print(f'Approximate 95% channel-only interval: [{difference-halfwidth:.9f}, {difference+halfwidth:.9f}] bit/s/Hz')
print('Note: channel-sampling interval only; the seed-inclusive interval is computed by revision/stats.py.')

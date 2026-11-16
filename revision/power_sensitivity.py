"""Break-even RIS allowance and SEE ranking versus power parameters (paper Section 5.12, Table 6).
Uses only the nominal secrecy rates in results/main/measured_sweep_summary.csv."""
import csv, json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader(open(REPO / 'results' / 'main' / 'measured_sweep_summary.csv')))
q = {r['method']: float(r['mean_secrecy']) for r in rows if r['sweep'] == 'n' and r['x'] == '128' and r['pt_dbm_nominal'] == '15.0'}
R_AO, R_NO = q['AO-RIS'], q['No-RIS']
PT = 10 ** ((15 - 30) / 10)  # 15 dBm in W
g = R_AO / R_NO - 1
ETA = [0.20, 0.30, 0.35, 0.40, 0.50]; PBS = [0.3, 0.4, 0.5, 0.6, 0.8]
table = {f'eta={e}': {f'P_BS={b}': round((PT / e + b) * g, 3) for b in PBS} for e in ETA}
P0 = PT / 0.35 + 0.5
allow = P0 * g
res = {'break_even_allowance_W_table6': table, 'nominal_allowance_W': allow,
       'assumed_RIS_overhead_W': 0.1 + 128 * 0.001,
       'max_per_element_W_with_0.1W_controller': (allow - 0.1) / 128,
       'max_per_element_W_with_0.01W_controller': (allow - 0.01) / 128,
       'AO_SEE_example_0.01W_ctrl_0.5mW_element': R_AO / (P0 + 0.01 + 128 * 0.0005),
       'NoRIS_SEE_nominal': R_NO / P0}
(REPO / 'results' / 'verification' / 'power_sensitivity.json').write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))

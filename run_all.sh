#!/usr/bin/env bash
# Reproduce all results of the paper. Run from the repository root.
#   bash run_all.sh            -> everything except the long TD3 ablations (about 20 min on 1 CPU core)
#   bash run_all.sh --ablation -> also the TD3 ablation of Table 5 (several additional hours)
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

python -m pytest -q tests                      # physics and code checks
python src/run_sweeps.py --out results/main    # Figs. 1-7, Table 2 (sweep CSVs)
python revision/sweeps_ext.py                  # per-channel/per-seed records + trained actors
python revision/stats.py                       # significance tests, seed-inclusive intervals
python revision/ao_study.py                    # Table 3: AO convergence, multi-start, quantization
python revision/failure_decomp.py              # Section 5.11: TD3 failure decomposition
python revision/nominal_10k.py                 # Section 5.10: 10,000-channel nominal repetition
python revision/power_sensitivity.py           # Table 6: power-parameter sensitivity
python revision/path_power.py                  # Section 6.1: direct vs reflected path power
python revision/timing.py                      # Table 4: CPU time (machine dependent)
python revision/make_figures.py                # Figs. 1-7
python scripts/recalculate_findings.py         # break-even allowance and paired AO-TD3 interval

if [[ "${1:-}" == "--ablation" ]]; then
  for v in paper_1200 steps_5000 phase_only_5000; do python revision/td3_ablation.py --variant "$v"; done
  for v in steps_20000 eve_obs_20000 phase2pi_20000; do python revision/td3_ablation.py --variant "$v" --seeds 3; done
fi

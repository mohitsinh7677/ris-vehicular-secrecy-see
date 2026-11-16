# Energy-Aware Secrecy in RIS-Assisted Vehicular Networks

Simulation code and results for the paper:

> **Energy-Aware Secrecy in RIS-Assisted Vehicular Networks: A Simulation Study with TD3 Baseline**
> International Journal of Wireless and Microwave Technologies (IJWMT), manuscript PAPER024821.
> Citation details will be added after publication.

The code simulates a 28 GHz downlink in which a base station with eight antennas serves a moving vehicle through a roadside reconfigurable intelligent surface (RIS) in the presence of a passive eavesdropper. It compares four methods on matched channel realizations:

- **AO-RIS**: alternating optimization of the beamformer and the RIS phases
- **Random-RIS**: random RIS phases with an optimized beamformer
- **No-RIS**: direct transmission without the surface
- **TD3 baseline**: a twin-delayed deep deterministic policy gradient controller trained with terminal one-step transitions (a contextual-bandit formulation)

It reports the secrecy rate, the secrecy energy efficiency (SEE) and the break-even RIS hardware power, together with the verification experiments of the paper (AO convergence, significance tests, TD3 ablation and power-parameter sensitivity).

## Repository structure

```
src/                 simulator, benchmark methods, TD3 baseline, sensitivity sweeps
revision/            verification experiments, statistics and figure generation
scripts/             recalculation of the break-even allowance and paired interval
tests/               physics and code checks (pytest)
results/main/        sweep records behind Figs. 1-7 and Table 2
results/verification/ AO check, significance tests, 10,000-channel check, timing, power sensitivity
results/ablation/    TD3 ablation records and failure analysis (Table 5)
figures/             Figs. 1-7 as used in the paper (600 dpi)
run_all.sh           runs the complete pipeline
```

Trained TD3 actors are written to `artifacts/actors/` when the pipeline runs; they are not stored in the repository because they are regenerated deterministically.

## Where each result comes from

| Paper item | Script | Output |
|---|---|---|
| Figs. 1-6, Table 2 (means, channel intervals, SEE) | `src/run_sweeps.py` | `results/main/measured_sweep_summary.csv` |
| Fig. 7 (training behaviour) | `src/run_sweeps.py` | `results/main/five_seed_training_checkpoints.csv` |
| Break-even allowance (Eq. 27), paired AO-TD3 interval | `scripts/recalculate_findings.py` | printed |
| Table 2 seed-inclusive interval; significance tests (Section 5.10) | `revision/sweeps_ext.py`, `revision/stats.py` | `results/verification/stats_summary.json`, `significance_all_points.csv` |
| Table 3 (AO convergence, multi-start, phase quantization) | `revision/ao_study.py` | `results/verification/ao_study.json`, `ao_convergence.csv` |
| Table 4 (CPU time) | `revision/timing.py` | `results/verification/timing.json` |
| 10,000-channel repetition (Section 5.10) | `revision/nominal_10k.py` | `results/verification/nominal_10k.json` |
| Table 5 (TD3 ablation) | `revision/td3_ablation.py` | `results/ablation/<variant>.json` |
| Failure analysis (Section 5.11) | `revision/failure_decomp.py` | `results/ablation/failure_decomposition.json` |
| Table 6 (power-parameter sensitivity) | `revision/power_sensitivity.py` | `results/verification/power_sensitivity.json` |
| Path-power comparison (Section 6.1) | `revision/path_power.py` | `results/verification/path_power.json` |
| Figs. 1-7 (image files) | `revision/make_figures.py` | `figures/` |

## Reproducing the results

Requirements: Python 3.13 and the packages in `requirements.txt`. A CPU is sufficient; no GPU is used.

```bash
pip install -r requirements.txt
bash run_all.sh              # all results except the long TD3 ablation
bash run_all.sh --ablation   # also the TD3 ablation of Table 5
```

All random seeds are fixed in the code. On the reference platform (Intel Xeon 2.1 GHz, two cores, 7.8 GB RAM, no GPU; Python 3.13, NumPy 2.5.3, SciPy 1.18.1, PyTorch 2.15.0), the pipeline reproduces every value reported in the paper exactly. `run_all.sh` without the ablation takes about 20 minutes on one core. The 20,000-transition ablation variants take roughly 12 minutes per seed. CPU timings (Table 4) depend on the machine, and other PyTorch or BLAS versions may cause small numerical differences.

## Model scope

- Direct links use the 3GPP TR 38.901 urban-microcell path-loss expressions; the RIS hops use a simplified geometric Rician model. The simulator is not a spatially consistent 3GPP channel generator.
- Each evaluation sample is an independent channel snapshot. Doppler aging is applied within a snapshot, between CSI estimation and action application; continuous vehicle trajectories are not simulated.
- The eavesdropper's instantaneous channel is not available to any method; it is used only to compute the offline reward and the evaluation metrics.
- The hardware-power coefficients are assumed values; their basis and the sensitivity of the results to them are given in Sections 5.1 and 5.12 of the paper.

## Licence

Released under the MIT Licence; see `LICENSE`.

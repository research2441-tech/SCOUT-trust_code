# SCOUT: entity-relative Bayesian trust for IoT users, devices and gateways

Reference implementation of the experiments in *"Entity-Relative Bayesian Trust with Responsibility Attribution and
Topology-Priced Containment for IoT Users, Devices and Gateways"*.

SCOUT keeps a compromise posterior for every user account and device. Flow evidence from an open-set screen is
turned into a truncated beta-binomial evidence ratio against an entity-relative null, a one-sided negative-binomial
volume ratio is added, evidence is shared between accounts and devices by responsibility weights, and a
benign/active/dormant hidden Markov filter turns it into a posterior. Neighbours of suspicious devices receive a
higher compromise hazard (scrutiny, not blame). A Bayes-cost policy chooses allow, restrict, egress isolation or
quarantine and prices the transit loss of isolating a cut vertex.

## Files (flat layout)

| File | Purpose |
|---|---|
| `data.py` | TON_IoT / CICIoT2023 loading, de-duplication, leave-one-family-out partitions |
| `flow.py` | E1: open-set flow screen (XGBoost, kNN, iForest, weighted sum, Cauchy fusion), stores conformal pools |
| `world.py` | trace-driven entity worlds (topologies, churn, users, adversaries, common random numbers) |
| `trust.py` | SCOUT and the baselines (EWMA, Beta, asymmetric, CUSUM, Beta+diffusion, Shiryaev, learned risk) |
| `sim.py` | closed-loop simulation, threshold and cost-aware policies, metrics |
| `entity.py` | E2: validation-only selection and budget-matched tuning, test runs |
| `rerun.py` | open-loop trust quality without lateral spread; blame-diffusion ablation |
| `supp.py` | E3-E9: sensitivity, reliability, duty-cycle sweep, robustness, scale/runtime, budget sweep |
| `analysis.py` | every table, figure and number of the paper from the raw result files |
| `fig_arch.py` | framework and filter diagrams |
| `run_entity.sh`, `run_after.sh`, `run_supp.sh` | drivers (two worker processes each) |
| `requirements.txt` | package versions used |

## Data

Place `TON_IoT_Train_Test_Network.csv` (TON_IoT network release) and a CICIoT2023 parquet with columns
`Label`, `Label_orig`, `attack_class`, `label` and the 46 numeric features in the data directory set at the top of
`data.py`. The datasets are not redistributed here; obtain them from their publishers.

## Reproduce

```bash
pip install -r requirements.txt
python flow.py TON_IoT && python flow.py CICIoT2023      # E1, writes pool_*.npz
./run_entity.sh                                          # E2 (main, 32 folds)
./run_after.sh                                           # re-run pass and E3-E9
python analysis.py                                       # tables and figures
```

Every random draw in a world is keyed on (world seed, window), so all methods face identical traffic; results are
deterministic for a given package set. Runtime on two CPU cores: about 15 min for E1, 3 h for E2, 2 h for the rest.

## License

MIT.

## Paths

`SCOUT_DATA` (default `data`), `SCOUT_RESULTS` (default `results`) and `SCOUT_FIGURES` (default `figures`) set the
input, result and figure directories; create `results` and `figures` before the first run.

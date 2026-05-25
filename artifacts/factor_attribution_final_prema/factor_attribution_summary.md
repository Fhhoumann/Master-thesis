# Factor Attribution Summary

This folder contains FF3, Carhart, and FF5 factor-attribution outputs for the frozen final specification used in the main thesis comparison.

## Sample and alignment
- Strategy sample: 1999-12-31 to 2024-11-29
- Observations per strategy: 300
- Attribution factor panel rows: 420
- Attribution factor panel dates: 1989-12-29 to 2024-11-29
- Factor set includes `mkt_rf`, `smb`, `hml`, `umd`, `rmw`, `cma`, and `rf`.

## Full-sample results

### ff3

| run_key | alpha_annualized | alpha_tstat | r2 | beta_mkt_rf | beta_smb | beta_hml | beta_umd | beta_rmw | beta_cma |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Elastic Net | -0.0772 | -2.1339 | 0.6537 | 1.3524 | 0.6603 | -0.4961 | nan | nan | nan |
| Random Forest | -0.0054 | -0.1875 | 0.5962 | 1.0240 | 0.2743 | -0.2799 | nan | nan | nan |
| Torch (gamma=5.0) | 0.0313 | 1.5684 | 0.7635 | 0.9864 | 0.3130 | -0.2275 | nan | nan | nan |

### carhart4

| run_key | alpha_annualized | alpha_tstat | r2 | beta_mkt_rf | beta_smb | beta_hml | beta_umd | beta_rmw | beta_cma |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Elastic Net | -0.0543 | -1.4858 | 0.6857 | 1.1914 | 0.7124 | -0.5655 | -0.3392 | nan | nan |
| Random Forest | 0.0101 | 0.3383 | 0.6198 | 0.9216 | 0.3075 | -0.3241 | -0.2158 | nan | nan |
| Torch (gamma=5.0) | 0.0247 | 1.1956 | 0.7689 | 1.0287 | 0.2993 | -0.2092 | 0.0893 | nan | nan |

### ff5

| run_key | alpha_annualized | alpha_tstat | r2 | beta_mkt_rf | beta_smb | beta_hml | beta_umd | beta_rmw | beta_cma |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Elastic Net | -0.0254 | -0.6899 | 0.6952 | 1.2655 | 0.3137 | -0.1475 | nan | -0.7870 | -0.1760 |
| Random Forest | 0.0306 | 0.9760 | 0.6338 | 0.9733 | 0.0258 | -0.0810 | nan | -0.5595 | -0.0114 |
| Torch (gamma=5.0) | 0.0437 | 2.1635 | 0.7706 | 0.9722 | 0.2202 | -0.1742 | nan | -0.2069 | 0.0432 |

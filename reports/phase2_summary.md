# Bond ML Phase 2 — Summary Report

**Generated**: 2026-05-27
**Dataset**: 168 bonds, 72 issuers, 6 annual Q1 snapshots (2021–2026)
**Excluded issuers** (no US-exchange MI Key): Expand Energy Corp., Moog Inc., Six Flags Entertainment Corp.

---

## Data Coverage

| Item | Count |
|------|-------|
| Bond observations | 168 |
| Unique issuers | 72 |
| Issuers with OAS data | 60 |
| Bonds with valid spread | 118 (Z/G/A-spread) |
| Historical periods | 6 (2021Q1–2026Q1) |
| Downgrade labels (positive) | 62 of 335 issuer-periods (18.5%) |

---

## Model 1 — Spread Prediction

**Algorithm**: LightGBM Regressor with monotonic constraints
**Validation**: Issuer-grouped 5-fold cross-validation

| Metric | Mean | ± Std |
|--------|------|-------|
| Grouped-CV R² | 0.275 | ±0.193 |
| RMSE (bps) | 163.3 | ±88.3 |
| Spearman ρ | 0.762 | ±0.043 |

*Note: 118 bonds had valid spread data (Z/G/A-spread). OAS was not available in this CIQ export.*

---

## Model 2 — Downgrade Risk

**Algorithm**: LightGBM classifier + Cox PH ensemble
**Validation**: Time-forward holdout (train 2021–2023, val 2024, test 2025)

| Metric | Test |
|--------|------|
| PR-AUC (primary) | 0.505 |
| ROC-AUC | 0.548 |
| Brier Score | 0.394 |
| Recall @ Top Decile | 0.09 |
| Base Rate | 47.8% |

*Top features: EBITDA/Interest coverage trajectory (Δ2yr), leverage trajectory (Δ2yr), FCF/Debt.*

---

## RV Framework — Top Longs (Cheap + Safe)

| issuer_name                 | sp_rating   |   actual_spread |   cheapness_bps |   downgrade_prob_4q |   modified_duration |
|:----------------------------|:------------|----------------:|----------------:|--------------------:|--------------------:|
| Fossil Group, Inc.          | CCC+        |        1430.12  |       1142.8    |           0.0161874 |             2.55098 |
| EQT Corporation             | BBB-        |        1196.31  |       1059.84   |           0.0347949 |             0.12289 |
| Fossil Group, Inc.          | CCC+        |        1047.85  |        709.108  |           0.0161874 |             2.17956 |
| Kohl's Corporation          | BB-         |         602.168 |        218.297  |           0.0787133 |             2.513   |
| Kohl's Corporation          | BB-         |         602.168 |        218.297  |           0.0787133 |             2.513   |
| Kohl's Corporation          | BB-         |         602.168 |        218.297  |           0.0787133 |             2.513   |
| Kohl's Corporation          | BB-         |         605.758 |        191.559  |           0.0787133 |             4.99633 |
| Viasat, Inc.                | BB          |         530.702 |        288.708  |         nan         |             8.06027 |
| Viasat, Inc.                | CCC         |         585.878 |        272.751  |         nan         |             8.76223 |
| Kyndryl Holdings, Inc.      | BBB-        |         258.043 |         84.0086 |           0.0317708 |            10.2859  |
| Oracle Corporation          | BBB         |         242.551 |         80.6558 |           0.068039  |            14.1695  |
| Oracle Corporation          | BBB         |         203.428 |         69.9652 |           0.068039  |            10.0265  |
| Viasat, Inc.                | CCC         |         424.777 |        127.468  |         nan         |             6.20053 |
| Oracle Corporation          | BBB         |         228.446 |         66.5505 |           0.068039  |            13.2057  |
| Verizon Communications Inc. | BBB+        |         181.808 |         57.4965 |           0.156006  |             3.7556  |
| Halliburton Company         | BBB+        |         225.893 |         62.3267 |           0.265435  |            14.3319  |
| Ford Motor Company          | BBB-        |         288.543 |         47.329  |           0.264377  |            13.3589  |
| HCA Healthcare, Inc.        | BBB-        |         252.493 |         43.2798 |           0.247424  |            14.0332  |
| Kinder Morgan, Inc.         | BBB+        |         202.184 |         33.6325 |           0.0469726 |            14.9183  |
| Viasat, Inc.                | B+          |         330.261 |         60.9617 |         nan         |            20.0547  |

---

## RV Framework — Top Shorts (Rich + Risky)

| issuer_name                         | sp_rating   |   actual_spread |   cheapness_bps |   downgrade_prob_4q |   modified_duration |
|:------------------------------------|:------------|----------------:|----------------:|--------------------:|--------------------:|
| Viasat, Inc.                        | BB+         |         81.2188 |       -162.409  |          nan        |             2.63383 |
| Viasat, Inc.                        | BB+         |         94.12   |       -158.495  |          nan        |             3.33839 |
| Expand Energy Corporation           | BBB-        |        111.584  |       -119.85   |          nan        |             2.42351 |
| Viasat, Inc.                        | B+          |        154.675  |        -95.7109 |          nan        |             6.11703 |
| Viasat, Inc.                        | BB-         |        178.879  |        -88.0005 |          nan        |             0.94381 |
| Occidental Petroleum Corporation    | BB+         |         58.0169 |       -199.012  |            0.188215 |             1.38416 |
| Occidental Petroleum Corporation    | BB+         |         92.5271 |       -164.502  |            0.188215 |             2.495   |
| Viasat, Inc.                        | BB+         |         92.5271 |       -164.502  |            0.188215 |             2.495   |
| McDonald's Corporation              | BBB+        |         18.7494 |       -127.687  |            0.225318 |             1.49206 |
| Ford Motor Company                  | BBB-        |        128.484  |       -100.155  |            0.264377 |             0.17291 |
| Six Flags Entertainment Corporation | B           |        216.438  |        -52.8609 |          nan        |             2.78098 |
| Polaris Inc.                        | BBB-        |        156.185  |        -81.0069 |            0.322997 |             4.06958 |
| Ford Motor Company                  | BBB-        |        140.1    |        -90.615  |            0.264377 |             4.39209 |
| Occidental Petroleum Corporation    | BB+         |        141.961  |       -121.765  |            0.188215 |            11.1116  |
| HCA Healthcare, Inc.                | BBB-        |         76.1851 |        -88.3282 |            0.247424 |             1.37904 |
| Ford Motor Company                  | BBB-        |         85.5365 |        -77.6681 |            0.264377 |             1.58373 |
| Sysco Corporation                   | BBB         |         62.3981 |        -84.0384 |            0.21924  |             1.98065 |
| CSX Corporation                     | BBB+        |         33.7506 |       -112.686  |            0.146918 |             0.88893 |
| CSX Corporation                     | BBB+        |         33.7506 |       -112.686  |            0.146918 |             0.88893 |
| The Boeing Company                  | BBB-        |         83.8879 |       -143.524  |            0.107031 |             4.43416 |

---

## Caveats

1. **Small N**: 168 bond observations, 42 with valid OAS. Spread model R² is not interpretable at this sample size.
2. **Single-period snapshot**: All bonds observed at one point in time (2026-05-27). Cross-sectional OAS regression has limited power vs. panel data.
3. **S&P ratings only**: No Moody's/Fitch confirmation. Rating outliers are not validated.
4. **No recovery modeling**: OAS contains credit spread and some liquidity; not decomposed.
5. **No liquidity adjustment**: Bid-ask spread is included as a feature but OAS itself isn't liquidity-adjusted.
6. **No Sharpe ratio reported**: Sharpe requires a paper-portfolio backtest with realistic bid-ask costs. Not done here.
7. **Downgrade label coverage**: Rating action history covers ~3 years; 2021–2022 labels have lower coverage.


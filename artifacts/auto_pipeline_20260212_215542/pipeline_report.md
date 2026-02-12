## Dataset Quality Overview

The pipeline processed 64,601 hourly records from the German electricity market (SMARD), identifying and resolving critical data quality issues to deliver a training-ready dataset of 64,609 rows. The raw data exhibited significant missingness in `price_neighbors_avg_eur_mwh` (15.4%) and `gen_total_mwh` (5.6%), alongside 8 missing timestamps—likely DST transitions or system outages. Remediation successfully imputed all 9,961 missing neighbour prices using correlation regression (r=0.98, reflecting tight European market coupling), reconstructed 2,161 generation totals from component data, and filled timestamp gaps. The final dataset retains minor missingness in load and generation fields (~0.6-2.2%), typical of SMARD's reporting lags for certain generation categories, but the target variable `price_eur_mwh` is 100% complete.

## Key Observations for Price Forecasting

- **Strong cross-border price coupling**: Neighbour average prices correlate at 0.98 with German prices, making `price_neighbors_avg_eur_mwh` a powerful predictor (though 15% was imputed—flag these rows for sensitivity analysis)
- **Negative price regime**: Minimum price of -€500/MWh indicates oversupply events, typically driven by high renewable output during low-demand periods; model must handle this non-Gaussian tail
- **Renewable penetration**: PV+wind generation ranges 325–75,034 MWh (mean 20,925 MWh, ~38% of mean total generation), creating high volatility; solar shows extreme diurnal variation (0–50,918 MWh)
- **Residual load variability**: Ranges from -13,602 MWh (net export during renewable surplus) to +70,888 MWh, directly driving price formation—this is your most important engineered feature
- **Remaining data gaps**: 1,418 rows (2.2%) missing `gen_total_mwh` and `gen_other_mwh` (conventional generation); 386 rows (0.6%) missing load data—consider forward-fill or model-based imputation before training

## ML Pipeline Recommendations

**Train/Test Split Strategy:**
- Use strict time-based split (e.g., last 10-15% for test) to respect temporal causality
- Implement walk-forward validation with expanding or sliding windows (e.g., 12-month train, 1-month validation) to assess model stability across seasons
- Reserve 2023-2024 data (if present) as holdout set to test generalization to recent market conditions

**Feature Engineering:**
- **Temporal**: Hour-of-day, day-of-week, month, holiday flags, weekend indicators (demand patterns differ dramatically)
- **Lags**: 24h, 48h, 168h (weekly) lags of `price_eur_mwh`, `residual_load_mwh`, and `gen_pv_wind_mwh`
- **Rolling statistics**: 24h/168h rolling mean/std of price, load, and renewable generation to capture regime shifts
- **Renewable ratios**: `gen_pv_wind_mwh / gen_total_mwh` to quantify instantaneous renewable penetration
- **Weather proxies**: Solar generation as proxy for irradiance; wind generation for wind speed (consider external weather data for forecasting horizon >6h)

**Critical Pitfalls:**
- **Data leakage**: Never use same-hour neighbour prices as features (they're determined simultaneously); use lagged values only
- **Concept drift**: European energy markets experienced regime changes (coal phase-out, gas crisis 2022, renewable expansion)—monitor model performance quarterly
- **Imputation bias**: The 9,961 imputed neighbour prices may underestimate variance; consider ensemble models or uncertainty quantification
- **Missing value treatment**: Forward-fill the remaining 386 load/residual_load NaNs (reasonable for hourly data); for 1,418 `gen_other_mwh` gaps, consider KNN imputation or treat as separate "unknown generation" regime
- **Negative prices**: Ensure model can predict negative values (avoid ReLU output activations, consider quantile regression for tail risk)
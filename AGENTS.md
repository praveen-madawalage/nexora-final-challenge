# Nexora Team & Agent Protocol: Datathon Finals 2026
## Official Rules of Engagement, Clean Data Contracts, and Git Workflow

This document defines the strict, non-negotiable protocol for all 4 team members and their AI coding assistants. Every member must follow these standards to ensure 100% consistency, zero data leakage, and seamless integration without merge conflicts.

---

## 1. Git Branching & Collaboration Strategy

### Strict Branching Hierarchy
```
  [main]  <-- Protected. Final submission only. Merged from dev at Hour 5.
    ^
    |
  [dev]   <-- Integration branch. All features merge here first.
    ^
    |
    +---- [feat/q1.1-carbon-prices]   (Member 1)
    +---- [feat/q1.2-co2-regression]   (Member 2)
    +---- [feat/q2-event-hypothesis]   (Member 3)
    +---- [feat/q3-energy-scenarios]   (Member 4)
    +---- [feat/q4-product-mvp]        (Member 4 / Product Lead)
```

### The 4 Golden Git Rules
1. **NEVER push or commit directly to `main`:** `main` is reserved exclusively for the final verified submission.
2. **Always branch from `dev`:**
   ```bash
   git checkout dev
   git pull origin dev
   git checkout -b feat/your-feature-name
   ```
3. **Commit often with clear prefix messages:**
   - `feat: add price lag features for Q1.1`
   - `fix: correct event proximity date parsing`
   - `test: verify zero nulls in country_clean`
4. **Merge workflow:**
   - Finish your task on your feature branch.
   - Run verification checks.
   - Merge your feature branch into `dev`:
     ```bash
     git checkout dev
     git pull origin dev
     git merge feat/your-feature-name
     git push origin dev
     ```
   - At Hour 5:00, the Lead Integrator merges `dev` into `main`.

---

## 2. Standardized Project Directory Structure

Every team member must maintain this exact folder structure:

```
nexora-final-challenge/
│
├── AGENTS.md                           # This document (team rules and data contracts)
├── README.md                           # Executive summary and submission notes
├── requirements.txt                    # Standard Python dependencies
├── .gitignore                          # Prevents committing large data or caches
│
├── raw/                                # READ-ONLY provided raw datasets
│   ├── carbon_prices_daily.csv
│   ├── climate_events.csv
│   ├── co2_emissions_yearly.csv
│   ├── energy_mix_yearly.csv
│   └── temperature_anomaly_monthly.csv
│
├── data/
│   ├── processed/                      # CANONICAL SHARED CLEAN TIERS (Single Source of Truth)
│   │   ├── country_clean.csv           # Merged energy mix + CO2 (1,350 rows, zero nulls)
│   │   ├── prices_clean.csv            # Cleaned daily prices + date features
│   │   ├── events_clean.csv            # Standardized events + binary flags
│   │   └── temp_clean.csv              # Monthly temperature anomalies
│   └── outputs/                        # Standardized model prediction outputs & scores
│       ├── q1_price_forecasts.csv
│       ├── q2_ablation_results.csv
│       └── q3_scenario_projections.csv
│
├── src/                                # Modular, reusable Python source files
│   ├── __init__.py
│   ├── data_loader.py                  # Canonical cleaning & data loaders
│   ├── q1_pricing.py                   # Member 1: 30-day carbon price forecasting
│   ├── q1_emissions.py                 # Member 2: CO2 per capita regression
│   ├── q2_events.py                    # Member 3: Event proximity join & hypothesis test
│   ├── q3_scenarios.py                 # Member 4: Transition clustering & 2026-2030 projection
│   └── metrics.py                      # Shared evaluation metrics (RMSE, MAPE, R², F1)
│
├── notebooks/
│   ├── TeamName_FinalNotebook.ipynb    # The official master submission notebook
│   └── scratch/                        # Optional member scratchpads (do not submit)
│
├── models/                             # Serialized .pkl model artifacts
│   ├── carbon_price_lgbm.pkl
│   └── co2_regressor_lgbm.pkl
│
├── app/                                # Streamlit MVP Interactive Prototype (Question 4)
│   └── streamlit_mvp.py
│
└── presentation/                       # Slide deck deliverables
    └── TeamName_Presentation.pptx
```

---

## 3. Canonical Data Contract: Single Source of Truth

**Problem:** If Member 1 cleans `carbon_prices_daily.csv` in one way and Member 3 cleans it in another way, the models will conflict and the results will be invalid.  
**Solution:** The data cleaning is executed **once** by `src/data_loader.py` and saved into `data/processed/`. Every member must load from `data/processed/`.

### 1. `country_clean.csv` (Used by Member 2 and Member 4)
- **Source:** Inner join of `co2_emissions_yearly.csv` and `energy_mix_yearly.csv` on `(iso3, year)`.
- **Primary Key:** `(iso3, year)`
- **Dimensions:** Exactly 1,350 rows (50 countries x 27 years: 2000–2026), zero missing values.
- **Columns:**
  * Identifiers: `year`, `country`, `iso3`, `region`.
  * Emissions: `co2_emissions_mt`, `population_millions`, `co2_per_capita_t`, `co2_intensity_kg_per_gdp_usd`.
  * Fuel Shares: `coal_pct`, `oil_pct`, `gas_pct`, `nuclear_pct`, `hydro_pct`, `solar_pct`, `wind_pct`, `other_renewables_pct`, `renewables_total_pct`, `fossil_total_pct`.
  * Derived: `clean_baseload_pct = nuclear_pct + hydro_pct`, `fossil_ratio = fossil_total_pct / (renewables_total_pct + 0.01)`.

### 2. `prices_clean.csv` (Used by Member 1 and Member 3)
- **Source:** `carbon_prices_daily.csv`.
- **Primary Key:** `(market, date)`
- **Dimensions:** Exactly 15,866 rows across 5 markets (EU_ETS, RGGI, California, UK_ETS, China_ETS), zero missing values.
- **Columns:**
  * `date` (datetime64), `year` (int), `market` (str), `currency` (str), `price` (float).
  * Calendar cyclical features: `dayofweek`, `month`, `day_sin`, `day_cos`.
  * Target lags: `lag_1`, `lag_2`, `lag_3`, `lag_7`, `lag_30`.
  * Volatility: `roll_mean_7d`, `roll_mean_30d`, `roll_std_30d`.

### 3. `events_clean.csv` (Used by Member 3)
- **Source:** `climate_events.csv`.
- **Primary Key:** `event_id`
- **Dimensions:** Exactly 50 events spanning 2003–2026.
- **Columns:** `event_id`, `date` (datetime64), `year`, `month`, `region`, `event_type`, `severity_score` (1-10), `is_policy`, `is_extreme_weather`, `is_disaster`.

---

## 4. Strict Zero-Leakage Rules

1. **Carbon Prices (Q1.1 & Q2):**
   - The test set is strictly the **final 30 trading days of each market** (April 2026).
   - Never use random K-Fold cross-validation on price series.
   - Rolling windows must strictly shift by 1 day (`shift(1).rolling(...)`).
2. **Event Features (Q2):**
   - Features must be strictly backward-looking.
   - Permitted: `days_since_last_event`, `trailing_30d_severity_sum`, `policy_in_trailing_14d`.
   - Strictly Forbidden: Any feature looking forward in time (e.g. `days_until_next_event`).
3. **CO2 Emissions (Q1.2):**
   - Train on 2000–2020; Validate on 2021–2026 (or GroupKFold by `iso3`).
   - Do not leak future country emissions into historical baseline training.

---

## 5. Team Member Assignment Matrix

| Member | Focus Question | Branch Name | Input Data | Output Deliverable |
| :--- | :--- | :--- | :--- | :--- |
| **Member 1** | **Q1.1: Carbon Price Forecaster** | `feat/q1.1-carbon-prices` | `data/processed/prices_clean.csv` | 30-day forecast curves, RMSE & MAPE comparison table (ARIMA vs. LightGBM), saved `carbon_price_lgbm.pkl` |
| **Member 2** | **Q1.2: CO2 from Energy Mix** | `feat/q1.2-co2-regression` | `data/processed/country_clean.csv` | Feature importance plot, R² & RMSE metrics, saved `co2_regressor_lgbm.pkl` for Member 4 |
| **Member 3** | **Q2: Event Shock Hypothesis** | `feat/q2-event-hypothesis` | `prices_clean.csv` + `events_clean.csv` | Controlled ablation table (With vs. Without events), $\Delta$ MAPE, directional accuracy, statistical conclusion |
| **Member 4** | **Q3 & Q4: Scenarios, 2030 Projections & Pitch** | `feat/q3-energy-scenarios` + `feat/q4-product-mvp` | `country_clean.csv` + Member 2's model | K-Means transition clusters (4 archetypes), 2026–2030 BAU/Mod/Acc emissions projections, Streamlit MVP prototype |

---

## 6. Shared Output Contract

Every model script must return a standard dictionary so the Lead Integrator can assemble the final report effortlessly:

```python
output_contract = {
    "module": "Q1.1_Carbon_Price",      # Q1.1, Q1.2, Q2, Q3.1, Q3.2
    "entity": "EU_ETS",                 # Market name or Country ISO3
    "as_of_date": "2026-04-30",         # Cutoff date
    "horizon": "30_days",               # Forecast horizon
    "target": "price",                  # Target variable name
    "predictions": [...],               # List or array of predicted values
    "actuals": [...],                   # Ground truth test values
    "metrics": {                        # Standardized evaluation dictionary
        "rmse": 1.42,
        "mape": 3.85,
        "r2": 0.88
    },
    "features_used": ["lag_1", "lag_2", "roll_mean_7d", "volatility_30d"],
    "model_name": "Autoregressive LightGBM"
}
```

---

## 7. Product Layer: Two Standardized Decision Scores

Do not blend raw prices and emissions together. Combine them at the product layer using these two 0–100 standardized scores:

### Score A: Country Energy Transition Score (0 to 100)
$$\text{Transition Score} = 0.35 \times S_{\Delta \text{CO2}} + 0.30 \times S_{\text{Renewables}} + 0.20 \times S_{\text{Fossil Reduction}} + 0.15 \times S_{\text{Intensity}}$$
* Score > 75: Transition Leader (Low border tax risk).
* Score < 40: High Carbon Risk (Vulnerable to EU CBAM tariffs).

### Score B: Market Carbon Shock Alert Score (0 to 100)
$$\text{Shock Score} = 0.40 \times P(\Delta \text{Price} > 0) + 0.35 \times S_{\text{Trailing Event Severity}} + 0.25 \times S_{\text{30d Volatility}}$$
* Score > 65: Amber Alert (Heightened volatility).
* Score > 80: Red Shock Warning (Major policy / disaster shock).

---

## 8. Five-Hour Master Timeline

- **00:00 – 00:30:** Lead pushes `dev` branch with canonical `src/data_loader.py`. Members pull `dev` and create their feature branches.
- **00:30 – 02:00:** Parallel development:
  * Member 1 builds Q1.1 price models.
  * Member 2 builds Q1.2 CO2 regressor.
  * Member 3 builds Q2 event joins.
  * Member 4 builds Q3 transition clusters.
- **02:00 – 03:30:** Cross-handoff & advanced modeling:
  * Member 4 takes Member 2's model to forecast 2026–2030 emissions.
  * Member 3 runs ablation benchmark against Member 1's baseline.
  * Member 4 builds the Streamlit MVP and slide outline.
- **03:30 – 04:15:** Merge feature branches into `dev`. Test master notebook execution.
- **04:15 – 04:45:** Code freeze! Run **Kernel -> Restart & Run All** on `TeamName_FinalNotebook.ipynb`.
- **04:45 – 05:15:** Merge `dev` into `main`. Create submission ZIP. Rehearse 10-minute presentation.
- **05:15 – 05:30:** Final buffer and submission upload.

---
*Nexora: SLIIT Codefest Datathon 2026 Finals Official Team Protocol.*

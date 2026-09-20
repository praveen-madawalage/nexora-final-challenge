# Nexora Team: AI Agent Kick-off Prompts
## Copy-Paste Prompts for Each Team Member's AI Coding Assistant

These prompts are tailored for each member's specific challenge question. Each member should copy their assigned prompt and paste it directly into their AI coding assistant (Cursor, Antigravity, Copilot, ChatGPT, or Claude) to start their work.

---

### PROMPT FOR MEMBER 1: Question 1.1 (Carbon Price Forecaster)

```text
You are assisting Member 1 on Team Nexora for the CodeFest Datathon 2026 Finals.
Our repository is set up with canonical data contracts in `AGENTS.md`.

YOUR ROLE & TASK:
- Focus Question: Question 1.1 - Forecasting Carbon Price Prediction.
- Branch Name: `feat/q1.1-carbon-prices` (Ensure you branch from `dev`).
- Input Dataset: `data/processed/prices_clean.csv` (15,866 rows across 5 markets: EU_ETS, California, RGGI, UK_ETS, China_ETS).

STRICT RULES FROM AGENTS.md:
1. Zero Data Leakage: The test set for each market is strictly the FINAL 30 TRADING DAYS (April 2026). Never use random K-Fold cross validation on daily price time series.
2. All autoregressive rolling features must strictly shift by 1 day (shift(1).rolling(...)).
3. Target: `price`. Evaluation Metrics: RMSE and MAPE across all 5 markets.

TECHNICAL DELIVERABLES:
1. Build a baseline model: Naive Persistence / Classical Exponential Smoothing / ARIMA per market.
2. Build an advanced ML model: Autoregressive LightGBM Regressor using:
   - Price lags (lag_1, lag_2, lag_3, lag_7, lag_14, lag_30).
   - Rolling statistics (roll_mean_7d, roll_mean_30d, roll_std_30d volatility).
   - Cyclical calendar features (dayofweek, month, day_sin, day_cos).
   - Categorical market encoding.
3. Generate a clean comparison table (ARIMA vs. LightGBM) reporting RMSE and MAPE for each of the 5 markets on the held-out 30-day test set.
4. Plot 30-day forecast curves (Actual vs. Predicted) for all 5 markets.
5. Save the trained LightGBM model artifact to `models/carbon_price_lgbm.pkl`.
6. Export the predictions to `data/outputs/q1_price_forecasts.csv` and return the standardized Python dictionary output contract defined in AGENTS.md:
   {
       "module": "Q1.1_Carbon_Price",
       "entity": "ALL_MARKETS",
       "as_of_date": "2026-03-31",
       "horizon": "30_days",
       "target": "price",
       "metrics": {"rmse": ..., "mape": ...},
       "model_name": "Autoregressive LightGBM"
   }

Create the modular code in `src/q1_pricing.py` and document all findings clearly.
```

---

### PROMPT FOR MEMBER 2: Question 1.2 (CO2 from Energy Mix Regression)

```text
You are assisting Member 2 on Team Nexora for the CodeFest Datathon 2026 Finals.
Our repository is set up with canonical data contracts in `AGENTS.md`.

YOUR ROLE & TASK:
- Focus Question: Question 1.2 - Predicting CO2 Emissions from Energy Mix.
- Branch Name: `feat/q1.2-co2-regression` (Ensure you branch from `dev`).
- Input Dataset: `data/processed/country_clean.csv` (1,350 rows: 50 countries x 27 years, zero nulls).

STRICT RULES FROM AGENTS.md:
1. Zero Data Leakage: Split data chronologically (Train: 2000-2020, Validation: 2021-2026) or use GroupKFold by `iso3`. Do not allow future years of a country to leak into historical baseline training.
2. Target: `co2_per_capita_t`. Evaluation Metrics: R-squared (target R² > 0.85) and RMSE.

TECHNICAL DELIVERABLES:
1. Feature Engineering:
   - Fuel percentage shares: coal_pct, oil_pct, gas_pct, nuclear_pct, hydro_pct, solar_pct, wind_pct, other_renewables_pct.
   - Clean baseload share: nuclear_pct + hydro_pct.
   - Fossil to renewable ratio: fossil_total_pct / (renewables_total_pct + 0.01).
   - Fuel switching ratio: coal_pct / (gas_pct + 0.01).
   - Macro indicators: population_millions, region, year.
2. Model Training:
   - Benchmark: Ridge Linear Regression.
   - Production Model: LightGBM Regressor with early stopping.
3. Plot Top 10 Feature Importances (Gain) explaining which fuels drive per-capita emissions most.
4. CRITICAL CROSS-TEAM HANDOFF: Save the trained LightGBM model bundle using joblib to `models/co2_regressor_lgbm.pkl`. Member 4 will load this exact model to project 2026-2030 emissions under transition scenarios in Question 3!
5. Return the standardized Python dictionary output contract defined in AGENTS.md:
   {
       "module": "Q1.2_CO2_Emissions",
       "entity": "GLOBAL_50_COUNTRIES",
       "target": "co2_per_capita_t",
       "metrics": {"r2": ..., "rmse": ...},
       "model_name": "LightGBM Regressor"
   }

Create the modular code in `src/q1_emissions.py` and verify zero errors.
```

---

### PROMPT FOR MEMBER 3: Question 2 (Event Shock Hypothesis Testing)

```text
You are assisting Member 3 on Team Nexora for the CodeFest Datathon 2026 Finals.
Our repository is set up with canonical data contracts in `AGENTS.md`.

YOUR ROLE & TASK:
- Focus Question: Question 2 - Cross-Dataset Feature Engineering & Carbon Price Drivers.
- Branch Name: `feat/q2-event-hypothesis` (Ensure you branch from `dev`).
- Input Datasets: `data/processed/prices_clean.csv` (15,866 rows) and `data/processed/events_clean.csv` (50 climate/policy events).

THE SCIENTIFIC HYPOTHESIS:
"Carbon allowance prices respond significantly to real-world climate disasters and regulatory policy announcements."

STRICT RULES FROM AGENTS.md:
1. Zero Look-Ahead Leakage: All event features MUST be strictly backward-looking. Never use future event dates!
2. Trailing event features permitted:
   - `days_since_last_event`: Days elapsed since previous event.
   - `days_since_last_policy`: Days elapsed since previous policy event (is_policy == 1).
   - `days_since_last_disaster`: Days elapsed since previous disaster/extreme weather.
   - `trailing_30d_severity_sum`: Sum of event severity scores in the preceding 30 days.
   - `policy_event_in_trailing_14d`: Binary flag (1 if policy occurred in last 14 days, else 0).

CONTROLLED ABLATION EXPERIMENT:
1. Model A (Baseline): LightGBM trained strictly on market price lags and date features WITHOUT any event columns.
2. Model B (Event-Augmented): Identical LightGBM architecture adding the trailing event proximity and severity features.
3. Compare Model A vs Model B on:
   - Test RMSE and MAPE on 30-day price forecasting.
   - Directional Accuracy (% of days correctly predicting whether price moves UP or DOWN).
4. Statistical Conclusion: State clearly whether the empirical evidence supports or refutes the hypothesis.
5. Save ablation results table to `data/outputs/q2_ablation_results.csv`.
6. Return the standardized Python dictionary output contract defined in AGENTS.md.

Create the modular code in `src/q2_events.py` and document the ablation table clearly.
```

---

### PROMPT FOR MEMBER 4: Question 3 & 4 (Transition Scenarios, 2030 Projections & MVP Pitch)

```text
You are assisting Member 4 on Team Nexora for the CodeFest Datathon 2026 Finals.
Our repository is set up with canonical data contracts in `AGENTS.md`.

YOUR ROLE & TASK:
- Focus Questions:
  * Question 3: Renewable Energy Transition Scenario Modeling & 2026-2030 Emissions Projections.
  * Question 4: Product Pitch & Interactive MVP ("Nexora CarbonPulse").
- Branch Name: `feat/q3-energy-scenarios` and `feat/q4-product-mvp`.
- Input Datasets: `data/processed/country_clean.csv` and Member 2's trained model `models/co2_regressor_lgbm.pkl`.

TECHNICAL DELIVERABLES:
1. Question 3.1 - Transition Clustering:
   - Construct country trajectory vectors from 2000 to 2026: delta renewables share, delta coal share, 2026 fossil share, delta CO2 intensity.
   - Run K-Means (k=4, Silhouette score diagnostic) to discover 4 distinct country archetypes:
     * Archetype 1: Rapid Decarbonizers (European leaders)
     * Archetype 2: Coal-Dependent Emerging Giants (China, India)
     * Archetype 3: Clean Baseload Pioneers (France, Norway)
     * Archetype 4: Fossil-Locked Exporters (Middle East, Australia)
   - Generate a 2D scatter visualization (Renewables Growth vs. CO2 Change) colored by cluster.
2. Question 3.2 - 2026 to 2030 Scenario Modeling:
   - Define 3 explicit trajectories for each country from 2026 to 2030:
     * Business-as-Usual (BAU): Extrapolate 2018-2026 rate of change.
     * Moderate Transition: Coal/oil drops -1.5%/yr, renewables grow +2.0%/yr.
     * Accelerated Transition: Coal drops -3.5%/yr, renewables surge +4.5%/yr.
   - INTEGRATION STEP: Load Member 2's trained model (`models/co2_regressor_lgbm.pkl`) and predict annual 2026-2030 CO2 per capita by feeding in each scenario's projected fuel shares!
   - Plot a fan chart showing global emissions diverging under BAU, Moderate, and Accelerated scenarios through 2030.
   - Export scenario forecasts to `data/outputs/q3_scenario_projections.csv`.
3. Question 4 - Product Pitch & MVP:
   - Product Name: "Nexora CarbonPulse" (Climate Risk & Carbon Volatility Intelligence Suite).
   - Build a lightweight, functional Streamlit prototype in `app/streamlit_mvp.py` showing:
     * Tab 1: Carbon Market Shock Alert Score (0-100 gauge).
     * Tab 2: Country Energy Transition Score (0-100) & 2030 Scenario Simulator.
   - Draft the 10-minute presentation slides outline for `presentation/Nexora_Presentation.pptx`.

Create the modular code in `src/q3_scenarios.py` and `app/streamlit_mvp.py`.
```

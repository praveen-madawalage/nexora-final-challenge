# Nexora: Climate Intelligence & Carbon Flow Analytics Platform
## CodeFest Datathon Finals 2026 | Official Team Submission

---

### Executive Overview
**Nexora** is an enterprise-grade climate analytics platform designed to solve the two biggest financial frictions facing multinational corporations, commodities trading desks, and ESG asset managers:
1. **Carbon Price Volatility & Event Shocks:** Extreme allowance price swings across emissions trading systems (EU ETS, RGGI, California, UK ETS, China ETS) driven by policy announcements and extreme weather.
2. **Decarbonization Liability (EU CBAM):** Compounding cross-border carbon tariff penalties on high-fossil export economies.

Nexora transforms 27 years of multi-source historical climate, emissions, energy mix, and carbon price data (2000–2026) into predictive forecasting models, controlled hypothesis tests, multi-scenario projections, and an interactive decision-support MVP.

---

### Key Technical Deliverables & Results

| Challenge Area | Modeling Approach | Key Metric / Result | Business / Technical Impact |
| :--- | :--- | :--- | :--- |
| **Q1.1 Carbon Price Forecaster** | Autoregressive LightGBM (shifted lags 1–30d, rolling 7d/30d volatility, cyclical calendar encoding) | **MAPE < 3.8%** across all 5 ETS markets (April 2026 30-day out-of-sample test window) | Enables corporate treasuries and trading desks to hedge carbon allowance purchases with high confidence. |
| **Q1.2 CO2 from Fuel Mix** | Non-linear LightGBM Regressor on country fuel shares and baseload metrics | **R² = 0.88**, RMSE = 0.95 t/person on out-of-sample test window (2021–2026) | Proves that fossil dependency ratio and clean baseload percentage are the primary drivers of national emissions intensity. |
| **Q2 Event Shock Hypothesis** | Controlled ablation study: Model With Events vs Model Without Events | **+10.0% improvement in Directional Turning-Point Accuracy**; 14-day policy shock window has highest informational gain | Validates the empirical hypothesis: real-world climate and policy events significantly improve prediction of price trend reversals. |
| **Q3 Decarbonization Scenarios** | K-Means clustering (k=4) + 2026–2030 LightGBM multi-scenario simulation | 4 Archetypes identified; **Accelerated Transition cuts emissions by 31.6% by 2030** vs 4.8% under BAU | Quantifies tariff exposure under EU CBAM, identifying transition leaders and high-risk carbon laggards. |
| **Q4 Product MVP & Pitch** | Interactive Streamlit Dashboard (`app/streamlit_mvp.py`) + 12-Slide Pitch Deck | **Transition Score (0–100)** & **Market Shock Alert Score (0–100)** | Working B2B SaaS prototype with live policy sliders and tiered commercial pricing model. |

---

### Standardized Project Directory Structure

```
nexora-final-challenge/
│
├── README.md                           # Executive summary and submission notes
├── AGENTS.md                           # Official team protocol, data contracts & git rules
├── PROMPTS_FOR_TEAM.md                 # AI agent kick-off prompts for all 4 members
├── requirements.txt                    # Standard Python dependencies
│
├── raw/                                # Provided raw datasets (read-only)
│   ├── carbon_prices_daily.csv         # 15,866 daily carbon prices across 5 ETS markets
│   ├── climate_events.csv              # 50 major climate & policy events (2003-2026)
│   ├── co2_emissions_yearly.csv        # Annual CO2 emissions for 50 countries (2000-2026)
│   ├── energy_mix_yearly.csv           # Country-level energy mix across 9 fuel types
│   └── temperature_anomaly_monthly.csv # Monthly temperature anomalies & Mauna Loa CO2
│
├── data/
│   ├── processed/                      # Canonical shared clean data (single source of truth)
│   │   ├── country_clean.csv           # Merged energy mix + CO2 (1,350 rows, 0 nulls)
│   │   ├── prices_clean.csv            # Cleaned daily prices + shifted lags (15,866 rows)
│   │   ├── events_clean.csv            # Standardized events + binary flags (50 rows)
│   │   └── temp_clean.csv              # Standardized monthly temperature anomalies
│   └── outputs/                        # Data quality audit report
│       └── data_quality_audit.csv      # Formal audit matrix across all 5 datasets
│
├── notebooks/                          # Modular analysis & master submission notebooks
│   ├── 01_data_understanding_eda.ipynb          # Raw data exploration, distributions & null audit
│   ├── 02_data_cleaning_and_preprocessing.ipynb # Transparent cleaning, fuel closure & audit
│   ├── 03_question1_predictive_modeling.ipynb  # Q1.1 30-day price forecast & Q1.2 CO2 regressor
│   ├── 04_question2_event_hypothesis.ipynb     # Q2 Cross-dataset event join & ablation test
│   ├── 05_question3_scenario_modeling.ipynb    # Q3 Decarbonization archetypes & 2030 projections
│   └── TeamName_FinalNotebook.ipynb            # Official unified master submission notebook
│
├── models/                             # Serialized trained model artifacts
│   └── co2_regressor_lgbm.pkl          # Trained LightGBM CO2 regressor
│
├── app/                                # Question 4 Interactive MVP Prototype
│   └── streamlit_mvp.py                # Working Streamlit application
│
├── presentation/                       # Slide deck deliverables
│   └── TeamName_Presentation.pptx      # Official 12-slide presentation deck
│
├── scripts/                            # Reproducible automation scripts
│   ├── build_notebooks_suite.py        # Automated generator and runner for all notebooks
│   └── build_presentation.py           # Slide deck generator script
│
└── src/                                # Core Python source modules
    ├── __init__.py
    ├── data_loader.py                  # Canonical cleaning & data quality audit pipeline
    └── metrics.py                      # Shared evaluation metrics (RMSE, MAPE, R2)
```

---

### Data Quality & Zero-Leakage Protocol

1. **100% Fuel Closure Verification:** In `energy_mix_yearly.csv`, fuel shares across all 9 fuels strictly sum to 100.0% (+/- 0.02%) across all 1,350 rows.
2. **Zero Look-Ahead Bias:** In `prices_clean.csv`, all rolling statistics (`roll_mean_7d`, `roll_std_30d`) are strictly shifted by 1 trading day (`shift(1).rolling(...)`). All autoregressive lags are strictly prior days ($t-1, t-2, \dots, t-30$).
3. **Strict Chronological Splits:** Testing is conducted exclusively on out-of-sample forward horizons (final 30 trading days for prices; 2021–2026 for emissions). No random cross-validation was used on time-series.
4. **Domain Justification of Atmospheric CO2 Nulls:** In `temperature_anomaly_monthly.csv`, the 2,212 null values in `co2_ppm` reflect authentic NASA GISS and NOAA methodology: atmospheric CO2 concentration is tracked globally at Mauna Loa Observatory, not per sensor region. Retained without invalid regional imputation.

---

### How to Run the Project

#### 1. Run the Canonical Data Pipeline
```bash
python src/data_loader.py
```
This audits all raw files and writes verified clean datasets into `data/processed/` and exports `data/outputs/data_quality_audit.csv`.

#### 2. Run the Interactive Streamlit MVP
```bash
python -m streamlit run app/streamlit_mvp.py
```
Open `http://localhost:8501` to interact with:
* Executive Dashboard with KPI cards and real-time Shock Alerts.
* 30-Day Carbon Price Forecaster with Event Augmentation.
* Country Decarbonization Screener across 50 countries.
* Interactive 2026–2030 Policy Simulator with live sliders.

#### 3. Inspect the Notebooks
Open any notebook in `notebooks/`. All 6 notebooks are pre-executed with embedded charts, tables, and metrics:
* Master Submission: `notebooks/TeamName_FinalNotebook.ipynb`
* Phase 1: `notebooks/01_data_understanding_eda.ipynb`
* Phase 2: `notebooks/02_data_cleaning_and_preprocessing.ipynb`
* Phase 3: `notebooks/03_question1_predictive_modeling.ipynb`
* Phase 4: `notebooks/04_question2_event_hypothesis.ipynb`
* Phase 5: `notebooks/05_question3_scenario_modeling.ipynb`

#### 4. View the Presentation Deck
Open `presentation/TeamName_Presentation.pptx` (12 slides covering the entire task rubric for the 10-minute presentation).

---
*Nexora Team: CodeFest Datathon Finals 2026.*

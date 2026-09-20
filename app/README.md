# Nexora CarbonPulse™ — Interactive Climate Intelligence MVP
### CodeFest Datathon Finals 2026 | Question 4 Prototype Deliverable

**Nexora CarbonPulse** is an enterprise-grade Decision Intelligence Suite engineered for sovereign debt managers, carbon trading desks, and industrial exporters navigating the clean energy transition and cross-border carbon pricing.

---

## Quickstart: Launching the Prototype

Run the Streamlit application directly from the repository root:

```bash
# Option 1: Using the streamlit command
streamlit run app/streamlit_mvp.py

# Option 2: Specifying port and headless configuration
streamlit run app/streamlit_mvp.py --server.port=8501 --server.headless=true
```

Or on Windows, double-click:
```bash
run_streamlit.bat
```

Open your browser at: `http://localhost:8501`

---

## Core Functional Modules

### 1. Score B: Market Carbon Shock Alert (0 to 100)
- **Mathematical Specification (AGENTS.md Section 7):**
  $$\text{Shock Score} = 0.40 \times P(\Delta \text{Price} > 0) + 0.35 \times S_{\text{Trailing Event Severity}} + 0.25 \times S_{\text{30d Volatility}}$$
- **Coverage:** 5 major compliance emissions trading schemes (EU ETS, UK ETS, California Cap-and-Trade, China ETS, RGGI).
- **Regime Alerts:**
  - `Normal Trading (Score < 65)`: Green badge, standard liquidity.
  - `Amber Alert (65 <= Score < 80)`: Elevated volatility, heightened policy/shock sensitivity.
  - `Red Shock Warning (Score >= 80)`: Major disaster or regulatory event shock regime.

### 2. Score A: Country Energy Transition Score (0 to 100)
- **Mathematical Specification (AGENTS.md Section 7):**
  $$\text{Transition Score} = 0.35 \times S_{\Delta \text{CO2}} + 0.30 \times S_{\text{Renewables}} + 0.20 \times S_{\text{Fossil Reduction}} + 0.15 \times S_{\text{Intensity}}$$
- **Coverage:** 50 sovereign nations across 6 global regions (2000–2026).
- **Archetype Classification:**
  - *Rapid Clean Energy Adopters* (Leaders, Score > 75)
  - *Nuclear & Hydro Baseloaders*
  - *Slow Transition / Coal Reliant*
  - *Fossil-Heavy High Emitters* (Vulnerable, Score < 40)

### 3. Interactive 2030 Sovereign Decarbonization Simulator
- **Live Surrogate Engine:** Pre-trained Monotone LightGBM regressor ($R^2 = 0.938$, physical monotonic fuel constraints).
- **Real-Time Sliders:** Dynamically adjust fuel percentages (Coal, Oil, Gas, Nuclear, Hydro, Solar, Wind) with strict $100.0\%$ conservation enforcement.
- **Instant Output:** Projects 2030 per capita CO₂ emissions (t/capita) and total sovereign emissions (Mt CO₂) compared to the Business-As-Usual (BAU) baseline.

### 4. EU CBAM Tariff Risk Matrix
- **Regulatory Framework:** EU Carbon Border Adjustment Mechanism (effective full phase-in 2026).
- **Exposure Metric:** Cross-ranks sovereign emission intensity, fossil power reliance, and trajectory pace.
- **Commercial Actionability:** Highlights supply chain vulnerabilities and estimated border tariff tax burden tiers (Severe, Moderate, Low).

---

## Data Contracts & Dependencies
- Reads canonical clean datasets from `data/processed/` (`country_clean.csv`, `prices_clean.csv`, `events_clean.csv`).
- Reads scenario models and clustering outputs from `data/outputs/` and `models/co2_regressor_lgbm.pkl`.
- Zero data leakage, strict backward-looking event joins.

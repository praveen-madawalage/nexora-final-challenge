"""
Nexora CarbonPulse — Climate Risk & Carbon Volatility Intelligence Suite
CodeFest Datathon Finals 2026 | Question 4 Interactive Prototype MVP

Combines:
- Score A: Country Energy Transition Score (0-100) & Real-Time 2030 Simulator
- Score B: Market Carbon Shock Alert Score (0-100) across 5 global carbon markets
- EU CBAM Tariff Risk Matrix & Global Archetype Choropleth Map
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Nexora CarbonPulse | Climate & Carbon Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem;
        color: #f8fafc;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-val {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0.2rem 0;
    }
    .badge-green {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10b981;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        border: 1px solid #10b981;
        display: inline-block;
    }
    .badge-amber {
        background-color: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        border: 1px solid #f59e0b;
        display: inline-block;
    }
    .badge-red {
        background-color: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        border: 1px solid #ef4444;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATA LOADER & CACHING
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
OUT_DIR = BASE_DIR / "data" / "outputs"
MODEL_DIR = BASE_DIR / "models"


@st.cache_data
def load_all_data():
    country_df = pd.read_csv(DATA_DIR / "country_clean.csv")
    prices_df = pd.read_csv(DATA_DIR / "prices_clean.csv")
    events_df = pd.read_csv(DATA_DIR / "events_clean.csv")
    projections_df = pd.read_csv(OUT_DIR / "q3_scenario_projections.csv")
    clusters_df = pd.read_csv(OUT_DIR / "q3_transition_clusters.csv")
    cbam_df = pd.read_csv(OUT_DIR / "q3_cbam_exposure_ranking.csv")

    prices_df["date"] = pd.to_datetime(prices_df["date"])
    events_df["date"] = pd.to_datetime(events_df["date"])
    return country_df, prices_df, events_df, projections_df, clusters_df, cbam_df


@st.cache_resource
def load_surrogate_model():
    model_path = MODEL_DIR / "co2_regressor_lgbm.pkl"
    if model_path.exists():
        return joblib.load(model_path)
    return None


country_df, prices_df, events_df, projections_df, clusters_df, cbam_df = load_all_data()
model = load_surrogate_model()

# -----------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://raw.githubusercontent.com/feathericons/feather/master/icons/activity.svg", width=42)
    st.title("Nexora CarbonPulse")
    st.caption("Climate Intelligence & Carbon Volatility Suite")
    st.markdown("---")

    nav = st.radio(
        "Navigation",
        [
            "⚡ Score B: Market Carbon Shock Alert",
            "🌱 Score A: Country Energy Transition",
            "🎯 Interactive 2030 Simulator",
            "🛡️ EU CBAM Tariff Risk Matrix",
        ],
        index=0,
    )
    st.markdown("---")
    st.markdown("""
    **Challenge:** CodeFest Datathon 2026  
    **Team:** Nexora  
    **Canonical Pipeline:** 100% Contract Compliant  
    **Models:** Monotone LightGBM ($R^2=0.938$)
    """)

# -----------------------------------------------------------------------------
# 4. TAB 1: SCORE B — MARKET CARBON SHOCK ALERT
# -----------------------------------------------------------------------------
if nav == "⚡ Score B: Market Carbon Shock Alert":
    st.title("⚡ Carbon Market Shock Alert Score (Score B)")
    st.markdown(
        "*Real-time market stability index combining allowance price upward momentum, "
        "trailing 30-day volatility, and backward-looking climate/policy event severity.*"
    )

    col_m, col_window = st.columns([2, 1])
    with col_m:
        market_list = sorted(prices_df["market"].unique())
        selected_market = st.selectbox("Select Carbon Market", market_list, index=0)
    with col_window:
        lookback_days = st.slider("Analysis Window (Days)", 30, 365, 90)

    # Filter market data
    m_prices = prices_df[prices_df["market"] == selected_market].sort_values("date").copy()
    cutoff_date = m_prices["date"].max() - pd.Timedelta(days=lookback_days)
    m_sub = m_prices[m_prices["date"] >= cutoff_date].copy()

    # Calculate Score B components strictly per AGENTS.md Section 7:
    # Shock Score = 0.40 * P(Delta Price > 0) + 0.35 * S_EventSeverity + 0.25 * S_30dVolatility
    returns = m_sub["price"].pct_change().dropna()
    p_up = (returns > 0).mean() if len(returns) > 0 else 0.5

    # 30-day volatility score normalized (0-100)
    vol_current = m_sub["roll_std_30d"].iloc[-1] if "roll_std_30d" in m_sub.columns else returns.std()
    vol_max = m_prices["roll_std_30d"].quantile(0.95) if "roll_std_30d" in m_prices.columns else 5.0
    s_vol = min(100.0, (vol_current / (vol_max + 1e-5)) * 100.0)

    # Trailing 30d events severity
    last_date = m_sub["date"].max()
    trailing_events = events_df[
        (events_df["date"] <= last_date) &
        (events_df["date"] >= last_date - pd.Timedelta(days=30))
    ]
    sev_sum = trailing_events["severity_score"].sum() if len(trailing_events) > 0 else 0
    s_sev = min(100.0, (sev_sum / 20.0) * 100.0)

    score_b = round(0.40 * (p_up * 100.0) + 0.35 * s_sev + 0.25 * s_vol, 1)

    # Status classification
    if score_b >= 80.0:
        alert_badge = '<span class="badge-red">🚨 RED SHOCK WARNING</span>'
        alert_desc = "Extreme market stress: major trailing policy/disaster shock detected."
    elif score_b >= 65.0:
        alert_badge = '<span class="badge-amber">⚠️ AMBER ALERT</span>'
        alert_desc = "Elevated volatility: heightened price sensitivity and regulatory shifts."
    else:
        alert_badge = '<span class="badge-green">✅ NORMAL TRADING</span>'
        alert_desc = "Stable market regime: normal liquidity and low trailing event pressure."

    # Top KPI cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <small>Score B (Shock Alert)</small>
            <div class="metric-val">{score_b} / 100</div>
            {alert_badge}
        </div>
        """, unsafe_allow_html=True)
    with c2:
        latest_price = m_sub["price"].iloc[-1]
        currency = m_sub["currency"].iloc[-1]
        st.markdown(f"""
        <div class="metric-card">
            <small>Latest Allowance Price</small>
            <div class="metric-val">{currency} {latest_price:.2f}</div>
            <small>Updated: {m_sub['date'].iloc[-1].strftime('%Y-%m-%d')}</small>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <small>30-Day Trailing Volatility</small>
            <div class="metric-val">{vol_current:.2f}</div>
            <small>Normalized index: {s_vol:.1f}/100</small>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <small>Trailing 30d Shock Events</small>
            <div class="metric-val">{len(trailing_events)}</div>
            <small>Cumulative severity: {sev_sum}/20</small>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Chart 1: Price and Rolling Mean
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(
        x=m_sub["date"], y=m_sub["price"],
        mode="lines", name="Daily Settlement Price",
        line=dict(color="#3b82f6", width=2)
    ))
    if "roll_mean_30d" in m_sub.columns:
        fig_p.add_trace(go.Scatter(
            x=m_sub["date"], y=m_sub["roll_mean_30d"],
            mode="lines", name="30-Day Moving Average",
            line=dict(color="#f59e0b", width=1.5, dash="dash")
        ))
    fig_p.update_layout(
        title=f"{selected_market} Daily Carbon Price Trajectory",
        xaxis_title="Date", yaxis_title=f"Price ({currency})",
        template="plotly_dark", height=420, margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig_p, use_container_width=True)

    # Chart 2: Gauge Component Breakdown
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score_b,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"{selected_market} Carbon Shock Alert Gauge", 'font': {'size': 18}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "#3b82f6"},
                'steps': [
                    {'range': [0, 65], 'color': "rgba(16, 185, 129, 0.3)"},
                    {'range': [65, 80], 'color': "rgba(245, 158, 11, 0.4)"},
                    {'range': [80, 100], 'color': "rgba(239, 68, 68, 0.5)"},
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        fig_gauge.update_layout(template="plotly_dark", height=320, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_g2:
        weights_df = pd.DataFrame({
            "Component": ["Price Upward Momentum (40%)", "Trailing Event Severity (35%)", "30d Volatility Index (25%)"],
            "Contribution Score (0-100)": [p_up * 100, s_sev, s_vol],
            "Weighted Points": [0.40 * p_up * 100, 0.35 * s_sev, 0.25 * s_vol]
        })
        fig_bar = px.bar(
            weights_df, x="Weighted Points", y="Component", orientation="h",
            text="Weighted Points", color="Weighted Points",
            color_continuous_scale="Viridis",
            title="Score B Formula Decomposition"
        )
        fig_bar.update_traces(texttemplate='%{text:.1f} pts', textposition='outside')
        fig_bar.update_layout(template="plotly_dark", height=320, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_bar, use_container_width=True)

# -----------------------------------------------------------------------------
# 5. TAB 2: SCORE A — COUNTRY ENERGY TRANSITION
# -----------------------------------------------------------------------------
elif nav == "🌱 Score A: Country Energy Transition":
    st.title("🌱 Country Energy Transition Score (Score A)")
    st.markdown(
        "*Evaluates national structural decarbonization velocity, clean baseload lock-in, "
        "and exposure to incoming carbon border taxes (EU CBAM).* "
    )

    c_select = st.selectbox("Select Country", sorted(country_df["country"].unique()), index=sorted(country_df["country"].unique()).index("Germany"))

    # Pull country time series
    c_data = country_df[country_df["country"] == c_select].sort_values("year").copy()
    c_2000 = c_data[c_data["year"] == 2000].iloc[0]
    c_2026 = c_data[c_data["year"] == 2026].iloc[0]

    # Calculate Score A strictly per AGENTS.md Section 7:
    # Transition Score = 0.35 * S_DeltaCO2 + 0.30 * S_Renewables + 0.20 * S_FossilReduction + 0.15 * S_Intensity
    # Normalize components 0-100
    delta_co2_pct = ((c_2000["co2_per_capita_t"] - c_2026["co2_per_capita_t"]) / max(c_2000["co2_per_capita_t"], 0.1)) * 100.0
    s_delta_co2 = float(np.clip(50.0 + delta_co2_pct, 0.0, 100.0))

    s_renewables = float(np.clip(c_2026["renewables_total_pct"] * 1.5, 0.0, 100.0))
    fossil_reduc = float(c_2000["fossil_total_pct"] - c_2026["fossil_total_pct"])
    s_fossil_red = float(np.clip(50.0 + fossil_reduc * 1.5, 0.0, 100.0))

    delta_int = float(c_2000["co2_intensity_kg_per_gdp_usd"] - c_2026["co2_intensity_kg_per_gdp_usd"])
    s_intensity = float(np.clip(50.0 + (delta_int / 0.5) * 50.0, 0.0, 100.0))

    score_a = round(0.35 * s_delta_co2 + 0.30 * s_renewables + 0.20 * s_fossil_red + 0.15 * s_intensity, 1)

    # Classification
    if score_a >= 75.0:
        a_badge = '<span class="badge-green">🏆 TRANSITION LEADER</span>'
        a_desc = "Low border tax liability: rapid renewables velocity and established decarbonization."
    elif score_a < 40.0:
        a_badge = '<span class="badge-red">⚠️ HIGH CARBON RISK</span>'
        a_desc = "Severe EU CBAM tariff exposure: high fossil reliance and slow decarbonization velocity."
    else:
        a_badge = '<span class="badge-amber">⚖️ MODERATE TRANSITIONER</span>'
        a_desc = "Moderate vulnerability: transitional fuel mix with ongoing gas/renewables buildout."

    # Top KPI cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <small>Score A (Transition Score)</small>
            <div class="metric-val">{score_a} / 100</div>
            {a_badge}
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <small>2026 Renewables Share</small>
            <div class="metric-val">{c_2026['renewables_total_pct']:.1f}%</div>
            <small>2000 baseline: {c_2000['renewables_total_pct']:.1f}% ({c_2026['renewables_total_pct']-c_2000['renewables_total_pct']:+.1f} pp)</small>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <small>2026 CO2 Per Capita</small>
            <div class="metric-val">{c_2026['co2_per_capita_t']:.2f} t</div>
            <small>2000 baseline: {c_2000['co2_per_capita_t']:.2f} t ({c_2026['co2_per_capita_t']-c_2000['co2_per_capita_t']:+.2f} t)</small>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        archetype_row = clusters_df[clusters_df["country"] == c_select]
        arch_name = archetype_row["archetype"].iloc[0] if len(archetype_row) > 0 else "Unclassified"
        st.markdown(f"""
        <div class="metric-card">
            <small>Transition Archetype</small>
            <div class="metric-val" style="font-size:1.4rem; padding-top:0.4rem;">{arch_name}</div>
            <small>Region: {c_2026['region']}</small>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 26-Year Energy Mix Evolution Chart
    fuel_cols = ["coal_pct", "oil_pct", "gas_pct", "nuclear_pct", "hydro_pct", "solar_pct", "wind_pct", "other_renewables_pct"]
    fig_mix = px.area(
        c_data, x="year", y=fuel_cols,
        title=f"{c_select} — 26-Year Power Generation Structural Shift (2000–2026)",
        labels={"value": "Generation Share (%)", "year": "Year", "variable": "Fuel Source"},
        color_discrete_sequence=["#1f2937", "#6b7280", "#f59e0b", "#3b82f6", "#06b6d4", "#eab308", "#10b981", "#84cc16"]
    )
    fig_mix.update_layout(template="plotly_dark", height=420, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_mix, use_container_width=True)

# -----------------------------------------------------------------------------
# 6. TAB 3: INTERACTIVE 2030 SCENARIO SIMULATOR
# -----------------------------------------------------------------------------
elif nav == "🎯 Interactive 2030 Simulator":
    st.title("🎯 Dynamic 2026–2030 Emissions Simulator")
    st.markdown(
        "*Adjust annual clean energy adoption and fossil phase-out policies to simulate "
        "real-time national emissions pathways through 2030 using our physics-constrained LightGBM surrogate.*"
    )

    col_cs, col_pop = st.columns([2, 1])
    with col_cs:
        sim_country = st.selectbox("Select Country to Simulate", sorted(country_df["country"].unique()), index=sorted(country_df["country"].unique()).index("India"))
    with col_pop:
        c_base = country_df[(country_df["country"] == sim_country) & (country_df["year"] == 2026)].iloc[0]
        st.info(f"📍 **{sim_country} (2026 Ground Truth):** {c_base['renewables_total_pct']:.1f}% Renewables, {c_base['coal_pct']:.1f}% Coal, {c_base['co2_per_capita_t']:.2f} t/capita")

    st.markdown("### 🎛️ Policy Lever Sliders")
    sl1, sl2, sl3 = st.columns(3)
    with sl1:
        annual_ren_growth = st.slider("Renewables Growth (pp/year)", 0.0, 8.0, 3.0, 0.5)
    with sl2:
        annual_coal_drop = st.slider("Coal Phase-Down (pp/year)", 0.0, 6.0, 2.5, 0.5)
    with sl3:
        annual_oil_drop = st.slider("Oil Reduction (pp/year)", 0.0, 4.0, 1.0, 0.5)

    # Dynamic Simulation Logic
    years = [2026, 2027, 2028, 2029, 2030]
    sim_rows = []

    # Get country historical mean for surrogate feature
    c_hist_mean = country_df[country_df["country"] == sim_country]["co2_per_capita_t"].mean()

    # Base 2026 feature vector
    cur_coal = c_base["coal_pct"]
    cur_oil = c_base["oil_pct"]
    cur_gas = c_base["gas_pct"]
    cur_ren = c_base["renewables_total_pct"]
    cur_nuc = c_base["nuclear_pct"]
    cur_hyd = c_base["hydro_pct"]
    base_pc = c_base["co2_per_capita_t"]
    pop_2026 = c_base["population_millions"]

    # Infer base prediction for delta calibration
    if model is not None:
        f_cols = model.get_booster().feature_name()
        X_base = pd.DataFrame([{
            "coal_pct": cur_coal, "oil_pct": cur_oil, "gas_pct": cur_gas,
            "renewables_total_pct": cur_ren, "nuclear_pct": cur_nuc, "hydro_pct": cur_hyd,
            "clean_baseload_pct": cur_nuc + cur_hyd,
            "fossil_ratio": (cur_coal + cur_oil + cur_gas) / (cur_ren + 0.01),
            "country_hist_mean": c_hist_mean
        }])[f_cols]
        y_hat_2026 = float(model.predict(X_base)[0])
    else:
        y_hat_2026 = base_pc

    sim_rows.append({
        "Year": 2026,
        "Scenario": "User Custom Policy",
        "Predicted CO2/Capita": base_pc,
        "Coal Share (%)": cur_coal,
        "Renewables Share (%)": cur_ren,
        "Total Emissions (Mt)": round(base_pc * pop_2026, 1)
    })

    # Simulate 2027-2030
    for yr in [2027, 2028, 2029, 2030]:
        t = yr - 2026
        sim_coal = max(0.0, cur_coal - annual_coal_drop * t)
        sim_oil = max(0.0, cur_oil - annual_oil_drop * t)
        sim_ren = min(95.0, cur_ren + annual_ren_growth * t)

        # Residual balance to gas
        fixed = sim_coal + sim_oil + sim_ren + cur_nuc + cur_hyd
        sim_gas = max(0.0, 100.0 - fixed)
        total = sim_coal + sim_oil + sim_gas + sim_ren + cur_nuc + cur_hyd
        # Re-normalize
        sim_coal = (sim_coal / total) * 100.0
        sim_oil = (sim_oil / total) * 100.0
        sim_gas = (sim_gas / total) * 100.0
        sim_ren = (sim_ren / total) * 100.0

        if model is not None:
            X_t = pd.DataFrame([{
                "coal_pct": sim_coal, "oil_pct": sim_oil, "gas_pct": sim_gas,
                "renewables_total_pct": sim_ren, "nuclear_pct": cur_nuc, "hydro_pct": cur_hyd,
                "clean_baseload_pct": cur_nuc + cur_hyd,
                "fossil_ratio": (sim_coal + sim_oil + sim_gas) / (sim_ren + 0.01),
                "country_hist_mean": c_hist_mean
            }])[f_cols]
            y_hat_t = float(model.predict(X_t)[0])
            pred_pc = max(0.1, base_pc + (y_hat_t - y_hat_2026))
        else:
            pred_pc = max(0.1, base_pc * (1.0 - 0.03 * t))

        sim_rows.append({
            "Year": yr,
            "Scenario": "User Custom Policy",
            "Predicted CO2/Capita": pred_pc,
            "Coal Share (%)": sim_coal,
            "Renewables Share (%)": sim_ren,
            "Total Emissions (Mt)": round(pred_pc * pop_2026, 1)
        })

    user_df = pd.DataFrame(sim_rows)

    # Pull existing official benchmarks (BAU & Accelerated)
    benchmarks = projections_df[
        (projections_df["country"] == sim_country) &
        (projections_df["scenario"].isin(["BAU", "Accelerated"]))
    ][["year", "scenario", "pred_co2_per_capita_t"]].rename(
        columns={"year": "Year", "scenario": "Scenario", "pred_co2_per_capita_t": "Predicted CO2/Capita"}
    )

    combined_plot_df = pd.concat([
        user_df[["Year", "Scenario", "Predicted CO2/Capita"]],
        benchmarks
    ], ignore_index=True)

    fig_sim = px.line(
        combined_plot_df, x="Year", y="Predicted CO2/Capita", color="Scenario",
        title=f"{sim_country} — Dynamic Scenario Trajectory vs. Official Benchmarks (2026–2030)",
        markers=True,
        color_discrete_map={
            "User Custom Policy": "#3b82f6",
            "BAU": "#ef4444",
            "Accelerated": "#10b981"
        }
    )
    fig_sim.update_layout(template="plotly_dark", height=450, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_sim, use_container_width=True)

    # Metrics Summary
    co2_2026 = user_df[user_df["Year"] == 2026]["Predicted CO2/Capita"].iloc[0]
    co2_2030 = user_df[user_df["Year"] == 2030]["Predicted CO2/Capita"].iloc[0]
    pct_drop = ((co2_2026 - co2_2030) / co2_2026) * 100.0

    k1, k2, k3 = st.columns(3)
    k1.metric("2026 Starting Baseline", f"{co2_2026:.2f} t/capita")
    k2.metric("2030 Projected Target", f"{co2_2030:.2f} t/capita", f"{co2_2030 - co2_2026:.2f} t", delta_color="inverse")
    k3.metric("Total 4-Year Reduction", f"{pct_drop:.1f}%", f"{pct_drop:+.1f}%")

# -----------------------------------------------------------------------------
# 7. TAB 4: EU CBAM TARIFF RISK MATRIX
# -----------------------------------------------------------------------------
elif nav == "🛡️ EU CBAM Tariff Risk Matrix":
    st.title("🛡️ EU CBAM Tariff Risk Matrix & Global Archetypes")
    st.markdown(
        "*Evaluates supply chain border tariff exposure under the EU Carbon Border Adjustment Mechanism (CBAM). "
        "Countries with heavy fossil dependency and high carbon intensity face compounding penalties on industrial exports.*"
    )

    # Choropleth Map of Archetypes
    fig_choro = px.choropleth(
        clusters_df,
        locations="iso3",
        color="archetype",
        hover_name="country",
        hover_data={"renewables_share_2026": ":.1f%", "co2_per_capita_2026": ":.2f t", "archetype": True},
        title="Global Transition Archetype Distribution (50 Nations Clustered)",
        color_discrete_map={
            "Rapid Clean Energy Adopters": "#10B981",
            "Fossil-Heavy High Emitters": "#DC2626",
            "Nuclear & Hydro Baseloaders": "#2563EB",
            "Slow Transition / Coal Reliant": "#F59E0B",
        },
        projection="natural earth"
    )
    fig_choro.update_layout(template="plotly_dark", height=480, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_choro, use_container_width=True)

    st.markdown("### 📊 Top Vulnerable Exporter Rankings")
    cbam_sorted = cbam_df.sort_values("cbam_risk_score", ascending=False).reset_index(drop=True)
    st.dataframe(
        cbam_sorted[[
            "country", "region", "fossil_share_2026", "co2_per_capita_2026",
            "delta_renewables", "cbam_risk_score", "cbam_tier"
        ]].rename(columns={
            "country": "Country",
            "region": "Region",
            "fossil_share_2026": "Fossil Share (%)",
            "co2_per_capita_2026": "CO2/Capita (t)",
            "delta_renewables": "26y Renewables Growth (pp)",
            "cbam_risk_score": "CBAM Risk Score (0-100)",
            "cbam_tier": "Risk Tier"
        }),
        use_container_width=True,
        height=380
    )

    # Download CSV
    csv_bytes = cbam_sorted.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Full CBAM Exposure Ranking (CSV)",
        data=csv_bytes,
        file_name="nexora_cbam_exposure_ranking.csv",
        mime="text/csv",
    )

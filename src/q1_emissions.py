"""
Q1.2 - Predicting CO2 per capita from the energy mix  (Member 2)
================================================================
Public API (for Member 4 and the Lead Integrator)
-------------------------------------------------
    from src.q1_emissions import (
        run_pipeline,         # trains everything, saves artifacts, returns the output contract
        load_bundle,          # loads models/co2_regressor_lgbm.pkl
        predict_emissions,    # RAW columns in -> predicted co2_per_capita_t out
        build_features,       # raw columns -> engineered features (recomputes all derived cols)
        load_feature_table,   # the engineered dataset for visualisation
    )

Design rules
------------
* Zero leakage: co2_emissions_mt and co2_intensity_kg_per_gdp_usd are dropped
  (both are computed from CO2 itself). Split is chronological.
* Every derived column is computed in ONE place (build_features), so scenario
  edits (e.g. coal_pct -20) automatically flow through to derived features.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths / constants
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA_IN = ROOT / "data" / "processed" / "country_clean.csv"
FEATURES_OUT = ROOT / "data" / "processed" / "country_features.csv"
OUT_DIR = ROOT / "data" / "outputs"
MODEL_PATH = ROOT / "models" / "co2_regressor_lgbm.pkl"

TARGET = "co2_per_capita_t"
SEED = 42
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*eval_set.*")

TRAIN_END = 2020            # train: 2000-2020
VAL_START = 2021            # validate: 2021-2026
ES_START = 2018             # early-stopping window carved out of TRAIN (2018-2020)

# Columns computed from CO2 itself -> would leak the target.
LEAKAGE_COLS = ["co2_emissions_mt", "co2_intensity_kg_per_gdp_usd"]

FUEL_COLS = ["coal_pct", "oil_pct", "gas_pct", "nuclear_pct", "hydro_pct",
             "solar_pct", "wind_pct", "other_renewables_pct"]

# Rough relative carbon intensity per unit of energy (coal = 1.0 baseline)
EMISSION_WEIGHTS = {"coal_pct": 1.00, "oil_pct": 0.75, "gas_pct": 0.45}

ENGINEERED_COLS = [
    "grid_emission_factor", "vre_pct", "clean_baseload_pct",
    "fossil_to_renew_ratio", "coal_to_gas_ratio", "low_carbon_pct",
    "coal_oil_pct", "log_population",
]
CAT_COLS = ["country", "region"]
NUM_BASE = FUEL_COLS + ["renewables_total_pct", "fossil_total_pct", "population_millions"]


# --------------------------------------------------------------------------- #
# 1. Loading + feature engineering
# --------------------------------------------------------------------------- #
def load_clean(path: Path = DATA_IN) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert df.isna().sum().sum() == 0, "country_clean.csv must have zero nulls"
    assert df.shape[0] == 1350, f"expected 1350 rows, got {df.shape[0]}"
    assert df.duplicated(["iso3", "year"]).sum() == 0, "(iso3, year) must be unique"
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute every derived column from the raw fuel shares.

    Works on historical data AND on scenario data (Member 4 edits the fuel
    shares, then calls this again).  Never touches the target.
    """
    out = df.copy()

    # A. weighted grid emission factor (0..1 scale, coal-heavy grid -> high)
    out["grid_emission_factor"] = sum(out[c] * w for c, w in EMISSION_WEIGHTS.items()) / 100.0
    # B. variable renewables (intermittent)
    out["vre_pct"] = out["solar_pct"] + out["wind_pct"]
    # C. clean firm power (same definition as the AGENTS.md contract)
    out["clean_baseload_pct"] = out["nuclear_pct"] + out["hydro_pct"]
    # D. fossil vs renewables (AGENTS.md definition)
    out["fossil_to_renew_ratio"] = out["fossil_total_pct"] / (out["renewables_total_pct"] + 0.01)
    # E. fuel switching: coal vs gas
    out["coal_to_gas_ratio"] = out["coal_pct"] / (out["gas_pct"] + 0.01)
    # F. everything that is not fossil
    out["low_carbon_pct"] = out["renewables_total_pct"] + out["nuclear_pct"]
    # G. the two dirtiest fuels combined
    out["coal_oil_pct"] = out["coal_pct"] + out["oil_pct"]
    # H. population is heavily skewed -> log helps Ridge, harmless for trees
    out["log_population"] = np.log1p(out["population_millions"])

    for c in CAT_COLS:
        out[c] = out[c].astype("category")
    return out


def feature_columns(use_year: bool = True) -> list[str]:
    cols = NUM_BASE + ENGINEERED_COLS + CAT_COLS
    return cols + ["year"] if use_year else cols


def chrono_split(df: pd.DataFrame):
    """train = 2000-2020, validation = 2021-2026 (as required by AGENTS.md)."""
    train = df[df["year"] <= TRAIN_END].copy()
    val = df[df["year"] >= VAL_START].copy()
    assert train["year"].max() < val["year"].min(), "temporal leakage!"
    return train, val


# --------------------------------------------------------------------------- #
# 2. Metrics
# --------------------------------------------------------------------------- #
def regression_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100),
    }


# --------------------------------------------------------------------------- #
# 3. Models
# --------------------------------------------------------------------------- #
def fit_ridge(train: pd.DataFrame, val: pd.DataFrame, use_year: bool = True):
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    cols = feature_columns(use_year)
    cats = [c for c in CAT_COLS if c in cols]
    nums = [c for c in cols if c not in cats]
    pre = ColumnTransformer([
        ("num", StandardScaler(), nums),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cats),
    ])
    pipe = Pipeline([("pre", pre), ("ridge", RidgeCV(alphas=np.logspace(-2, 3, 30)))])
    pipe.fit(train[cols], train[TARGET])
    pred = pipe.predict(val[cols])
    return pipe, pred, regression_metrics(val[TARGET], pred)


def _lgbm_params() -> dict:
    return dict(
        objective="regression", n_estimators=3000, learning_rate=0.03,
        num_leaves=15, min_child_samples=10, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=1.0, cat_smooth=10, min_data_per_group=10,
        random_state=SEED, verbose=-1,
    )


def fit_lgbm(train: pd.DataFrame, val: pd.DataFrame, use_year: bool = True):
    """Early stopping uses 2018-2020 (inside TRAIN) so 2021-2026 stays untouched."""
    import lightgbm as lgb

    cols = feature_columns(use_year)
    fit_part = train[train["year"] < ES_START]
    es_part = train[train["year"] >= ES_START]

    model = lgb.LGBMRegressor(**_lgbm_params())
    model.fit(
        fit_part[cols], fit_part[TARGET],
        eval_set=[(es_part[cols], es_part[TARGET])], eval_metric="rmse",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    best_iter = int(model.best_iteration_ or model.n_estimators)
    pred = model.predict(val[cols])
    return model, best_iter, pred, regression_metrics(val[TARGET], pred)


def refit_full(df_all: pd.DataFrame, best_iter: int, use_year: bool = True):
    """Production model for Member 4: same settings, ALL years 2000-2026."""
    import lightgbm as lgb
    params = _lgbm_params() | {"n_estimators": max(best_iter, 50)}
    model = lgb.LGBMRegressor(**params)
    cols = feature_columns(use_year)
    model.fit(df_all[cols], df_all[TARGET])
    return model


# --------------------------------------------------------------------------- #
# 4. Validation suite
# --------------------------------------------------------------------------- #
def group_kfold_check(df: pd.DataFrame, n_splits: int = 5) -> dict:
    """Unseen-country test: countries in the test fold are never in training.
    Country is excluded here (the model cannot know an unseen country)."""
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold

    cols = [c for c in feature_columns(use_year=True) if c != "country"]
    scores = []
    for tr, te in GroupKFold(n_splits).split(df, groups=df["iso3"]):
        m = lgb.LGBMRegressor(**(_lgbm_params() | {"n_estimators": 400}))
        m.fit(df.iloc[tr][cols], df.iloc[tr][TARGET])
        scores.append(regression_metrics(df.iloc[te][TARGET], m.predict(df.iloc[te][cols])))
    return {k: float(np.mean([s[k] for s in scores])) for k in scores[0]}


def ablation(train, val) -> pd.DataFrame:
    """What does each feature group add? (validation R2 / RMSE)"""
    import lightgbm as lgb
    full = feature_columns(True)
    variants = {
        "full": full,
        "no_year": [c for c in full if c != "year"],
        "no_country": [c for c in full if c != "country"],
        "no_engineered": [c for c in full if c not in ENGINEERED_COLS],
        "raw_shares_only": FUEL_COLS,
    }
    rows = []
    for name, cols in variants.items():
        m = lgb.LGBMRegressor(**(_lgbm_params() | {"n_estimators": 600}))
        m.fit(train[cols], train[TARGET])
        rows.append({"variant": name, **regression_metrics(val[TARGET], m.predict(val[cols]))})
    return pd.DataFrame(rows).round(4)


def _shift(base: pd.DataFrame, src: str, dst: str, pp: float = 10.0) -> pd.DataFrame:
    """Move up to `pp` percentage points of the grid from `src` fuel to `dst` fuel."""
    x = base.copy()
    mv = np.minimum(x[src], pp)
    x[src] -= mv
    x[dst] += mv
    fossil, renew = {"coal_pct", "oil_pct", "gas_pct"}, {"solar_pct", "wind_pct", "hydro_pct", "other_renewables_pct"}
    if src in fossil and dst not in fossil:
        x["fossil_total_pct"] -= mv
    if dst in renew and src not in renew:
        x["renewables_total_pct"] += mv
    return x


def sanity_checks(bundle: dict, df_feat: pd.DataFrame) -> dict:
    """Directional / physical checks that Member 4's scenarios rely on."""
    base = df_feat[df_feat["year"] == df_feat["year"].max()].copy()
    res = {"n_countries": int(len(base))}
    p0 = predict_emissions(base, bundle=bundle)
    res["no_nans_in_predictions"] = bool(np.isfinite(p0).all())
    res["no_negative_predictions"] = bool((p0 >= 0).all())
    for name, mdl in [("lgbm", "lgbm"), ("ridge", "ridge")]:
        b = predict_emissions(base, bundle, mdl)
        for tag, dst in [("coal_to_gas", "gas_pct"), ("coal_to_solar", "solar_pct"), ("coal_to_wind", "wind_pct")]:
            d = predict_emissions(_shift(base, "coal_pct", dst), bundle, mdl) - b
            res[f"{name}_delta_{tag}"] = round(float(d.mean()), 3)
    r = res
    r["ridge_scenarios_physically_ordered"] = bool(
        r["ridge_delta_coal_to_gas"] < 0 and r["ridge_delta_coal_to_solar"] < r["ridge_delta_coal_to_gas"])
    return r


# --------------------------------------------------------------------------- #
# 5. Plot
# --------------------------------------------------------------------------- #
NON_ENERGY = {"country", "region", "year", "population_millions", "log_population"}


def plot_top_features(model, cols, path: Path, top_n: int = 10) -> pd.DataFrame:
    """Two panels: (left) all features, (right) energy-mix features only.
    Country/population dominate the left panel, so the right one answers
    'which fuels drive emissions?'."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gain = model.booster_.feature_importance(importance_type="gain")
    imp = (pd.DataFrame({"feature": cols, "gain": gain})
           .assign(gain_pct=lambda d: 100 * d["gain"] / d["gain"].sum())
           .sort_values("gain", ascending=False).reset_index(drop=True))
    imp["is_energy_feature"] = ~imp["feature"].isin(NON_ENERGY)
    energy = imp[imp["is_energy_feature"]].copy()
    energy["gain_pct_within_energy"] = 100 * energy["gain"] / energy["gain"].sum()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    t = imp.head(top_n).iloc[::-1]
    axes[0].barh(t["feature"], t["gain_pct"], color="#888780")
    axes[0].set_title("Top 10 - all features"); axes[0].set_xlabel("Share of total gain (%)")
    e = energy.head(top_n).iloc[::-1]
    axes[1].barh(e["feature"], e["gain_pct_within_energy"], color="#1D9E75")
    axes[1].set_title("Top 10 - energy-mix features only"); axes[1].set_xlabel("Share of energy-feature gain (%)")
    fig.suptitle("What drives CO2 per capita? (LightGBM, gain)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return imp.merge(energy[["feature", "gain_pct_within_energy"]], on="feature", how="left")


# --------------------------------------------------------------------------- #
# 6. Handoff API (Member 4 / integrator)
# --------------------------------------------------------------------------- #
def load_bundle(path: Path = MODEL_PATH) -> dict:
    """Bundle keys: model, features, categories, target, metrics, train_years, model_name"""
    return joblib.load(path)


def predict_emissions(df: pd.DataFrame, bundle: dict | None = None, model: str = "lgbm") -> np.ndarray:
    """RAW columns in -> predicted co2_per_capita_t out.

    model="lgbm"  : production LightGBM (best accuracy; use for levels / historical fit)
    model="ridge" : country-fixed-effects Ridge (smooth, physically ordered response;
                    use for scenario DELTAS - see predict_scenario_delta)

    df needs: country, region, year, population_millions, all *_pct fuel shares,
    renewables_total_pct, fossil_total_pct.  Derived columns are recomputed here.
    """
    bundle = bundle or load_bundle()
    X = build_features(df)[bundle["features"]].copy()
    for c, cats in bundle["categories"].items():          # lock category levels
        X[c] = pd.Categorical(X[c].astype(str), categories=cats)
    est = bundle["model"] if model == "lgbm" else bundle["scenario_model"]
    return est.predict(X)


def predict_scenario_delta(baseline: pd.DataFrame, scenario: pd.DataFrame,
                           bundle: dict | None = None) -> np.ndarray:
    """Change in CO2 per capita caused by moving from `baseline` to `scenario`
    (same rows, edited fuel shares).  Uses the fixed-effects Ridge because it
    responds smoothly and in the physically correct direction to fuel shifts.
    Recommended use:  lgbm_level(baseline) + delta."""
    bundle = bundle or load_bundle()
    return (predict_emissions(scenario, bundle, "ridge")
            - predict_emissions(baseline, bundle, "ridge"))


def load_feature_table(path: Path = FEATURES_OUT) -> pd.DataFrame:
    """Engineered dataset for plots (includes actuals + predictions + split flag)."""
    return pd.read_csv(path)


# --------------------------------------------------------------------------- #
# 7. Orchestrator
# --------------------------------------------------------------------------- #
def run_pipeline(save: bool = True, verbose: bool = True) -> dict:
    df_raw = load_clean()
    df = build_features(df_raw.drop(columns=LEAKAGE_COLS))
    assert not set(LEAKAGE_COLS) & set(df.columns), "leakage column survived"
    assert not set(LEAKAGE_COLS) & set(feature_columns()), "leakage column in features"

    train, val = chrono_split(df)
    # share category levels so train/val encode identically
    cats = {c: sorted(df[c].astype(str).unique()) for c in CAT_COLS}
    for part in (train, val, df):
        for c in CAT_COLS:
            part[c] = pd.Categorical(part[c].astype(str), categories=cats[c])

    # ---- models -----------------------------------------------------------
    ridge_pipe, ridge_pred, ridge_m = fit_ridge(train, val)
    lgbm, best_iter, lgbm_pred, lgbm_m = fit_lgbm(train, val)
    cols = feature_columns()

    # ---- validation suite -------------------------------------------------
    abl = ablation(train, val)
    gkf = group_kfold_check(df)
    imp = plot_top_features(lgbm, cols, OUT_DIR / "q1_2_feature_importance.png")

    # ---- production model + bundle ---------------------------------------
    final_model = refit_full(df, best_iter)
    ridge_full = fit_ridge(df, val)[0]           # refit below on ALL years for scenarios
    ridge_full.fit(df[cols], df[TARGET])
    bundle = {
        "model": final_model, "scenario_model": ridge_full, "features": cols, "categories": cats,
        "target": TARGET, "model_name": "LightGBM Regressor",
        "train_years": (int(df["year"].min()), int(df["year"].max())),
        "best_iteration": best_iter, "metrics": lgbm_m,
        "notes": ("Validated on 2021-2026 with a 2000-2020 model; shipped models refit on 2000-2026. "
                  "model=LightGBM (levels), scenario_model=FE-Ridge (scenario deltas)."),
    }
    checks = sanity_checks(bundle, df)

    # ---- tables for visualisation ----------------------------------------
    feat = df.copy()
    feat["split"] = np.where(feat["year"] <= TRAIN_END, "train", "validation")
    feat["pred_co2_per_capita_t"] = np.nan
    feat.loc[val.index, "pred_co2_per_capita_t"] = lgbm_pred
    feat["ridge_pred_co2_per_capita_t"] = np.nan
    feat.loc[val.index, "ridge_pred_co2_per_capita_t"] = ridge_pred
    feat["residual"] = feat[TARGET] - feat["pred_co2_per_capita_t"]

    contract = {
        "module": "Q1.2_CO2_Emissions",
        "entity": "GLOBAL_50_COUNTRIES",
        "as_of_date": "2026-12-31",
        "horizon": "validation_2021_2026",
        "target": TARGET,
        "predictions": lgbm_pred.round(4).tolist(),
        "actuals": val[TARGET].round(4).tolist(),
        "metrics": {"r2": round(lgbm_m["r2"], 4), "rmse": round(lgbm_m["rmse"], 4)},
        "features_used": cols,
        "model_name": "LightGBM Regressor",
    }

    if save:
        for p in (OUT_DIR, MODEL_PATH.parent):
            p.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, MODEL_PATH)
        feat.to_csv(FEATURES_OUT, index=False)
        imp.to_csv(OUT_DIR / "q1_2_feature_importance.csv", index=False)
        abl.to_csv(OUT_DIR / "q1_2_ablation.csv", index=False)
        summary = {
            "ridge_val": ridge_m, "lgbm_val": lgbm_m, "group_kfold_unseen_countries": gkf,
            "sanity_checks": checks, "best_iteration": best_iter,
        }
        (OUT_DIR / "q1_2_metrics.json").write_text(json.dumps(summary, indent=2))

    if verbose:
        print("Ridge  :", {k: round(v, 4) for k, v in ridge_m.items()})
        print("LightGBM:", {k: round(v, 4) for k, v in lgbm_m.items()}, "| best_iter", best_iter)
        print("GroupKFold (unseen countries):", {k: round(v, 4) for k, v in gkf.items()})
        print("\nAblation:\n", abl.to_string(index=False))
        print("\nTop 10 features:\n", imp.head(10).round(3).to_string(index=False))
        print("\nSanity:", checks)

    contract["_extras"] = {"ridge_metrics": ridge_m, "lgbm_metrics": lgbm_m,
                           "group_kfold": gkf, "sanity": checks}
    return contract


if __name__ == "__main__":
    out = run_pipeline()
    print("\nOUTPUT CONTRACT:", {k: v for k, v in out.items()
                                 if k not in ("predictions", "actuals", "features_used", "_extras")})

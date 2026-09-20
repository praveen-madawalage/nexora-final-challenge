"""Run:  pytest tests/test_q1_emissions.py -q      (after src/q1_emissions.py has been run once)"""
import numpy as np
import pandas as pd
import pytest

from src import q1_emissions as q


@pytest.fixture(scope="module")
def raw():
    return q.load_clean()


@pytest.fixture(scope="module")
def bundle():
    return q.load_bundle()


def test_raw_contract(raw):
    assert raw.shape[0] == 1350 and raw["iso3"].nunique() == 50
    assert raw["year"].min() == 2000 and raw["year"].max() == 2026
    assert raw.isna().sum().sum() == 0


def test_no_leakage_columns_in_features():
    assert not set(q.LEAKAGE_COLS) & set(q.feature_columns())
    assert q.TARGET not in q.feature_columns()


def test_leakage_columns_really_leak(raw):
    """Documents WHY they are dropped: emissions / population reproduces the target."""
    r = np.corrcoef(raw["co2_emissions_mt"] / raw["population_millions"], raw[q.TARGET])[0, 1]
    assert r > 0.99


def test_chronological_split(raw):
    df = q.build_features(raw.drop(columns=q.LEAKAGE_COLS))
    tr, va = q.chrono_split(df)
    assert tr["year"].max() == 2020 and va["year"].min() == 2021
    assert len(tr) + len(va) == 1350


def test_engineered_values(raw):
    f = q.build_features(raw)
    assert np.allclose(f["vre_pct"], raw["solar_pct"] + raw["wind_pct"])
    assert np.allclose(f["clean_baseload_pct"], raw["clean_baseload_pct"], atol=0.02)
    exp = (raw.coal_pct * 1 + raw.oil_pct * .75 + raw.gas_pct * .45) / 100
    assert np.allclose(f["grid_emission_factor"], exp)
    assert f[q.ENGINEERED_COLS].isna().sum().sum() == 0
    assert np.isfinite(f[q.ENGINEERED_COLS].to_numpy()).all()


def test_bundle_keys(bundle):
    for k in ("model", "scenario_model", "features", "categories", "target", "metrics", "model_name"):
        assert k in bundle


def test_validation_target_met(bundle):
    assert bundle["metrics"]["r2"] > 0.85


def test_predict_from_raw_columns(raw, bundle):
    p = q.predict_emissions(raw.head(50), bundle=bundle)
    assert p.shape == (50,) and np.isfinite(p).all()


def test_unseen_country_does_not_crash(raw, bundle):
    x = raw.head(5).copy()
    x["country"], x["iso3"] = "Atlantis", "ATL"
    assert np.isfinite(q.predict_emissions(x, bundle=bundle)).all()


def test_scenario_delta_direction(raw, bundle):
    base = raw[raw["year"] == 2026]
    coal_solar = q.predict_scenario_delta(base, q._shift(base, "coal_pct", "solar_pct"), bundle)
    coal_gas = q.predict_scenario_delta(base, q._shift(base, "coal_pct", "gas_pct"), bundle)
    assert coal_solar.mean() < coal_gas.mean() < 0


def test_saved_tables_exist():
    t = q.load_feature_table()
    assert {"pred_co2_per_capita_t", "split", "grid_emission_factor"} <= set(t.columns)
    assert len(t) == 1350

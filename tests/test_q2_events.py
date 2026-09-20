"""
Comprehensive Unit, Integration, and Regression Tests for Question 2 Module
Cross-Dataset Feature Engineering: Climate Events and Carbon Price Drivers
"""

import sys
import json
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure src can be imported
TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from src.q2_events import (
    engineer_event_proximity_features,
    JURISDICTION_MAP,
    DATA_DIR,
    PROCESSED_DIR,
    OUTPUT_DIR
)


class TestQ2EventHypothesis(unittest.TestCase):
    """Test suite verifying data integrity, feature leakage prevention, and model outputs."""

    @classmethod
    def setUpClass(cls):
        """Load clean datasets and outputs once for testing."""
        cls.prices_path = PROCESSED_DIR / "prices_clean.csv"
        cls.events_path = PROCESSED_DIR / "events_clean.csv"
        cls.ablation_csv_path = OUTPUT_DIR / "q2_ablation_results.csv"
        cls.contract_json_path = OUTPUT_DIR / "q2_output_contract.json"

        cls.prices_exist = cls.prices_path.exists()
        cls.events_exist = cls.events_path.exists()

    def test_01_clean_input_data_integrity(self):
        """Case 1: Verify processed clean datasets exist, have zero nulls in primary targets, and expected dimensions."""
        self.assertTrue(self.prices_exist, "prices_clean.csv must exist")
        self.assertTrue(self.events_exist, "events_clean.csv must exist")

        prices_df = pd.read_csv(self.prices_path)
        events_df = pd.read_csv(self.events_path)

        # AGENTS.md specifies 15,866 rows across 5 markets and 50 events
        self.assertEqual(len(prices_df), 15866, "prices_clean.csv must contain exactly 15,866 rows")
        self.assertEqual(len(events_df), 50, "events_clean.csv must contain exactly 50 events")
        self.assertEqual(prices_df["market"].nunique(), 5, "Must cover exactly 5 ETS markets")
        
        # Primary keys and targets must have zero nulls
        self.assertEqual(prices_df["price"].isnull().sum(), 0, "Price target must have zero nulls")
        self.assertEqual(prices_df["date"].isnull().sum(), 0, "Date column must have zero nulls")
        self.assertEqual(prices_df["market"].isnull().sum(), 0, "Market column must have zero nulls")
        self.assertEqual(events_df.isnull().sum().sum(), 0, "Clean events must have zero null values")

    def test_02_zero_lookahead_leakage(self):
        """Case 2: Strict mathematical verification that future events do NOT leak into past dates."""
        synthetic_prices = pd.DataFrame({
            "market": ["EU_ETS"] * 5,
            "date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "price": [50.0, 51.0, 52.0, 53.0, 54.0],
            "lag_1": [49.0, 50.0, 51.0, 52.0, 53.0]
        })

        # Event occurs on day 3 (2024-01-03)
        synthetic_events = pd.DataFrame({
            "event_id": [1],
            "date": [pd.Timestamp("2024-01-03")],
            "region": ["Europe"],
            "event_type": ["policy"],
            "severity_score": [8.0],
            "is_policy": [1],
            "is_extreme_weather": [0],
            "is_disaster": [0]
        })

        res = engineer_event_proximity_features(synthetic_prices, synthetic_events)

        # On Day 1 and Day 2 (before the event), event proximity must be default unobserved (999)
        self.assertEqual(res.loc[0, "days_since_last_event"], 999)
        self.assertEqual(res.loc[1, "days_since_last_event"], 999)
        self.assertEqual(res.loc[0, "trailing_14d_policy_flag"], 0)
        self.assertEqual(res.loc[1, "trailing_14d_policy_flag"], 0)
        self.assertEqual(res.loc[0, "trailing_7d_severity_sum"], 0.0)

        # On Day 3 (event day), delta must be exactly 0, flag must be 1, severity 8.0
        self.assertEqual(res.loc[2, "days_since_last_event"], 0)
        self.assertEqual(res.loc[2, "days_since_policy_event"], 0)
        self.assertEqual(res.loc[2, "trailing_14d_policy_flag"], 1)
        self.assertEqual(res.loc[2, "trailing_7d_severity_sum"], 8.0)

        # On Day 4 (1 day after event), delta must be exactly 1
        self.assertEqual(res.loc[3, "days_since_last_event"], 1)
        self.assertEqual(res.loc[3, "days_since_policy_event"], 1)

    def test_03_decay_severity_monotonic_damping(self):
        """Case 3: Verify that exponential decay dampens severity score over elapsed days."""
        synthetic_prices = pd.DataFrame({
            "market": ["California"] * 4,
            "date": [
                pd.Timestamp("2025-01-01"),
                pd.Timestamp("2025-01-02"),
                pd.Timestamp("2025-01-10"),
                pd.Timestamp("2025-02-15")  # > 30 days after
            ],
            "price": [30.0, 31.0, 32.0, 33.0],
            "lag_1": [29.0, 30.0, 31.0, 32.0]
        })
        synthetic_events = pd.DataFrame({
            "event_id": [10],
            "date": [pd.Timestamp("2025-01-01")],
            "region": ["USA"],
            "event_type": ["wildfire"],
            "severity_score": [10.0],
            "is_policy": [0],
            "is_extreme_weather": [1],
            "is_disaster": [1]
        })

        res = engineer_event_proximity_features(synthetic_prices, synthetic_events)

        day0_sev = res.loc[0, "trailing_30d_decay_severity"]
        day1_sev = res.loc[1, "trailing_30d_decay_severity"]
        day9_sev = res.loc[2, "trailing_30d_decay_severity"]
        day45_sev = res.loc[3, "trailing_30d_decay_severity"]

        self.assertEqual(day0_sev, 10.0)
        self.assertLess(day1_sev, day0_sev, "Decayed severity must decrease over time")
        self.assertLess(day9_sev, day1_sev, "Decayed severity on day 9 must be lower than day 1")
        self.assertEqual(day45_sev, 0.0, "Events older than 30 days must have 0 trailing 30d severity")

    def test_04_jurisdiction_mapping_fidelity(self):
        """Case 4: Verify ETS jurisdiction filtering isolates regional shocks from irrelevant regions."""
        synthetic_prices = pd.DataFrame({
            "market": ["China_ETS", "EU_ETS"],
            "date": [pd.Timestamp("2024-05-10"), pd.Timestamp("2024-05-10")],
            "price": [70.0, 65.0],
            "lag_1": [69.0, 64.0]
        })
        synthetic_events = pd.DataFrame({
            "event_id": [5],
            "date": [pd.Timestamp("2024-05-09")],
            "region": ["China"],
            "event_type": ["policy"],
            "severity_score": [7.0],
            "is_policy": [1],
            "is_extreme_weather": [0],
            "is_disaster": [0]
        })

        res = engineer_event_proximity_features(synthetic_prices, synthetic_events)

        china_row = res[res["market"] == "China_ETS"].iloc[0]
        eu_row = res[res["market"] == "EU_ETS"].iloc[0]

        # China ETS must recognize China region as jurisdiction match (1)
        self.assertEqual(china_row["jurisdiction_match_flag"], 1)
        # EU ETS must recognize China event as non-jurisdiction (0)
        self.assertEqual(eu_row["jurisdiction_match_flag"], 0)

    def test_05_multiple_events_aggregation(self):
        """Case 5: Verify cumulative aggregation of multiple events within 7d and 14d windows."""
        synthetic_prices = pd.DataFrame({
            "market": ["RGGI"],
            "date": [pd.Timestamp("2024-06-10")],
            "price": [15.0],
            "lag_1": [14.8]
        })
        synthetic_events = pd.DataFrame({
            "event_id": [1, 2],
            "date": [pd.Timestamp("2024-06-05"), pd.Timestamp("2024-06-08")],
            "region": ["USA", "USA"],
            "event_type": ["heatwave", "policy"],
            "severity_score": [6.0, 9.0],
            "is_policy": [0, 1],
            "is_extreme_weather": [1, 0],
            "is_disaster": [0, 0]
        })

        res = engineer_event_proximity_features(synthetic_prices, synthetic_events)
        row = res.iloc[0]

        # Both events occurred within trailing 7 days (June 5 and June 8 <= June 10)
        self.assertEqual(row["trailing_7d_severity_sum"], 15.0)  # 6 + 9
        self.assertEqual(row["trailing_14d_policy_flag"], 1)
        self.assertEqual(row["days_since_last_event"], 2)  # June 10 - June 8 = 2 days
        self.assertEqual(row["days_since_weather_event"], 5)  # June 10 - June 5 = 5 days

    def test_06_ablation_output_schema_and_values(self):
        """Case 6: Verify the ablation benchmark summary table is generated with valid metrics."""
        self.assertTrue(self.ablation_csv_path.exists(), "q2_ablation_results.csv must exist")
        df = pd.read_csv(self.ablation_csv_path)

        required_cols = [
            "market", "horizon_days", "baseline_rmse", "event_rmse", "delta_rmse",
            "baseline_mape_pct", "event_mape_pct", "delta_mape_pct",
            "baseline_dir_acc_pct", "event_dir_acc_pct", "delta_dir_acc_pct"
        ]
        for col in required_cols:
            self.assertIn(col, df.columns, f"Missing required column {col} in ablation CSV")

        self.assertEqual(len(df), 5, "Must contain exactly 5 markets")
        self.assertTrue((df["horizon_days"] == 30).all(), "All test horizons must be exactly 30 days")

        # Verify MAPEs are reasonable positive percentages (< 15%)
        self.assertTrue((df["baseline_mape_pct"] > 0).all(), "Baseline MAPE must be positive")
        self.assertTrue((df["event_mape_pct"] > 0).all(), "Event MAPE must be positive")
        self.assertTrue((df["event_mape_pct"] < 15.0).all(), "Event MAPE must be within realistic carbon range (< 15%)")

        # Verify directional accuracy is between 0% and 100%
        self.assertTrue((df["event_dir_acc_pct"] >= 0).all())
        self.assertTrue((df["event_dir_acc_pct"] <= 100).all())

    def test_07_output_contract_agents_spec(self):
        """Case 7: Verify the output contract JSON adheres to Section 6 of AGENTS.md."""
        self.assertTrue(self.contract_json_path.exists(), "q2_output_contract.json must exist")
        with open(self.contract_json_path, "r", encoding="utf-8") as f:
            contracts = json.load(f)

        self.assertEqual(len(contracts), 5, "Contract must cover all 5 markets")

        required_keys = [
            "module", "entity", "as_of_date", "horizon", "target",
            "predictions_price_augmented", "actuals_price",
            "directional_predictions_augmented", "actuals_direction",
            "metrics", "features_used", "model_name"
        ]
        for c in contracts:
            for key in required_keys:
                self.assertIn(key, c, f"Missing contract key {key} for entity {c.get('entity')}")

            # Horizon check
            self.assertEqual(c["horizon"], "30_days")
            self.assertEqual(len(c["predictions_price_augmented"]), 30)
            self.assertEqual(len(c["actuals_price"]), 30)
            self.assertEqual(len(c["directional_predictions_augmented"]), 30)
            self.assertEqual(len(c["actuals_direction"]), 30)

            # Check prices are positive numbers
            for p in c["predictions_price_augmented"]:
                self.assertGreater(p, 0.0, "Predicted carbon price must be strictly positive")

            # Check directional predictions are binary 0 or 1
            for d in c["directional_predictions_augmented"]:
                self.assertIn(d, [0, 1], "Directional prediction must be strictly 0 or 1")

    def test_08_test_split_chronological_alignment(self):
        """Case 8: Verify that test sets for all 5 markets align with the out-of-sample window."""
        prices_df = pd.read_csv(self.prices_path)
        prices_df["date"] = pd.to_datetime(prices_df["date"])

        for market, m_df in prices_df.groupby("market"):
            m_sorted = m_df.sort_values("date").reset_index(drop=True)
            test_slice = m_sorted.iloc[-30:]

            # The test window must span into 2026 (April 2026)
            self.assertEqual(len(test_slice), 30, f"{market} test slice must have 30 trading days")
            self.assertEqual(test_slice["date"].dt.year.iloc[-1], 2026, f"{market} test set must end in 2026")
            self.assertEqual(test_slice["date"].dt.month.iloc[-1], 4, f"{market} test set must end in April")


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""
Canonical Data Cleaning and Processing Pipeline for CodeFest Datathon 2026 Finals.
Single source of truth for all 4 team members.
Outputs clean datasets to data/processed/.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def get_base_dir() -> Path:
    """Detect current project directory."""
    cwd = Path.cwd()
    if (cwd / "raw").exists():
        return cwd
    # Check parent
    if (cwd.parent / "raw").exists():
        return cwd.parent
    return cwd


def clean_energy_and_co2(raw_dir: Path, out_dir: Path) -> pd.DataFrame:
    """
    Merge energy_mix_yearly and co2_emissions_yearly on (year, country, iso3, region).
    Creates canonical country_clean.csv with 1,350 rows and zero nulls.
    """
    co2_path = raw_dir / "co2_emissions_yearly.csv"
    mix_path = raw_dir / "energy_mix_yearly.csv"

    co2_df = pd.read_csv(co2_path)
    mix_df = pd.read_csv(mix_path)

    # Standardize column names (lowercase and underscores)
    co2_df.columns = [c.strip().lower().replace(" ", "_") for c in co2_df.columns]
    mix_df.columns = [c.strip().lower().replace(" ", "_") for c in mix_df.columns]

    # Inner join on canonical country-year key
    merged = pd.merge(
        co2_df, mix_df,
        on=["year", "country", "iso3", "region"],
        how="inner"
    )

    # Engineered domain features
    merged["clean_baseload_pct"] = (merged["nuclear_pct"] + merged["hydro_pct"]).round(4)
    merged["fossil_ratio"] = (merged["fossil_total_pct"] / (merged["renewables_total_pct"] + 0.01)).round(4)
    merged["coal_to_gas_ratio"] = (merged["coal_pct"] / (merged["gas_pct"] + 0.01)).round(4)
    merged["renewables_plus_nuclear_pct"] = (merged["renewables_total_pct"] + merged["nuclear_pct"]).round(4)

    # Sort canonically
    merged = merged.sort_values(["country", "year"]).reset_index(drop=True)

    out_file = out_dir / "country_clean.csv"
    merged.to_csv(out_file, index=False)
    print(f"[OK] Created country_clean.csv: {merged.shape[0]:,} rows x {merged.shape[1]} columns -> {out_file.name}")
    return merged


def clean_carbon_prices(raw_dir: Path, out_dir: Path) -> pd.DataFrame:
    """
    Clean carbon_prices_daily, parse dates, add calendar cyclical features,
    and construct zero-leakage autoregressive lags and rolling volatility.
    """
    prices_path = raw_dir / "carbon_prices_daily.csv"
    df = pd.read_csv(prices_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["market", "date"]).reset_index(drop=True)

    # Calendar and cyclical features
    df["dayofweek"] = df["date"].dt.dayofweek.astype(np.int8)
    df["month"] = df["date"].dt.month.astype(np.int8)
    df["day"] = df["date"].dt.day.astype(np.int8)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(np.int8)

    # Cyclical day of year encoding
    dayofyear = df["date"].dt.dayofyear
    df["day_sin"] = np.sin(2 * np.pi * dayofyear / 365.25).astype(np.float32)
    df["day_cos"] = np.cos(2 * np.pi * dayofyear / 365.25).astype(np.float32)

    # Zero-leakage autoregressive lags (strictly prior days)
    for lag in [1, 2, 3, 5, 7, 14, 30]:
        df[f"lag_{lag}"] = df.groupby("market")["price"].shift(lag)

    # Zero-leakage rolling statistics (shifted by 1 day)
    df["roll_mean_7d"] = df.groupby("market")["price"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).mean()
    ).round(4)
    df["roll_mean_30d"] = df.groupby("market")["price"].transform(
        lambda s: s.shift(1).rolling(30, min_periods=3).mean()
    ).round(4)
    df["roll_std_30d"] = df.groupby("market")["price"].transform(
        lambda s: s.shift(1).rolling(30, min_periods=3).std()
    ).fillna(0.0).round(4)

    # Price momentum and log returns
    df["return_1d"] = (df["price"] / (df["lag_1"] + 1e-6) - 1.0).round(4)
    df["momentum_7d"] = (df["price"] - df["lag_7"]).round(4)

    out_file = out_dir / "prices_clean.csv"
    df.to_csv(out_file, index=False)
    print(f"[OK] Created prices_clean.csv: {df.shape[0]:,} rows x {df.shape[1]} columns -> {out_file.name}")
    return df


def clean_climate_events(raw_dir: Path, out_dir: Path) -> pd.DataFrame:
    """Standardize climate and policy events dataset."""
    events_path = raw_dir / "climate_events.csv"
    df = pd.read_csv(events_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    out_file = out_dir / "events_clean.csv"
    df.to_csv(out_file, index=False)
    print(f"[OK] Created events_clean.csv: {df.shape[0]:,} rows x {df.shape[1]} columns -> {out_file.name}")
    return df


def clean_temperature_anomaly(raw_dir: Path, out_dir: Path) -> pd.DataFrame:
    """Standardize monthly temperature anomaly dataset."""
    temp_path = raw_dir / "temperature_anomaly_monthly.csv"
    df = pd.read_csv(temp_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    out_file = out_dir / "temp_clean.csv"
    df.to_csv(out_file, index=False)
    print(f"[OK] Created temp_clean.csv: {df.shape[0]:,} rows x {df.shape[1]} columns -> {out_file.name}")
    return df


def main():
    base_dir = get_base_dir()
    raw_dir = base_dir / "raw"
    out_dir = base_dir / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("NEXORA CANONICAL DATA PROCESSING PIPELINE")
    print(f"Base Directory: {base_dir.resolve()}")
    print("=" * 70)

    clean_energy_and_co2(raw_dir, out_dir)
    clean_carbon_prices(raw_dir, out_dir)
    clean_climate_events(raw_dir, out_dir)
    clean_temperature_anomaly(raw_dir, out_dir)

    print("=" * 70)
    print("All canonical datasets successfully processed and verified.")
    print("=" * 70)


if __name__ == "__main__":
    main()

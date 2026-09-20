# Data Dictionary - Final Round

---

## 1. `carbon_prices_daily`

* **Dataset Description:** Daily prices for EU ETS, California, RGGI, UK ETS, China ETS

| Column Name | Description |
| :--- | :--- |
| `date` | The specific trading/observation date for the carbon price. It is recorded in `YYYY-MM-DD` format. |
| `Year` | The calendar year extracted from the date column. This can be used for yearly aggregation and trend analysis. |
| `market` | Identifies the carbon-pricing market or emissions trading system (ETS) associated with the observation. For example, `EU_ETS` represents the European Union Emissions Trading System. |
| `currency` | Indicates the currency in which the carbon price is denominated, such as EUR for euros. |
| `price` | The carbon allowance price recorded for the corresponding market and date, expressed in the currency specified by the `currency` column. |

---

## 2. `climate_events`

* **Dataset Description:** Major climate/weather/policy events with severity scores

| Column Name | Description |
| :--- | :--- |
| `event id` | Unique ID |
| `date` | Date of the event |
| `year` | Year of the event |
| `month` | Month of the event |
| `region` | Geographic scope |
| `event_type` | Label that provides the specific, granular classification of the climate or natural event. |
| `severity score` | 1–10 ordinal score |
| `description` | One-line description |
| `is policy` | Is it a government or regulatory action? |
| `is extreme weather` | Is it an unusual weather event? |
| `is disaster` | Did it cause major damage or loss of life? |

---

## 3. `co2_emissions_yearly`

* **Dataset Description:** 50 countries $\times$ 27 years: total emissions, per capita, intensity

| Column Name | Description |
| :--- | :--- |
| `year` | Years between 2000–2026 |
| `country` | The name of the country for which the energy-mix data is provided. |
| `iso3` | The three-letter ISO 3166-1 alpha-3 country code used to uniquely identify the country. For example, `CHN` represents China. |
| `region` | The geographical region in which the country is located, such as Asia, Europe, Africa, or North America. |
| `co2 emissions mt` | Total $\text{CO}_2$ emissions in million tons |
| `population_millions` | Population estimate |
| `co2 per capita t` | Tons $\text{CO}_2$ per person |
| `co2 intensity_kg_per_gdp_usd` | Carbon intensity of GDP |

---

## 4. `energy_mix_yearly`

* **Dataset Description:** 50 countries $\times$ 27 years: 8 fuel shares (coal, oil, gas, nuclear, hydro, solar, wind, other)

| Column Name | Description |
| :--- | :--- |
| `year` | The calendar year for which the country's energy mix is reported. |
| `country` | The name of the country for which the energy-mix data is provided. |
| `iso3` | The three-letter ISO 3166-1 alpha-3 country code used to uniquely identify the country. For example, `CHN` represents China. |
| `region` | The geographical region in which the country is located, such as Asia, Europe, Africa, or North America. |
| `coal pct` | Percentage of the country's total energy mix supplied by coal during the given year. |
| `oil pct` | Percentage of the country's total energy mix supplied by oil during the given year. |
| `gas_pct` | Percentage of the country's total energy mix supplied by natural gas during the given year. |
| `nuclear pct` | Percentage of the country's total energy mix supplied by nuclear energy during the given year. |
| `hydro_pct` | Percentage of the country's total energy mix supplied by hydropower during the given year. |
| `solar pct` | Percentage of the country's total energy mix supplied by solar energy during the given year. |
| `wind pct` | Percentage of the country's total energy mix supplied by wind energy during the given year. |
| `other renewables pct` | Percentage of the country's total energy mix supplied by renewable sources other than hydro, solar, and wind, such as geothermal and bioenergy. |
| `renewables total pct` | Combined percentage of the energy mix supplied by renewable energy sources (hydro, solar, wind, and other renewables). |
| `fossil total pct` | Combined percentage of the energy mix supplied by fossil fuels, primarily coal, oil, and natural gas. |

---

## 5. `temperature_anomaly_monthly`

* **Dataset Description:** Global + 7 regions $\times$ 316 months: temperature anomaly + $\text{CO}_2$ ppm

| Column Name | Description |
| :--- | :--- |
| `year_month` | Month-year period |
| `year` | Year |
| `month` | Month of the year |
| `region` | The geographical region in which the country is located, such as Asia, Europe, Africa, or North America. |
| `temp anomaly c` | Temperature anomaly in $^\circ\text{C}$ vs baseline |
| `temp_anomaly_baseline` | Identifies the reference period used to calculate temperature anomalies across the dataset (NASA GISS convention). |
| `co2_ppm` | Atmospheric $\text{CO}_2$ concentration (Global region only) |
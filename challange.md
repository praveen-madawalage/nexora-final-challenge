# CodeFest Datathon 2026 - Task Brief[cite: 2]

## 1) Overview[cite: 2]
Climate change is transforming global temperatures, energy systems, emissions, and economic policy at an unprecedented pace[cite: 2]. As governments and businesses accelerate the transition toward low-carbon economies, the ability to analyze complex climate and energy data has become essential for informed decision-making[cite: 2].

In CodeFest Datathon 2026, your team will explore a multi-source climate dataset spanning 2000–2026 and develop data-driven solutions that uncover meaningful trends, build predictive models, and propose innovative products that support the global energy transition[cite: 2].

Rather than solving a single prediction problem, this challenge emphasizes exploratory analytics, machine learning, feature engineering, forecasting, and business innovation[cite: 2].

---

## 2) Background Story[cite: 2]
Imagine you are part of a climate analytics consultancy advising governments, energy companies, financial institutions, and sustainability leaders[cite: 2].

Your clients need answers to questions such as[cite: 2]:
- How are carbon markets reacting to climate policies and extreme weather?[cite: 2]
- Which countries are transitioning fastest toward renewable energy?[cite: 2]
- How will different energy strategies affect future $\text{CO}_2$ emissions?[cite: 2]
- Can climate data be transformed into commercially valuable products?[cite: 2]

Using historical climate, emissions, energy mix, and carbon pricing data, your team will generate actionable insights and predictive models that help stakeholders make evidence-based decisions[cite: 2].

---

## 3) What You Will Be Given[cite: 2]
Participants will receive five CSV datasets containing historical climate and energy information[cite: 2].

| File | Description |
| :--- | :--- |
| `carbon_prices_daily.csv` | Daily carbon prices across five major emissions trading systems[cite: 2] |
| `climate_events.csv` | Major climate, policy, and extreme weather events[cite: 2] |
| `co2_emissions_yearly.csv` | Annual $\text{CO}_2$ emissions and emissions intensity for 50 countries[cite: 2] |
| `energy_mix_yearly.csv` | Country-level energy generation mix (coal, oil, gas, renewables, etc.)[cite: 2] |
| `temperature_anomaly_monthly.csv` | Monthly temperature anomalies and atmospheric $\text{CO}_2$ concentrations[cite: 2] |

> **Note:** All datasets are provided for use during the competition[cite: 2]. No external datasets or pre-trained models are permitted[cite: 2].

---

## 4) Challenge Objectives[cite: 2]

### Question 1 - Predictive Modeling for Climate and Energy Data[cite: 2]

#### 1.1 Forecasting Carbon Price Prediction[cite: 2]
- Build a model to forecast daily market carbon prices for the next 30 trading days[cite: 2].
- You must justify your modeling approach (classical statistical models vs. ML algorithms) and evaluate performance on a test set using RMSE or MAPE[cite: 2].

#### 1.2 Predicting $\text{CO}_2$ Emissions from Energy Mix[cite: 2]
- Build a regression model predicting `co2_per_capita_t` based on a country's energy mix profile (fuel shares across coal, oil, gas, and renewables)[cite: 2].
- You will be evaluated on data wrangling completeness, feature engineering, and predictive accuracy ($R^2$ / RMSE)[cite: 2].

---

### Question 2 - Cross-Dataset Feature Engineering - Carbon Price Drivers[cite: 2]
- Test the hypothesis that carbon prices respond to real-world climate events or policy shifts — a claim often assumed but rarely verified[cite: 2].
- Use `carbon_prices_daily.csv` with `climate_events.csv` to engineer event-proximity features (e.g., days since last major event, days until next known policy date)[cite: 2].
- Then prove that these features add predictive value, by benchmarking a price-prediction or up/down price-movement classification model with these features against an identical baseline model without them[cite: 2].

---

### Question 3 - Renewable Energy Transition Scenario Modelling[cite: 2]

#### 3.1 Energy Mix to $\text{CO}_2$ Relationship[cite: 2]
- Analyze the data to uncover global energy-transition patterns from 2000 to 2026[cite: 2].
- Create a visualization that maps energy mix shares (coal, oil, gas, renewables, nuclear) against $\text{CO}_2$ emissions footprints across the countries[cite: 2].
- Identify and categorize any distinct transition patterns or cluster archetypes that emerge[cite: 2].
- Identify the defined energy-mix trajectories for the past years[cite: 2]:
  - **Business-as-Usual:** The energy mix continues following its current trend[cite: 2].
  - **Moderate Transition:** Renewable energy gradually replaces fossil fuels[cite: 2].
  - **Accelerated Transition:** Renewable energy adoption increases significantly[cite: 2].
- You must justify your choice of visualization/s used[cite: 2].

#### 3.2 Forecasting - Future Emissions[cite: 2]
- Project annual $\text{CO}_2$ emissions for each selected country or region across the 2026–2030 forecast window[cite: 2].
- Generate annual $\text{CO}_2$ emissions forecasts (total or per capita) for 2026 through 2030[cite: 2].
- Explicitly mention all underlying assumptions, key parameter choices, and assumed annual rates of change driving your model[cite: 2].

#### 3.3 Key Insights & Findings[cite: 2]
- Highlight the most important discoveries from the data[cite: 2].
- Translate technical findings into meaningful business or real-world insights[cite: 2].

---

### 4. Pitch your Product[cite: 2]
Translate your technical data analysis and models into a viable, data-driven commercial product or service that addresses real-world Environmental, Social, and Governance (ESG), Financial, or regulatory challenges[cite: 2].

#### Core Deliverables[cite: 2]:
- **Value Proposition & Target Market:** Identify a specific target customer (e.g., ESG fund managers, corporate sustainability officers, commodities traders, municipal planners) and the pain point your solution solves[cite: 2].
- **Solution Architecture Diagram for the Proposed Product**[cite: 2]
- **Data & Predictive Modelling:** Explain how the dataset (emissions, energy mix, climate events, carbon prices) and your predictive models power the core functionality of your solution[cite: 2].
- **MVP & Technical Proof-of-Concept:** Demonstrate a working, low-fidelity prototype or functional dashboard (e.g., a Carbon Shock Alert System, Climate Risk Scoring API, or ESG Transition Screener)[cite: 2].
- **Commercial Strategy:** Define your monetization strategy (e.g., SaaS tier, API call billing, enterprise licensing) and pitch the business viability alongside your technical solution[cite: 2].

---

## 5) Dataset Schema[cite: 2]
> **Note:** Please refer to the `Data_Dictionary.xlsx` file for the dataset schema[cite: 2].

---

## 6) Submission Requirements[cite: 2]
Submit a single compressed folder containing[cite: 2]:
- `Teamname_notebook.ipynb`: Complete notebook with preprocessing, analysis, modelling, and visualizations[cite: 2].
- `Teamname_presentation.pptx`: Final presentation including the business pitch[cite: 2].
- `README.md` *(Optional)*: Brief explanation of methodology and assumptions[cite: 2].

### Important Notes[cite: 2]
- All code must be contained within the submitted notebook[cite: 2].
- Results should be reproducible from the provided datasets[cite: 2].
- Clearly document preprocessing and modelling decisions[cite: 2].
- Be prepared to explain your methodology during judging[cite: 2].

---

## 7) Rules & Regulations[cite: 2]
- All codes must be original work developed during the Datathon[cite: 2].
- Each team's final presentation must not exceed 10 minutes[cite: 2]. Presentations will be stopped immediately at the 10-minute mark, so teams should manage their time accordingly[cite: 2].
- Only the provided datasets may be used for model training and analysis[cite: 2]. External datasets, pretrained models, and pretrained weights are prohibited[cite: 2].
- AI tools (ChatGPT, Copilot, Gemini, etc.) may be used for debugging, documentation, and brainstorming, but teams must implement and understand their own solution[cite: 2].
- Teams must be able to explain their preprocessing, feature engineering, model architecture, and evaluation methodology[cite: 2].
- Any form of plagiarism, collusion, data leakage, or rule violation may result in disqualification[cite: 2].
- Judges' decisions are final[cite: 2].

---

## 8) Judging Criteria[cite: 2]

| Criterion | Focus |
| :--- | :--- |
| **Data Analysis** | Insight generation and exploratory analysis[cite: 2] |
| **Technical Quality** | Preprocessing, feature engineering, modelling[cite: 2] |
| **Predictive Performance** | Accuracy and evaluation methodology[cite: 2] |
| **Visualization** | Clarity and communication of findings[cite: 2] |
| **Business Innovation** | Practicality, originality, and commercial viability[cite: 2] |
| **Presentation** | Technical understanding and storytelling[cite: 2] |

---

*Good luck and help shape the future of climate intelligence through data[cite: 2].*
# Nexora: Climate Intelligence & Carbon Flow Analytics
## SLIIT CodeFest Datathon 2026 - Finals

### Quickstart & Setup
1. Clone the repository and checkout the integration branch:
   `ash
   git checkout dev
   `
2. Install dependencies:
   `ash
   pip install -r requirements.txt
   `
3. Run the canonical data pipeline (Single Source of Truth):
   `ash
   python src/data_loader.py
   `
   This generates the verified datasets into data/processed/.

4. Check out your feature branch according to your assignment in AGENTS.md:
   - Member 1: eat/q1.1-carbon-prices
   - Member 2: eat/q1.2-co2-regression
   - Member 3: eat/q2-event-hypothesis
   - Member 4: eat/q3-energy-scenarios

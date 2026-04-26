# Marketing Funnel Analysis

A Streamlit dashboard for exploring and experimenting with marketing funnels (visit → signup → activation → purchase). Load the built-in demo data, upload your own, or store data per company — then dig into conversion rates, cohorts, revenue, user journeys, A/B comparisons, and an ML-powered "what if" simulator.

## Features

- Interactive funnel charts, drop-off analysis, and segment breakdowns
- Cohort analysis and revenue analytics
- A/B comparison between segments (traffic source, device, country, etc.)
- ML-powered Impact Simulator that trains logistic regression models on your data and projects how UX improvements at each stage would change signups, activations, purchases, and revenue
- Multi-company data storage with role-based login (guest / company / admin)
- Upload data as CSV, Excel, JSON, or Parquet

## How it works

Raw event data comes from one of three sources: the built-in synthetic generator, an uploaded file, or the local database. An ETL layer turns those events into per-user flags and aggregated metrics. A Plotly-based visualization layer renders everything as interactive charts. The Impact Simulator trains a logistic regression per funnel transition (using traffic source, device, and country as features), then applies your "lift" via an odds-ratio update so a positive change always means more conversions.

## Tech stack

Python · Streamlit · Pandas · NumPy · Plotly · DuckDB · scikit-learn · bcrypt

## Running locally

```bash
streamlit run app.py --server.port 5000
```

Open http://localhost:5000 in your browser.

## Default login

You can click **Continue as Guest** to explore the demo data without logging in.

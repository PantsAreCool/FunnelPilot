# Marketing Funnel Analysis Application

## Overview
A comprehensive Streamlit-based data application for analyzing marketing funnels (visit → signup → activation → purchase). The app provides interactive visualizations, conversion rate analysis, and multi-dimensional breakdowns to help understand user behavior through the marketing funnel.

## Current State
- **Status**: Fully functional
- **Last Updated**: November 2024

## Features
- **Synthetic Data Generator**: Creates realistic demo data with 10,000 users and configurable drop-off rates
- **CSV Upload Support**: Import your own marketing event data for analysis
- **Interactive Filters**: Filter by traffic source, device, country, and date range
- **KPI Dashboard**: Top-line metrics including visits, signups, activations, purchases, and revenue
- **Funnel Visualization**: Interactive Plotly funnel chart showing stage progression
- **Drop-off Analysis**: Identify where users are leaving the funnel
- **Breakdown Charts**: Analyze conversion by traffic source, device, and country
- **Time-to-Conversion Analysis**: Box plots and histograms showing conversion timing

## Project Structure
```
├── app.py                      # Main Streamlit application
├── data/
│   ├── __init__.py
│   └── synthetic_generator.py  # Synthetic data generation & CSV validation
├── etl/
│   ├── __init__.py
│   └── funnel_etl.py           # ETL pipeline for funnel metrics
├── utils/
│   ├── __init__.py
│   └── plots.py                # Plotly visualization utilities
├── .streamlit/
│   └── config.toml             # Streamlit configuration
└── pyproject.toml              # Python dependencies
```

## Dependencies
- **streamlit**: Web framework for the data app
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical operations
- **plotly**: Interactive visualizations
- **duckdb**: SQL analytics on CSV files
- **statsmodels**: Statistical computations

## Running the Application
The app runs automatically on port 5000:
```bash
streamlit run app.py --server.port 5000
```

## CSV Upload Format
When uploading custom data, the CSV should include:

**Required columns:**
- `user_id` - Unique user identifier
- `event_name` - One of: visit, signup, activation, purchase
- `event_timestamp` - Event datetime

**Optional columns:**
- `traffic_source` - e.g., organic, paid_search, social
- `device` - e.g., desktop, mobile, tablet
- `country` - e.g., USA, UK, Germany
- `revenue` - Purchase revenue (numeric)

## Architecture Decisions
- Uses `st.cache_data` for performance optimization
- Modular design with separate data, ETL, and visualization layers
- DuckDB integration available for SQL-based analytics
- Responsive layout with wide page configuration

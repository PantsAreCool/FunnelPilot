# Marketing Funnel Analysis Application

## Overview
A comprehensive Streamlit-based data application for analyzing marketing funnels (visit → signup → activation → purchase). The app provides interactive visualizations, conversion rate analysis, multi-dimensional breakdowns, advanced analytics, and A/B testing capabilities to help understand user behavior through the marketing funnel.

## Current State
- **Status**: Fully functional with all features complete
- **Last Updated**: November 2024

## Features

### Core Analytics
- **Synthetic Data Generator**: Creates realistic demo data with 10,000 users and configurable drop-off rates
- **Multi-Format Upload Support**: Import data in CSV, Excel (.xlsx, .xls), JSON, or Parquet formats
- **Column Mapping Interface**: Auto-detection of columns with manual mapping option for custom schemas
- **Interactive Filters**: Filter by traffic source, device, country, and date range
- **KPI Dashboard**: Top-line metrics including visits, signups, activations, purchases, and revenue

### Funnel Analysis Tab
- **Funnel Visualization**: Interactive Plotly funnel chart showing stage progression
- **Drop-off Analysis**: Identify where users are leaving the funnel
- **Breakdown Charts**: Analyze conversion by traffic source, device, and country
- **Time-to-Conversion Analysis**: Box plots and histograms showing conversion timing

### Cohort Analysis Tab
- **Weekly/Monthly Cohorts**: Track user behavior by signup date cohorts
- **Cohort Retention Heatmaps**: Visualize retention across cohorts
- **Cohort Size Trends**: Monitor signup volume over time

### Revenue Analytics Tab
- **LTV Analysis**: Lifetime value calculations by segment
- **ARPU Metrics**: Average revenue per user by traffic source, device, country
- **Revenue Distribution**: Histograms and statistics for purchase amounts
- **Revenue Trends**: Time-series analysis of revenue patterns

### User Journeys Tab
- **Individual User Exploration**: View complete journey for any user
- **Journey Path Analysis**: Common path patterns through the funnel
- **User Search**: Find users by ID for detailed investigation

### A/B Comparison Tab
- **Segment Comparison**: Compare funnel performance between two segments
- **Comparison Dimensions**: Traffic source, device, or country
- **Statistical Metrics**: Conversion lift, ARPU comparison, significance indicators
- **Side-by-Side Charts**: Visual comparison of funnel stages

### Export Functionality
- **CSV Export**: Download analysis data as CSV files
- **Excel Export**: Download formatted Excel spreadsheets with multiple sheets
- **Available for All Sections**: Breakdowns, cohorts, revenue, journeys, A/B comparisons

## Project Structure
```
├── app.py                      # Main Streamlit application (1400+ lines)
├── data/
│   ├── __init__.py
│   └── synthetic_generator.py  # Synthetic data generation, multi-format file reading, column mapping
├── etl/
│   ├── __init__.py
│   └── funnel_etl.py           # ETL pipeline: funnel metrics, cohorts, revenue, journeys, A/B comparison
├── utils/
│   ├── __init__.py
│   └── plots.py                # Plotly visualization utilities (650+ lines)
├── .streamlit/
│   └── config.toml             # Streamlit configuration
└── pyproject.toml              # Python dependencies
```

## Dependencies
- **streamlit**: Web framework for the data app
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical operations
- **plotly**: Interactive visualizations
- **duckdb**: SQL analytics on data files
- **statsmodels**: Statistical computations
- **openpyxl**: Excel file reading and writing
- **pyarrow**: Parquet file support

## Running the Application
The app runs automatically on port 5000:
```bash
streamlit run app.py --server.port 5000
```

## Data Upload Format
When uploading custom data, the file should include:

**Required columns:**
- `user_id` - Unique user identifier
- `event_name` - One of: visit, signup, activation, purchase
- `event_timestamp` - Event datetime

**Optional columns:**
- `traffic_source` - e.g., organic, paid_search, social
- `device` - e.g., desktop, mobile, tablet
- `country` - e.g., USA, UK, Germany
- `revenue` - Purchase revenue (numeric)

**Supported formats:**
- CSV (.csv)
- Excel (.xlsx, .xls)
- JSON (.json) - records orientation
- Parquet (.parquet)

## Architecture Decisions
- Uses `st.cache_data` for performance optimization
- Modular design with separate data, ETL, and visualization layers
- DuckDB integration available for SQL-based analytics
- Responsive layout with wide page configuration
- BytesIO used for in-memory file generation (exports)
- Conversion rates use `step_conversion_rate` column naming convention

## Key Implementation Notes
- ETL functions in `funnel_etl.py` return DataFrames with `step_conversion_rate` (not `stage_conversion_rate`)
- A/B comparison uses `calculate_ab_comparison()` function returning comparison DataFrame and summary dict
- All export functions generate files in-memory using BytesIO for download buttons

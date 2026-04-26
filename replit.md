# Marketing Funnel Analysis Application

## Overview

A Streamlit-based data analytics dashboard for analyzing marketing funnels (visit → signup → activation → purchase). The application supports synthetic demo data generation, user file uploads in multiple formats (CSV, Excel, JSON, Parquet), and persistent multi-company data storage using DuckDB. Key features include interactive funnel visualizations, cohort analysis, revenue analytics, and user journey exploration.

## System Architecture

### Application Framework
- **Frontend/Backend**: Streamlit serves as both the UI framework and application server
- **Entry Point**: `app.py` is the main application file; `main.py` is a placeholder

### Module Organization
The codebase follows a clean separation of concerns:

| Module | Purpose |
|--------|---------|
| `data/` | Data generation and database management |
| `etl/` | Data transformation and funnel calculations |
| `utils/` | Visualization and helper functions |

### Data Layer (`data/`)
- **`synthetic_generator.py`**: Generates realistic demo data with 10,000 users and configurable drop-off rates. Also handles file upload validation, column mapping, and format detection for CSV, Excel, JSON, and Parquet files.
- **`db_manager.py`**: Manages DuckDB database operations including company CRUD, funnel event storage, SQL-based queries, and user authentication with bcrypt password hashing.

### Authentication System
- **Password Security**: bcrypt hashing with per-password salts
- **User Roles**:
  - `guest`: Can only view synthetic demo data (no login required)
  - `company`: Can only view their linked company's data
  - `admin`: Full access to all data, admin dashboard for managing companies and users
- **Session Management**: Streamlit session state tracks authentication and user info
- **Default Admin**: Created automatically on first run (username: `admin`, password: `admin123`) - change immediately after deployment

### ETL Pipeline (`etl/funnel_etl.py`)
- Transforms raw event data into funnel metrics
- Calculates conversion rates between stages (visit → signup → activation → purchase)
- Computes time-to-conversion statistics
- Supports cohort analysis, revenue metrics, and A/B comparison
- Uses DuckDB for SQL-based analytical queries

### Experiment Impact Simulator
Two layers — deterministic baseline (`utils/simulator.py`) and ML-powered simulator (`utils/ml_simulator.py`):

**Deterministic (`utils/simulator.py`)**:
- `compute_baseline_metrics()`, `simulate_funnel_impact()`, `compute_deltas()`, `generate_insight()`
- Applies flat percentage lifts to aggregate conversion rates

**ML-powered (`utils/ml_simulator.py`)**: actively used in the UI
- `train_funnel_models()`: Trains one **logistic regression** (scikit-learn) per funnel transition (Visit→Signup, Signup→Activation, Activation→Purchase) using one-hot encoded user features (traffic_source, device, country)
- `_apply_lift_to_probs()`: Applies lift via Bayesian odds-ratio update — `new_odds = old_odds * (1 + lift)` — guaranteed monotonic and clipped to [0,1]
- `simulate_with_models()`: Computes a per-stage multiplicative ratio (lifted population sum / baseline population sum) from the trained models and applies it to actual funnel counts
- `compute_ml_deltas()`, `generate_ml_insight()`: Diagnostics + plain-English insight including model AUC and top learned drivers
- Reports per-stage AUC, accuracy, sample sizes and a feature-importance heatmap (`create_feature_importance_chart` in `utils/plots.py`)
- Preset scenarios in sidebar: "Improve Signup UX (+15%)", "Improve Onboarding (+20%)", "Improve Checkout (+25%)", "All Stages (+10%)", or fully custom sliders

### Visualization Layer (`utils/plots.py`)
- Plotly-based interactive charts
- Funnel charts, heatmaps, histograms, and time-series visualizations
- Simulator charts: baseline vs simulated funnel comparison, incremental gains bar chart
- Consistent color scheme defined in module constants

### Data Flow
1. Data enters via synthetic generation, file upload, or database retrieval
2. ETL pipeline transforms events into user-level flags and aggregated metrics
3. Visualization layer renders interactive Plotly charts in Streamlit

## External Dependencies

### Database
- **DuckDB**: Local file-based analytical database (`funnel_data.duckdb`)
  - Stores company metadata and funnel events
  - Provides SQL-based analytical queries
  - No external server required

### Python Libraries
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation
- **Plotly**: Interactive visualizations
- **NumPy**: Numerical operations
- **DuckDB**: Embedded analytics database
- **bcrypt**: Secure password hashing

### Supported File Formats
- CSV, Excel (.xlsx, .xls), JSON, Parquet for data import
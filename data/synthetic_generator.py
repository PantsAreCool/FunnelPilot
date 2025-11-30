"""
Synthetic Data Generator for Marketing Funnel Analysis

Generates realistic event-level data with configurable drop-off rates
for testing and demonstration purposes.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import io
from typing import Union


def generate_synthetic_data(
    n_users: int = 10000,
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic marketing funnel event data.
    
    Args:
        n_users: Number of unique users to generate
        start_date: Start date for event timestamps
        end_date: End date for event timestamps
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with columns: user_id, event_name, event_timestamp, 
                               traffic_source, device, country, revenue
    """
    np.random.seed(seed)
    
    traffic_sources = ["organic", "paid_search", "social", "email", "referral", "direct"]
    traffic_weights = [0.25, 0.20, 0.18, 0.15, 0.12, 0.10]
    
    devices = ["desktop", "mobile", "tablet"]
    device_weights = [0.45, 0.45, 0.10]
    
    countries = ["USA", "UK", "Canada", "Germany", "France", "Australia", "Brazil", "India", "Japan", "Other"]
    country_weights = [0.30, 0.12, 0.10, 0.10, 0.08, 0.07, 0.06, 0.06, 0.05, 0.06]
    
    visit_to_signup = 0.45
    signup_to_activation = 0.55
    activation_to_purchase = 0.50
    
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    date_range_days = (end - start).days
    
    events = []
    
    for user_id in range(1, n_users + 1):
        user_traffic = np.random.choice(traffic_sources, p=traffic_weights)
        user_device = np.random.choice(devices, p=device_weights)
        user_country = np.random.choice(countries, p=country_weights)
        
        base_timestamp = start + timedelta(
            days=np.random.randint(0, max(1, date_range_days - 30)),
            hours=np.random.randint(0, 24),
            minutes=np.random.randint(0, 60)
        )
        
        events.append({
            "user_id": user_id,
            "event_name": "visit",
            "event_timestamp": base_timestamp,
            "traffic_source": user_traffic,
            "device": user_device,
            "country": user_country,
            "revenue": 0.0
        })
        
        conversion_boost = 1.0
        if user_traffic in ["email", "referral"]:
            conversion_boost = 1.15
        elif user_traffic == "paid_search":
            conversion_boost = 1.10
        
        if user_device == "desktop":
            conversion_boost *= 1.05
        
        if np.random.random() < min(visit_to_signup * conversion_boost, 0.65):
            signup_time = base_timestamp + timedelta(
                hours=np.random.randint(0, 48),
                minutes=np.random.randint(0, 60)
            )
            events.append({
                "user_id": user_id,
                "event_name": "signup",
                "event_timestamp": signup_time,
                "traffic_source": user_traffic,
                "device": user_device,
                "country": user_country,
                "revenue": 0.0
            })
            
            if np.random.random() < min(signup_to_activation * conversion_boost, 0.70):
                activation_time = signup_time + timedelta(
                    days=np.random.randint(0, 7),
                    hours=np.random.randint(0, 24),
                    minutes=np.random.randint(0, 60)
                )
                events.append({
                    "user_id": user_id,
                    "event_name": "activation",
                    "event_timestamp": activation_time,
                    "traffic_source": user_traffic,
                    "device": user_device,
                    "country": user_country,
                    "revenue": 0.0
                })
                
                if np.random.random() < min(activation_to_purchase * conversion_boost, 0.60):
                    purchase_time = activation_time + timedelta(
                        days=np.random.randint(0, 14),
                        hours=np.random.randint(0, 24),
                        minutes=np.random.randint(0, 60)
                    )
                    
                    base_revenue = np.random.lognormal(mean=3.5, sigma=0.8)
                    revenue = round(min(max(base_revenue, 9.99), 999.99), 2)
                    
                    events.append({
                        "user_id": user_id,
                        "event_name": "purchase",
                        "event_timestamp": purchase_time,
                        "traffic_source": user_traffic,
                        "device": user_device,
                        "country": user_country,
                        "revenue": revenue
                    })
    
    df = pd.DataFrame(events)
    df = df.sort_values("event_timestamp").reset_index(drop=True)
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
    
    return df


def save_synthetic_data(filepath: str = "data/events.csv", **kwargs) -> pd.DataFrame:
    """
    Generate and save synthetic data to CSV.
    
    Args:
        filepath: Path to save the CSV file
        **kwargs: Arguments passed to generate_synthetic_data
    
    Returns:
        Generated DataFrame
    """
    df = generate_synthetic_data(**kwargs)
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    
    return df


def load_or_generate_data(filepath: str = "data/events.csv", **kwargs) -> pd.DataFrame:
    """
    Load existing data or generate new synthetic data if file doesn't exist.
    
    Args:
        filepath: Path to the CSV file
        **kwargs: Arguments passed to generate_synthetic_data if generating
    
    Returns:
        DataFrame with event data
    """
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
        return df
    else:
        return save_synthetic_data(filepath, **kwargs)


def validate_uploaded_data(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Validate that uploaded data has required columns and correct formats.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    required_columns = ["user_id", "event_name", "event_timestamp"]
    optional_columns = ["traffic_source", "device", "country", "revenue"]
    
    errors = []
    
    missing_required = [col for col in required_columns if col not in df.columns]
    if missing_required:
        errors.append(f"Missing required columns: {', '.join(missing_required)}")
        return False, errors
    
    valid_events = {"visit", "signup", "activation", "purchase"}
    unique_events = set(df["event_name"].unique())
    invalid_events = unique_events - valid_events
    if invalid_events:
        errors.append(f"Invalid event names found: {', '.join(invalid_events)}. Valid events are: {', '.join(valid_events)}")
    
    if "visit" not in unique_events:
        errors.append("Data must contain at least 'visit' events")
    
    try:
        pd.to_datetime(df["event_timestamp"], format="mixed", dayfirst=False)
    except Exception:
        try:
            pd.to_datetime(df["event_timestamp"], infer_datetime_format=True)
        except Exception:
            test_parse = pd.to_datetime(df["event_timestamp"], errors="coerce")
            if test_parse.isna().all():
                errors.append("Could not parse event_timestamp column as datetime")
    
    if errors:
        return False, errors
    
    return True, []


def prepare_uploaded_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare uploaded data by adding missing optional columns with defaults.
    
    Args:
        df: Uploaded DataFrame
    
    Returns:
        DataFrame with all required columns
    """
    df = df.copy()
    
    try:
        df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], format="mixed", dayfirst=False)
    except Exception:
        try:
            df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], infer_datetime_format=True)
        except Exception:
            df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], errors="coerce")
    
    if "traffic_source" not in df.columns:
        df["traffic_source"] = "unknown"
    
    if "device" not in df.columns:
        df["device"] = "unknown"
    
    if "country" not in df.columns:
        df["country"] = "unknown"
    
    if "revenue" not in df.columns:
        df["revenue"] = 0.0
    else:
        df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce").fillna(0.0)
    
    return df


def read_uploaded_file(uploaded_file, file_type: str = None) -> tuple[pd.DataFrame, str]:
    """
    Read uploaded file in various formats (CSV, Excel, JSON, Parquet).
    
    Args:
        uploaded_file: Streamlit uploaded file object or file path
        file_type: Optional explicit file type. If None, inferred from filename.
    
    Returns:
        Tuple of (DataFrame, error message or empty string)
    """
    try:
        if hasattr(uploaded_file, 'name'):
            filename = uploaded_file.name.lower()
        else:
            filename = str(uploaded_file).lower()
        
        if file_type:
            ext = file_type.lower()
        elif filename.endswith('.csv'):
            ext = 'csv'
        elif filename.endswith('.xlsx') or filename.endswith('.xls'):
            ext = 'excel'
        elif filename.endswith('.json'):
            ext = 'json'
        elif filename.endswith('.parquet') or filename.endswith('.pq'):
            ext = 'parquet'
        else:
            return None, f"Unsupported file format. Please upload CSV, Excel (.xlsx/.xls), JSON, or Parquet files."
        
        if ext == 'csv':
            df = pd.read_csv(uploaded_file)
        elif ext == 'excel':
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        elif ext == 'json':
            if hasattr(uploaded_file, 'read'):
                content = uploaded_file.read()
                if isinstance(content, bytes):
                    content = content.decode('utf-8')
                df = pd.read_json(io.StringIO(content))
            else:
                df = pd.read_json(uploaded_file)
        elif ext == 'parquet':
            df = pd.read_parquet(uploaded_file)
        else:
            return None, f"Unsupported file format: {ext}"
        
        return df, ""
    
    except Exception as e:
        return None, f"Error reading file: {str(e)}"


def get_supported_formats() -> list[str]:
    """Return list of supported file formats for upload."""
    return ["csv", "xlsx", "xls", "json", "parquet"]


def get_file_format_help() -> str:
    """Return help text describing supported file formats."""
    return """
**Supported File Formats:**
- **CSV** (.csv): Comma-separated values
- **Excel** (.xlsx, .xls): Microsoft Excel spreadsheets
- **JSON** (.json): JavaScript Object Notation (records or array format)
- **Parquet** (.parquet, .pq): Apache Parquet columnar format

**Required Columns:**
- `user_id` - Unique user identifier
- `event_name` - One of: visit, signup, activation, purchase
- `event_timestamp` - Event datetime

**Optional Columns:**
- `traffic_source` - e.g., organic, paid_search, social
- `device` - e.g., desktop, mobile, tablet
- `country` - e.g., USA, UK, Germany
- `revenue` - Purchase revenue (numeric)
"""


def get_required_columns() -> list[str]:
    """Return list of required column names."""
    return ["user_id", "event_name", "event_timestamp"]


def get_optional_columns() -> list[str]:
    """Return list of optional column names."""
    return ["traffic_source", "device", "country", "revenue"]


def apply_column_mapping(df: pd.DataFrame, column_mapping: dict) -> pd.DataFrame:
    """
    Apply column name mapping to a DataFrame.
    
    Args:
        df: Source DataFrame
        column_mapping: Dictionary mapping target column names to source column names
                       e.g., {"user_id": "customer_id", "event_name": "action"}
    
    Returns:
        DataFrame with renamed columns
    """
    df = df.copy()
    
    reverse_mapping = {v: k for k, v in column_mapping.items() if v and v != "(none)"}
    
    if reverse_mapping:
        df = df.rename(columns=reverse_mapping)
    
    return df


def auto_detect_columns(df: pd.DataFrame) -> dict:
    """
    Attempt to automatically detect column mappings based on column names.
    
    Args:
        df: DataFrame to analyze
    
    Returns:
        Dictionary with suggested mappings
    """
    columns = [c.lower() for c in df.columns]
    original_columns = list(df.columns)
    
    mappings = {}
    
    user_id_keywords = ["user_id", "userid", "user", "customer_id", "customerid", "customer", "id"]
    for keyword in user_id_keywords:
        for i, col in enumerate(columns):
            if keyword in col or col == keyword:
                mappings["user_id"] = original_columns[i]
                break
        if "user_id" in mappings:
            break
    
    event_keywords = ["event_name", "eventname", "event", "action", "activity", "type"]
    for keyword in event_keywords:
        for i, col in enumerate(columns):
            if keyword in col or col == keyword:
                mappings["event_name"] = original_columns[i]
                break
        if "event_name" in mappings:
            break
    
    timestamp_keywords = ["event_timestamp", "timestamp", "time", "date", "datetime", "created_at", "created"]
    for keyword in timestamp_keywords:
        for i, col in enumerate(columns):
            if keyword in col or col == keyword:
                mappings["event_timestamp"] = original_columns[i]
                break
        if "event_timestamp" in mappings:
            break
    
    source_keywords = ["traffic_source", "source", "channel", "medium", "utm_source"]
    for keyword in source_keywords:
        for i, col in enumerate(columns):
            if keyword in col or col == keyword:
                mappings["traffic_source"] = original_columns[i]
                break
        if "traffic_source" in mappings:
            break
    
    device_keywords = ["device", "device_type", "platform"]
    for keyword in device_keywords:
        for i, col in enumerate(columns):
            if keyword in col or col == keyword:
                mappings["device"] = original_columns[i]
                break
        if "device" in mappings:
            break
    
    country_keywords = ["country", "region", "location", "geo"]
    for keyword in country_keywords:
        for i, col in enumerate(columns):
            if keyword in col or col == keyword:
                mappings["country"] = original_columns[i]
                break
        if "country" in mappings:
            break
    
    revenue_keywords = ["revenue", "amount", "value", "price", "total"]
    for keyword in revenue_keywords:
        for i, col in enumerate(columns):
            if keyword in col or col == keyword:
                mappings["revenue"] = original_columns[i]
                break
        if "revenue" in mappings:
            break
    
    return mappings

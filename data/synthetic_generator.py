"""
Synthetic Data Generator for Marketing Funnel Analysis

Generates realistic event-level data with configurable drop-off rates
for testing and demonstration purposes.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


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
        df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
    except Exception:
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
    
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
    
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

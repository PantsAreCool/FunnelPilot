"""
ETL Pipeline for Marketing Funnel Analysis

Transforms raw event data into funnel metrics, conversion rates,
and time-to-conversion calculations.
"""

import pandas as pd
import numpy as np
import duckdb
from typing import Optional


FUNNEL_STAGES = ["visit", "signup", "activation", "purchase"]
STAGE_ORDER = {stage: i for i, stage in enumerate(FUNNEL_STAGES)}


def create_user_stage_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create per-user flags indicating which stages they reached.
    
    Args:
        df: Event-level DataFrame
    
    Returns:
        DataFrame with one row per user and boolean flags for each stage
    """
    user_events = df.groupby("user_id")["event_name"].apply(set).reset_index()
    user_events.columns = ["user_id", "events"]
    
    user_events["visited"] = user_events["events"].apply(lambda x: "visit" in x)
    user_events["signed_up"] = user_events["events"].apply(lambda x: "signup" in x)
    user_events["activated"] = user_events["events"].apply(lambda x: "activation" in x)
    user_events["purchased"] = user_events["events"].apply(lambda x: "purchase" in x)
    
    user_attrs = df.groupby("user_id").agg({
        "traffic_source": "first",
        "device": "first",
        "country": "first",
        "revenue": "sum"
    }).reset_index()
    
    result = user_events.merge(user_attrs, on="user_id")
    result = result.drop(columns=["events"])
    
    return result


def calculate_funnel_counts(user_flags: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate counts at each funnel stage.
    
    Args:
        user_flags: DataFrame with per-user stage flags
    
    Returns:
        DataFrame with stage names and counts
    """
    counts = {
        "visit": user_flags["visited"].sum(),
        "signup": user_flags["signed_up"].sum(),
        "activation": user_flags["activated"].sum(),
        "purchase": user_flags["purchased"].sum()
    }
    
    result = pd.DataFrame([
        {"stage": stage, "count": counts[stage], "order": STAGE_ORDER[stage]}
        for stage in FUNNEL_STAGES
    ])
    
    return result.sort_values("order").reset_index(drop=True)


def calculate_conversion_rates(funnel_counts: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate step-by-step and overall conversion rates.
    
    Args:
        funnel_counts: DataFrame with stage counts
    
    Returns:
        DataFrame with conversion metrics
    """
    counts = funnel_counts.sort_values("order")
    
    result = counts.copy()
    result["previous_count"] = result["count"].shift(1)
    
    result["step_conversion_rate"] = result.apply(
        lambda row: round((row["count"] / row["previous_count"] * 100), 2) 
        if row["previous_count"] and row["previous_count"] > 0 else 0.0, 
        axis=1
    )
    result.loc[result["order"] == 0, "step_conversion_rate"] = 100.0
    
    visit_rows = result.loc[result["order"] == 0, "count"]
    total_visitors = visit_rows.values[0] if len(visit_rows) > 0 else 0
    
    result["overall_conversion_rate"] = result.apply(
        lambda row: round((row["count"] / total_visitors * 100), 2) 
        if total_visitors > 0 else 0.0, 
        axis=1
    )
    
    result["dropoff_count"] = result.apply(
        lambda row: row["previous_count"] - row["count"] 
        if pd.notna(row["previous_count"]) else 0, 
        axis=1
    )
    result.loc[result["order"] == 0, "dropoff_count"] = 0
    
    result["dropoff_rate"] = (100 - result["step_conversion_rate"]).round(2)
    result.loc[result["order"] == 0, "dropoff_rate"] = 0.0
    
    return result.drop(columns=["previous_count"])


def calculate_time_to_conversion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate time between funnel stages for each user.
    
    Args:
        df: Event-level DataFrame
    
    Returns:
        DataFrame with time-to-conversion metrics per user
    """
    pivot = df.pivot_table(
        index="user_id",
        columns="event_name",
        values="event_timestamp",
        aggfunc="min"
    ).reset_index()
    
    result = pd.DataFrame({"user_id": pivot["user_id"]})
    
    if "visit" in pivot.columns and "signup" in pivot.columns:
        result["time_visit_to_signup"] = (
            pivot["signup"] - pivot["visit"]
        ).dt.total_seconds() / 3600
    
    if "signup" in pivot.columns and "activation" in pivot.columns:
        result["time_signup_to_activation"] = (
            pivot["activation"] - pivot["signup"]
        ).dt.total_seconds() / 3600
    
    if "activation" in pivot.columns and "purchase" in pivot.columns:
        result["time_activation_to_purchase"] = (
            pivot["purchase"] - pivot["activation"]
        ).dt.total_seconds() / 3600
    
    if "visit" in pivot.columns and "purchase" in pivot.columns:
        result["time_visit_to_purchase"] = (
            pivot["purchase"] - pivot["visit"]
        ).dt.total_seconds() / 3600
    
    return result


def calculate_breakdown_metrics(
    user_flags: pd.DataFrame,
    group_by: str
) -> pd.DataFrame:
    """
    Calculate funnel metrics broken down by a dimension.
    
    Args:
        user_flags: DataFrame with per-user stage flags
        group_by: Column to group by (traffic_source, device, country)
    
    Returns:
        DataFrame with metrics per group
    """
    grouped = user_flags.groupby(group_by).agg({
        "visited": "sum",
        "signed_up": "sum",
        "activated": "sum",
        "purchased": "sum",
        "revenue": "sum",
        "user_id": "count"
    }).reset_index()
    
    grouped.columns = [group_by, "visits", "signups", "activations", "purchases", "revenue", "users"]
    
    grouped["visit_to_signup_rate"] = (grouped["signups"] / grouped["visits"] * 100).round(2)
    grouped["signup_to_activation_rate"] = (grouped["activations"] / grouped["signups"] * 100).round(2)
    grouped["activation_to_purchase_rate"] = (grouped["purchases"] / grouped["activations"] * 100).round(2)
    grouped["overall_conversion_rate"] = (grouped["purchases"] / grouped["visits"] * 100).round(2)
    
    grouped = grouped.fillna(0)
    
    return grouped.sort_values("visits", ascending=False)


def run_funnel_analysis_sql(df: pd.DataFrame) -> dict:
    """
    Run comprehensive funnel analysis using DuckDB SQL.
    
    Args:
        df: Event-level DataFrame
    
    Returns:
        Dictionary with analysis results
    """
    conn = duckdb.connect(":memory:")
    conn.register("events", df)
    
    funnel_query = """
    WITH user_stages AS (
        SELECT 
            user_id,
            MAX(CASE WHEN event_name = 'visit' THEN 1 ELSE 0 END) as visited,
            MAX(CASE WHEN event_name = 'signup' THEN 1 ELSE 0 END) as signed_up,
            MAX(CASE WHEN event_name = 'activation' THEN 1 ELSE 0 END) as activated,
            MAX(CASE WHEN event_name = 'purchase' THEN 1 ELSE 0 END) as purchased,
            MAX(traffic_source) as traffic_source,
            MAX(device) as device,
            MAX(country) as country,
            SUM(revenue) as total_revenue
        FROM events
        GROUP BY user_id
    )
    SELECT 
        SUM(visited) as visit_count,
        SUM(signed_up) as signup_count,
        SUM(activated) as activation_count,
        SUM(purchased) as purchase_count,
        SUM(total_revenue) as total_revenue,
        ROUND(SUM(signed_up) * 100.0 / NULLIF(SUM(visited), 0), 2) as visit_to_signup_rate,
        ROUND(SUM(activated) * 100.0 / NULLIF(SUM(signed_up), 0), 2) as signup_to_activation_rate,
        ROUND(SUM(purchased) * 100.0 / NULLIF(SUM(activated), 0), 2) as activation_to_purchase_rate,
        ROUND(SUM(purchased) * 100.0 / NULLIF(SUM(visited), 0), 2) as overall_conversion_rate
    FROM user_stages
    """
    
    summary = conn.execute(funnel_query).fetchdf()
    
    conn.close()
    
    return summary.to_dict(orient="records")[0]


def filter_events(
    df: pd.DataFrame,
    traffic_sources: Optional[list] = None,
    devices: Optional[list] = None,
    countries: Optional[list] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Filter events based on various criteria.
    
    Args:
        df: Event-level DataFrame
        traffic_sources: List of traffic sources to include
        devices: List of devices to include
        countries: List of countries to include
        start_date: Start date filter
        end_date: End date filter
    
    Returns:
        Filtered DataFrame
    """
    filtered = df.copy()
    
    if traffic_sources and len(traffic_sources) > 0:
        filtered = filtered[filtered["traffic_source"].isin(traffic_sources)]
    
    if devices and len(devices) > 0:
        filtered = filtered[filtered["device"].isin(devices)]
    
    if countries and len(countries) > 0:
        filtered = filtered[filtered["country"].isin(countries)]
    
    if start_date:
        filtered = filtered[filtered["event_timestamp"] >= pd.to_datetime(start_date)]
    
    if end_date:
        filtered = filtered[filtered["event_timestamp"] <= pd.to_datetime(end_date)]
    
    return filtered


def get_time_to_conversion_stats(time_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate summary statistics for time-to-conversion metrics.
    
    Args:
        time_df: DataFrame with time-to-conversion columns
    
    Returns:
        DataFrame with summary statistics
    """
    time_cols = [col for col in time_df.columns if col.startswith("time_")]
    
    stats = []
    for col in time_cols:
        data = time_df[col].dropna()
        if len(data) > 0:
            stats.append({
                "metric": col.replace("time_", "").replace("_", " → ").title(),
                "count": len(data),
                "mean_hours": round(data.mean(), 2),
                "median_hours": round(data.median(), 2),
                "std_hours": round(data.std(), 2),
                "min_hours": round(data.min(), 2),
                "max_hours": round(data.max(), 2)
            })
    
    return pd.DataFrame(stats)

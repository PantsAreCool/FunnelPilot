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
        start_dt = pd.to_datetime(start_date)
        filtered = filtered[filtered["event_timestamp"] >= start_dt]
    
    if end_date:
        end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        filtered = filtered[filtered["event_timestamp"] <= end_dt]
    
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


def calculate_cohort_analysis(df: pd.DataFrame, cohort_period: str = "week") -> pd.DataFrame:
    """
    Calculate funnel metrics by cohort (based on first visit date).
    
    Args:
        df: Event-level DataFrame
        cohort_period: 'week' or 'month' for cohort grouping
    
    Returns:
        DataFrame with funnel metrics per cohort
    """
    first_visits = df[df["event_name"] == "visit"].groupby("user_id")["event_timestamp"].min().reset_index()
    first_visits.columns = ["user_id", "first_visit"]
    
    if cohort_period == "week":
        first_visits["cohort"] = first_visits["first_visit"].dt.to_period("W").dt.start_time
    else:
        first_visits["cohort"] = first_visits["first_visit"].dt.to_period("M").dt.start_time
    
    user_events = df.groupby("user_id")["event_name"].apply(set).reset_index()
    user_events.columns = ["user_id", "events"]
    
    user_events["visited"] = user_events["events"].apply(lambda x: "visit" in x)
    user_events["signed_up"] = user_events["events"].apply(lambda x: "signup" in x)
    user_events["activated"] = user_events["events"].apply(lambda x: "activation" in x)
    user_events["purchased"] = user_events["events"].apply(lambda x: "purchase" in x)
    
    revenue = df[df["event_name"] == "purchase"].groupby("user_id")["revenue"].sum().reset_index()
    
    cohort_users = first_visits.merge(user_events, on="user_id")
    cohort_users = cohort_users.merge(revenue, on="user_id", how="left")
    cohort_users["revenue"] = cohort_users["revenue"].fillna(0)
    
    cohort_metrics = cohort_users.groupby("cohort").agg({
        "visited": "sum",
        "signed_up": "sum",
        "activated": "sum",
        "purchased": "sum",
        "revenue": "sum",
        "user_id": "count"
    }).reset_index()
    
    cohort_metrics.columns = ["cohort", "visits", "signups", "activations", "purchases", "revenue", "users"]
    
    cohort_metrics["visit_to_signup_rate"] = (cohort_metrics["signups"] / cohort_metrics["visits"] * 100).round(2).fillna(0)
    cohort_metrics["signup_to_activation_rate"] = (cohort_metrics["activations"] / cohort_metrics["signups"] * 100).round(2).fillna(0)
    cohort_metrics["activation_to_purchase_rate"] = (cohort_metrics["purchases"] / cohort_metrics["activations"] * 100).round(2).fillna(0)
    cohort_metrics["overall_conversion_rate"] = (cohort_metrics["purchases"] / cohort_metrics["visits"] * 100).round(2).fillna(0)
    
    cohort_metrics = cohort_metrics.sort_values("cohort")
    
    return cohort_metrics


def calculate_revenue_metrics(user_flags: pd.DataFrame) -> dict:
    """
    Calculate revenue-related metrics: LTV, ARPU, revenue by segment.
    
    Args:
        user_flags: DataFrame with per-user stage flags and revenue
    
    Returns:
        Dictionary with revenue metrics
    """
    total_users = len(user_flags)
    paying_users = user_flags[user_flags["purchased"] == True]
    total_revenue = user_flags["revenue"].sum()
    
    arpu = total_revenue / total_users if total_users > 0 else 0
    arppu = total_revenue / len(paying_users) if len(paying_users) > 0 else 0
    
    ltv_by_source = user_flags.groupby("traffic_source").agg({
        "revenue": ["sum", "mean"],
        "user_id": "count",
        "purchased": "sum"
    }).reset_index()
    ltv_by_source.columns = ["traffic_source", "total_revenue", "avg_revenue", "users", "purchasers"]
    ltv_by_source["ltv"] = (ltv_by_source["total_revenue"] / ltv_by_source["users"]).round(2)
    ltv_by_source["conversion_rate"] = (ltv_by_source["purchasers"] / ltv_by_source["users"] * 100).round(2)
    
    ltv_by_device = user_flags.groupby("device").agg({
        "revenue": ["sum", "mean"],
        "user_id": "count",
        "purchased": "sum"
    }).reset_index()
    ltv_by_device.columns = ["device", "total_revenue", "avg_revenue", "users", "purchasers"]
    ltv_by_device["ltv"] = (ltv_by_device["total_revenue"] / ltv_by_device["users"]).round(2)
    ltv_by_device["conversion_rate"] = (ltv_by_device["purchasers"] / ltv_by_device["users"] * 100).round(2)
    
    ltv_by_country = user_flags.groupby("country").agg({
        "revenue": ["sum", "mean"],
        "user_id": "count",
        "purchased": "sum"
    }).reset_index()
    ltv_by_country.columns = ["country", "total_revenue", "avg_revenue", "users", "purchasers"]
    ltv_by_country["ltv"] = (ltv_by_country["total_revenue"] / ltv_by_country["users"]).round(2)
    ltv_by_country["conversion_rate"] = (ltv_by_country["purchasers"] / ltv_by_country["users"] * 100).round(2)
    
    revenue_distribution = paying_users["revenue"].describe().to_dict() if len(paying_users) > 0 else {}
    
    return {
        "total_revenue": round(total_revenue, 2),
        "arpu": round(arpu, 2),
        "arppu": round(arppu, 2),
        "paying_users": len(paying_users),
        "total_users": total_users,
        "conversion_to_paid": round(len(paying_users) / total_users * 100, 2) if total_users > 0 else 0,
        "ltv_by_source": ltv_by_source.sort_values("ltv", ascending=False),
        "ltv_by_device": ltv_by_device.sort_values("ltv", ascending=False),
        "ltv_by_country": ltv_by_country.sort_values("ltv", ascending=False),
        "revenue_distribution": revenue_distribution
    }


def get_user_journeys(df: pd.DataFrame, limit: int = 100) -> pd.DataFrame:
    """
    Get individual user journey paths through the funnel.
    
    Args:
        df: Event-level DataFrame
        limit: Maximum number of users to return
    
    Returns:
        DataFrame with user journey details
    """
    user_journeys = df.sort_values(["user_id", "event_timestamp"])
    
    journey_summary = user_journeys.groupby("user_id").agg({
        "event_name": lambda x: " → ".join(x),
        "event_timestamp": ["min", "max", "count"],
        "traffic_source": "first",
        "device": "first",
        "country": "first",
        "revenue": "sum"
    }).reset_index()
    
    journey_summary.columns = ["user_id", "journey_path", "first_event", "last_event", "event_count", 
                               "traffic_source", "device", "country", "revenue"]
    
    journey_summary["journey_duration_hours"] = (
        (journey_summary["last_event"] - journey_summary["first_event"]).dt.total_seconds() / 3600
    ).round(2)
    
    events_set = df.groupby("user_id")["event_name"].apply(set).reset_index()
    events_set.columns = ["user_id", "events"]
    
    journey_summary = journey_summary.merge(events_set, on="user_id")
    journey_summary["final_stage"] = journey_summary["events"].apply(
        lambda x: "purchase" if "purchase" in x else 
                  "activation" if "activation" in x else 
                  "signup" if "signup" in x else "visit"
    )
    journey_summary = journey_summary.drop(columns=["events"])
    
    journey_summary = journey_summary.sort_values("revenue", ascending=False).head(limit)
    
    return journey_summary


def calculate_ab_comparison(
    df: pd.DataFrame,
    segment_column: str,
    segment_a: str,
    segment_b: str
) -> dict:
    """
    Calculate A/B test comparison between two segments.
    
    Args:
        df: Event-level DataFrame
        segment_column: Column to segment by (e.g., 'traffic_source', 'device')
        segment_a: Value for segment A
        segment_b: Value for segment B
    
    Returns:
        Dictionary with comparison metrics
    """
    df_a = df[df[segment_column] == segment_a]
    df_b = df[df[segment_column] == segment_b]
    
    if len(df_a) == 0 or len(df_b) == 0:
        return {
            "error": "One or both segments have no data",
            "segment_a_count": len(df_a),
            "segment_b_count": len(df_b)
        }
    
    flags_a = create_user_stage_flags(df_a)
    flags_b = create_user_stage_flags(df_b)
    
    funnel_a = calculate_funnel_counts(flags_a)
    funnel_b = calculate_funnel_counts(flags_b)
    
    rates_a = calculate_conversion_rates(funnel_a)
    rates_b = calculate_conversion_rates(funnel_b)
    
    comparison_df = pd.DataFrame({
        "stage": rates_a["stage"],
        f"{segment_a}_count": funnel_a["count"],
        f"{segment_b}_count": funnel_b["count"],
        f"{segment_a}_rate": rates_a["step_conversion_rate"],
        f"{segment_b}_rate": rates_b["step_conversion_rate"],
        "rate_diff": rates_a["step_conversion_rate"] - rates_b["step_conversion_rate"],
        "rate_lift": ((rates_a["step_conversion_rate"] - rates_b["step_conversion_rate"]) / 
                      rates_b["step_conversion_rate"].replace(0, np.nan) * 100).round(2)
    })
    
    users_a = len(flags_a)
    users_b = len(flags_b)
    purchases_a = flags_a["purchased"].sum()
    purchases_b = flags_b["purchased"].sum()
    
    overall_conv_a = (purchases_a / users_a * 100) if users_a > 0 else 0
    overall_conv_b = (purchases_b / users_b * 100) if users_b > 0 else 0
    
    revenue_a = flags_a["revenue"].sum()
    revenue_b = flags_b["revenue"].sum()
    arpu_a = revenue_a / users_a if users_a > 0 else 0
    arpu_b = revenue_b / users_b if users_b > 0 else 0
    
    summary = {
        "segment_a": {
            "name": segment_a,
            "users": users_a,
            "purchases": int(purchases_a),
            "conversion_rate": round(overall_conv_a, 2),
            "revenue": round(revenue_a, 2),
            "arpu": round(arpu_a, 2)
        },
        "segment_b": {
            "name": segment_b,
            "users": users_b,
            "purchases": int(purchases_b),
            "conversion_rate": round(overall_conv_b, 2),
            "revenue": round(revenue_b, 2),
            "arpu": round(arpu_b, 2)
        },
        "comparison": {
            "conversion_diff": round(overall_conv_a - overall_conv_b, 2),
            "conversion_lift": round((overall_conv_a - overall_conv_b) / overall_conv_b * 100, 2) if overall_conv_b > 0 else 0,
            "arpu_diff": round(arpu_a - arpu_b, 2),
            "arpu_lift": round((arpu_a - arpu_b) / arpu_b * 100, 2) if arpu_b > 0 else 0,
            "winner": segment_a if overall_conv_a > overall_conv_b else segment_b if overall_conv_b > overall_conv_a else "tie"
        }
    }
    
    return {
        "stage_comparison": comparison_df,
        "summary": summary,
        "funnel_a": funnel_a,
        "funnel_b": funnel_b,
        "rates_a": rates_a,
        "rates_b": rates_b
    }


def get_segment_options(df: pd.DataFrame, segment_column: str) -> list[str]:
    """
    Get unique values for a segment column.
    
    Args:
        df: Event-level DataFrame
        segment_column: Column to get unique values from
    
    Returns:
        List of unique values
    """
    return sorted(df[segment_column].unique().tolist())

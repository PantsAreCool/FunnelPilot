"""
DuckDB Database Manager for Multi-Company Funnel Data Storage

Provides persistent storage for marketing funnel data from multiple companies.
"""

import duckdb
import pandas as pd
from datetime import datetime
from typing import Optional, List, Tuple
import os


DB_PATH = "funnel_data.duckdb"

_db_initialized = False


def get_connection() -> duckdb.DuckDBPyConnection:
    """Get a connection to the DuckDB database."""
    return duckdb.connect(DB_PATH)


def init_database():
    """Initialize the database with required tables."""
    global _db_initialized
    
    if _db_initialized:
        return
    
    conn = get_connection()
    
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS companies (
                company_id INTEGER PRIMARY KEY,
                company_name VARCHAR UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS company_id_seq START 1
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS funnel_events (
                event_id INTEGER PRIMARY KEY,
                company_id INTEGER NOT NULL,
                user_id VARCHAR NOT NULL,
                event_name VARCHAR NOT NULL,
                event_timestamp TIMESTAMP NOT NULL,
                traffic_source VARCHAR,
                device VARCHAR,
                country VARCHAR,
                revenue DOUBLE DEFAULT 0.0,
                FOREIGN KEY (company_id) REFERENCES companies(company_id)
            )
        """)
        
        conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS event_id_seq START 1
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_company ON funnel_events(company_id)
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_user ON funnel_events(company_id, user_id)
        """)
        
        _db_initialized = True
    except Exception:
        pass
    finally:
        conn.close()


def get_all_companies() -> pd.DataFrame:
    """Get list of all stored companies with stats."""
    init_database()
    conn = get_connection()
    
    result = conn.execute("""
        SELECT 
            c.company_id,
            c.company_name,
            c.created_at,
            c.updated_at,
            COUNT(DISTINCT e.user_id) as user_count,
            COUNT(e.event_id) as event_count
        FROM companies c
        LEFT JOIN funnel_events e ON c.company_id = e.company_id
        GROUP BY c.company_id, c.company_name, c.created_at, c.updated_at
        ORDER BY c.company_name
    """).fetchdf()
    
    conn.close()
    return result


def get_company_names() -> List[str]:
    """Get list of all company names."""
    companies = get_all_companies()
    return companies["company_name"].tolist() if len(companies) > 0 else []


def company_exists(company_name: str) -> bool:
    """Check if a company already exists."""
    init_database()
    conn = get_connection()
    
    result = conn.execute(
        "SELECT COUNT(*) FROM companies WHERE company_name = ?",
        [company_name]
    ).fetchone()[0]
    
    conn.close()
    return result > 0


def create_company(company_name: str) -> int:
    """Create a new company and return its ID."""
    init_database()
    conn = get_connection()
    
    conn.execute(
        "INSERT INTO companies (company_id, company_name) VALUES (nextval('company_id_seq'), ?)",
        [company_name]
    )
    
    company_id = conn.execute(
        "SELECT company_id FROM companies WHERE company_name = ?",
        [company_name]
    ).fetchone()[0]
    
    conn.close()
    return company_id


def get_company_id(company_name: str) -> Optional[int]:
    """Get company ID by name."""
    init_database()
    conn = get_connection()
    
    result = conn.execute(
        "SELECT company_id FROM companies WHERE company_name = ?",
        [company_name]
    ).fetchone()
    
    conn.close()
    return result[0] if result else None


def save_company_data(company_name: str, df: pd.DataFrame, replace: bool = True) -> Tuple[bool, str]:
    """
    Save funnel data for a company.
    
    Args:
        company_name: Name of the company
        df: DataFrame with funnel events
        replace: If True, replace existing data; if False, append
    
    Returns:
        Tuple of (success, message)
    """
    if df is None or len(df) == 0:
        return False, "Cannot save empty data. Please upload a file with valid data."
    
    init_database()
    conn = get_connection()
    
    try:
        events_df = df.copy()
        
        required_cols = ["user_id", "event_name", "event_timestamp"]
        for col in required_cols:
            if col not in events_df.columns:
                conn.close()
                return False, f"Missing required column: {col}"
        
        try:
            events_df["event_timestamp"] = pd.to_datetime(events_df["event_timestamp"], errors="raise")
        except Exception as e:
            conn.close()
            return False, f"Invalid timestamp format: {str(e)}"
        
        events_df["user_id"] = events_df["user_id"].astype(str)
        events_df["event_name"] = events_df["event_name"].astype(str).str.lower()
        
        valid_events = ["visit", "signup", "activation", "purchase"]
        invalid_events = events_df[~events_df["event_name"].isin(valid_events)]["event_name"].unique()
        if len(invalid_events) > 0:
            conn.close()
            return False, f"Invalid event names: {', '.join(invalid_events[:5])}. Must be: {', '.join(valid_events)}"
        
        if company_exists(company_name):
            company_id = get_company_id(company_name)
            if replace:
                conn.execute(
                    "DELETE FROM funnel_events WHERE company_id = ?",
                    [company_id]
                )
                conn.execute(
                    "UPDATE companies SET updated_at = CURRENT_TIMESTAMP WHERE company_id = ?",
                    [company_id]
                )
        else:
            company_id = create_company(company_name)
        
        events_df["company_id"] = company_id
        
        if "traffic_source" not in events_df.columns:
            events_df["traffic_source"] = "unknown"
        else:
            events_df["traffic_source"] = events_df["traffic_source"].fillna("unknown").astype(str)
        
        if "device" not in events_df.columns:
            events_df["device"] = "unknown"
        else:
            events_df["device"] = events_df["device"].fillna("unknown").astype(str)
        
        if "country" not in events_df.columns:
            events_df["country"] = "unknown"
        else:
            events_df["country"] = events_df["country"].fillna("unknown").astype(str)
        
        if "revenue" not in events_df.columns:
            events_df["revenue"] = 0.0
        else:
            events_df["revenue"] = pd.to_numeric(events_df["revenue"], errors="coerce").fillna(0.0)
        
        insert_df = events_df[["company_id", "user_id", "event_name", "event_timestamp", 
                               "traffic_source", "device", "country", "revenue"]].copy()
        
        conn.register("insert_data", insert_df)
        conn.execute("""
            INSERT INTO funnel_events 
            (event_id, company_id, user_id, event_name, event_timestamp, 
             traffic_source, device, country, revenue)
            SELECT 
                nextval('event_id_seq'),
                company_id, user_id, event_name, event_timestamp,
                traffic_source, device, country, revenue
            FROM insert_data
        """)
        
        event_count = len(insert_df)
        user_count = insert_df["user_id"].nunique()
        
        conn.close()
        return True, f"Saved {event_count:,} events from {user_count:,} users for '{company_name}'"
        
    except Exception as e:
        conn.close()
        return False, f"Error saving data: {str(e)}"


def load_company_data(company_name: str) -> Optional[pd.DataFrame]:
    """Load funnel data for a specific company."""
    init_database()
    conn = get_connection()
    
    company_id = get_company_id(company_name)
    if company_id is None:
        conn.close()
        return None
    
    result = conn.execute("""
        SELECT 
            user_id,
            event_name,
            event_timestamp,
            traffic_source,
            device,
            country,
            revenue
        FROM funnel_events
        WHERE company_id = ?
        ORDER BY user_id, event_timestamp
    """, [company_id]).fetchdf()
    
    conn.close()
    
    if len(result) > 0:
        result["event_timestamp"] = pd.to_datetime(result["event_timestamp"])
    
    return result


def delete_company(company_name: str) -> Tuple[bool, str]:
    """Delete a company and all its data."""
    init_database()
    conn = get_connection()
    
    try:
        company_id = get_company_id(company_name)
        if company_id is None:
            conn.close()
            return False, f"Company '{company_name}' not found"
        
        conn.execute("DELETE FROM funnel_events WHERE company_id = ?", [company_id])
        conn.execute("DELETE FROM companies WHERE company_id = ?", [company_id])
        
        conn.close()
        return True, f"Deleted company '{company_name}' and all associated data"
        
    except Exception as e:
        conn.close()
        return False, f"Error deleting company: {str(e)}"


def run_funnel_analysis_sql(company_name: str) -> Optional[dict]:
    """
    Run comprehensive funnel analysis using DuckDB SQL for a specific company.
    
    Args:
        company_name: Name of the company to analyze
    
    Returns:
        Dictionary with analysis results or None if company not found
    """
    init_database()
    conn = get_connection()
    
    company_id = get_company_id(company_name)
    if company_id is None:
        conn.close()
        return None
    
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
        FROM funnel_events
        WHERE company_id = ?
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
    
    summary = conn.execute(funnel_query, [company_id]).fetchdf()
    conn.close()
    
    if len(summary) > 0:
        return summary.to_dict(orient="records")[0]
    return None


def get_database_stats() -> dict:
    """Get overall database statistics."""
    init_database()
    conn = get_connection()
    
    stats = {}
    
    stats["total_companies"] = conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0]
    stats["total_events"] = conn.execute("SELECT COUNT(*) FROM funnel_events").fetchone()[0]
    stats["total_users"] = conn.execute("SELECT COUNT(DISTINCT user_id) FROM funnel_events").fetchone()[0]
    
    if os.path.exists(DB_PATH):
        stats["db_size_mb"] = round(os.path.getsize(DB_PATH) / (1024 * 1024), 2)
    else:
        stats["db_size_mb"] = 0
    
    conn.close()
    return stats

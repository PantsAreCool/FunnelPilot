"""
Marketing Funnel Analysis Application

A comprehensive Streamlit dashboard for analyzing marketing funnel metrics
with support for both synthetic and user-uploaded data.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

from data.synthetic_generator import (
    generate_synthetic_data,
    validate_uploaded_data,
    prepare_uploaded_data,
    load_or_generate_data
)
from etl.funnel_etl import (
    create_user_stage_flags,
    calculate_funnel_counts,
    calculate_conversion_rates,
    calculate_time_to_conversion,
    calculate_breakdown_metrics,
    filter_events,
    get_time_to_conversion_stats,
    run_funnel_analysis_sql,
    calculate_cohort_analysis,
    calculate_revenue_metrics,
    get_user_journeys
)
from utils.plots import (
    create_funnel_chart,
    create_conversion_rate_chart,
    create_dropoff_chart,
    create_breakdown_bar_chart,
    create_time_distribution_chart,
    create_multi_metric_breakdown,
    create_cohort_heatmap,
    create_cohort_trend_chart,
    create_revenue_bar_chart,
    create_revenue_distribution_chart
)


st.set_page_config(
    page_title="Marketing Funnel Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E5E7EB;
    }
    .stMetric {
        background-color: #F9FAFB;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #E5E7EB;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_synthetic_data():
    """Load or generate synthetic data with caching."""
    return generate_synthetic_data(n_users=10000)


@st.cache_data
def process_funnel_data(df: pd.DataFrame):
    """Process data through the funnel ETL pipeline with caching."""
    user_flags = create_user_stage_flags(df)
    funnel_counts = calculate_funnel_counts(user_flags)
    conversion_rates = calculate_conversion_rates(funnel_counts)
    time_metrics = calculate_time_to_conversion(df)
    
    return {
        "user_flags": user_flags,
        "funnel_counts": funnel_counts,
        "conversion_rates": conversion_rates,
        "time_metrics": time_metrics
    }


@st.cache_data
def get_breakdown_data(user_flags: pd.DataFrame):
    """Calculate breakdown metrics with caching."""
    return {
        "traffic_source": calculate_breakdown_metrics(user_flags, "traffic_source"),
        "device": calculate_breakdown_metrics(user_flags, "device"),
        "country": calculate_breakdown_metrics(user_flags, "country")
    }


def render_sidebar(df: pd.DataFrame):
    """Render sidebar with filters and data source options."""
    st.sidebar.markdown("## Data Source")
    
    data_source = st.sidebar.radio(
        "Choose data source:",
        ["Demo Data (Synthetic)", "Upload Your Data"],
        help="Use demo data to explore the app, or upload your own marketing event data"
    )
    
    uploaded_df = None
    if data_source == "Upload Your Data":
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Upload CSV File")
        
        with st.sidebar.expander("Required CSV Format", expanded=False):
            st.markdown("""
            **Required columns:**
            - `user_id` - Unique user identifier
            - `event_name` - One of: visit, signup, activation, purchase
            - `event_timestamp` - Event datetime
            
            **Optional columns:**
            - `traffic_source` - e.g., organic, paid_search
            - `device` - e.g., desktop, mobile, tablet
            - `country` - e.g., USA, UK, Germany
            - `revenue` - Purchase revenue (numeric)
            """)
        
        uploaded_file = st.sidebar.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload your marketing event data in CSV format"
        )
        
        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                is_valid, errors = validate_uploaded_data(uploaded_df)
                
                if is_valid:
                    uploaded_df = prepare_uploaded_data(uploaded_df)
                    st.sidebar.success(f"Loaded {len(uploaded_df):,} events from {uploaded_df['user_id'].nunique():,} users")
                else:
                    st.sidebar.error("Data validation failed:")
                    for error in errors:
                        st.sidebar.error(f"• {error}")
                    uploaded_df = None
            except Exception as e:
                st.sidebar.error(f"Error reading file: {str(e)}")
                uploaded_df = None
    
    active_df = uploaded_df if uploaded_df is not None else df
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("## Filters")
    
    traffic_sources = sorted(active_df["traffic_source"].unique().tolist())
    selected_sources = st.sidebar.multiselect(
        "Traffic Source",
        options=traffic_sources,
        default=[],
        help="Filter by traffic source (leave empty for all)"
    )
    
    devices = sorted(active_df["device"].unique().tolist())
    selected_devices = st.sidebar.multiselect(
        "Device",
        options=devices,
        default=[],
        help="Filter by device type (leave empty for all)"
    )
    
    countries = sorted(active_df["country"].unique().tolist())
    selected_countries = st.sidebar.multiselect(
        "Country",
        options=countries,
        default=[],
        help="Filter by country (leave empty for all)"
    )
    
    min_date = active_df["event_timestamp"].min().date()
    max_date = active_df["event_timestamp"].max().date()
    
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        help="Filter events by date range"
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date
    
    return {
        "data_source": data_source,
        "uploaded_df": uploaded_df,
        "traffic_sources": selected_sources if selected_sources else None,
        "devices": selected_devices if selected_devices else None,
        "countries": selected_countries if selected_countries else None,
        "start_date": str(start_date),
        "end_date": str(end_date)
    }


def render_kpis(conversion_rates: pd.DataFrame, user_flags: pd.DataFrame):
    """Render top-line KPI metrics."""
    st.markdown('<div class="section-header">Key Performance Indicators</div>', unsafe_allow_html=True)
    
    def get_stage_count(stage_name):
        stage_data = conversion_rates[conversion_rates["stage"] == stage_name]["count"]
        return int(stage_data.values[0]) if len(stage_data) > 0 else 0
    
    visits = get_stage_count("visit")
    signups = get_stage_count("signup")
    activations = get_stage_count("activation")
    purchases = get_stage_count("purchase")
    
    total_revenue = user_flags["revenue"].sum() if len(user_flags) > 0 else 0
    overall_rate = (purchases / visits * 100) if visits > 0 else 0
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Visits", f"{visits:,}", help="Total unique visitors")
    
    with col2:
        st.metric("Sign-ups", f"{signups:,}", 
                  delta=f"{signups/visits*100:.1f}%" if visits > 0 else "0%",
                  help="Users who signed up")
    
    with col3:
        st.metric("Activations", f"{activations:,}",
                  delta=f"{activations/signups*100:.1f}%" if signups > 0 else "0%",
                  help="Users who activated")
    
    with col4:
        st.metric("Purchases", f"{purchases:,}",
                  delta=f"{purchases/activations*100:.1f}%" if activations > 0 else "0%",
                  help="Users who purchased")
    
    with col5:
        st.metric("Revenue", f"${total_revenue:,.2f}", help="Total revenue from purchases")
    
    with col6:
        st.metric("Overall Conversion", f"{overall_rate:.2f}%", help="Visit to Purchase rate")


def render_funnel_section(funnel_counts: pd.DataFrame, conversion_rates: pd.DataFrame):
    """Render funnel visualization and conversion rates."""
    st.markdown('<div class="section-header">Funnel Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        funnel_fig = create_funnel_chart(funnel_counts)
        st.plotly_chart(funnel_fig, key="funnel_chart")
    
    with col2:
        conversion_fig = create_conversion_rate_chart(conversion_rates)
        st.plotly_chart(conversion_fig, key="conversion_chart")


def render_dropoff_section(conversion_rates: pd.DataFrame):
    """Render drop-off analysis."""
    st.markdown('<div class="section-header">Drop-off Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        dropoff_fig = create_dropoff_chart(conversion_rates)
        st.plotly_chart(dropoff_fig, key="dropoff_chart")
    
    with col2:
        st.markdown("#### Drop-off Summary")
        
        dropoff_data = conversion_rates[conversion_rates["order"] > 0][
            ["stage", "dropoff_count", "dropoff_rate"]
        ].copy()
        
        if len(dropoff_data) > 0:
            dropoff_data.columns = ["Stage", "Users Lost", "Drop-off Rate (%)"]
            dropoff_data["Stage"] = dropoff_data["Stage"].str.title()
            dropoff_data["Users Lost"] = dropoff_data["Users Lost"].astype(int)
            
            st.dataframe(dropoff_data, hide_index=True)
            
            if dropoff_data["Drop-off Rate (%)"].max() > 0:
                highest_dropoff = dropoff_data.loc[dropoff_data["Drop-off Rate (%)"].idxmax()]
                st.info(f"**Biggest drop-off:** {highest_dropoff['Stage']} stage with {highest_dropoff['Drop-off Rate (%)']:.1f}% drop-off rate")
        else:
            st.info("No drop-off data available for the selected filters.")


def render_breakdown_section(breakdowns: dict):
    """Render breakdown analysis by dimensions."""
    st.markdown('<div class="section-header">Breakdown Analysis</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["By Traffic Source", "By Device", "By Country"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = create_breakdown_bar_chart(breakdowns["traffic_source"], "traffic_source", "overall_conversion_rate")
            st.plotly_chart(fig, key="traffic_conversion")
        with col2:
            fig = create_multi_metric_breakdown(breakdowns["traffic_source"], "traffic_source")
            st.plotly_chart(fig, key="traffic_multi")
        
        st.markdown("#### Detailed Metrics by Traffic Source")
        display_df = breakdowns["traffic_source"][
            ["traffic_source", "visits", "signups", "activations", "purchases", "revenue", "overall_conversion_rate"]
        ].copy()
        display_df.columns = ["Traffic Source", "Visits", "Sign-ups", "Activations", "Purchases", "Revenue", "Conversion Rate (%)"]
        display_df["Revenue"] = display_df["Revenue"].apply(lambda x: f"${x:,.2f}")
        st.dataframe(display_df, hide_index=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig = create_breakdown_bar_chart(breakdowns["device"], "device", "overall_conversion_rate")
            st.plotly_chart(fig, key="device_conversion")
        with col2:
            fig = create_multi_metric_breakdown(breakdowns["device"], "device")
            st.plotly_chart(fig, key="device_multi")
        
        st.markdown("#### Detailed Metrics by Device")
        display_df = breakdowns["device"][
            ["device", "visits", "signups", "activations", "purchases", "revenue", "overall_conversion_rate"]
        ].copy()
        display_df.columns = ["Device", "Visits", "Sign-ups", "Activations", "Purchases", "Revenue", "Conversion Rate (%)"]
        display_df["Revenue"] = display_df["Revenue"].apply(lambda x: f"${x:,.2f}")
        st.dataframe(display_df, hide_index=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            fig = create_breakdown_bar_chart(breakdowns["country"], "country", "overall_conversion_rate")
            st.plotly_chart(fig, key="country_conversion")
        with col2:
            fig = create_breakdown_bar_chart(breakdowns["country"], "country", "visits")
            st.plotly_chart(fig, key="country_visits")
        
        st.markdown("#### Detailed Metrics by Country")
        display_df = breakdowns["country"][
            ["country", "visits", "signups", "activations", "purchases", "revenue", "overall_conversion_rate"]
        ].copy()
        display_df.columns = ["Country", "Visits", "Sign-ups", "Activations", "Purchases", "Revenue", "Conversion Rate (%)"]
        display_df["Revenue"] = display_df["Revenue"].apply(lambda x: f"${x:,.2f}")
        st.dataframe(display_df, hide_index=True)


def render_time_analysis(time_metrics: pd.DataFrame):
    """Render time-to-conversion analysis."""
    st.markdown('<div class="section-header">Time-to-Conversion Analysis</div>', unsafe_allow_html=True)
    
    time_stats = get_time_to_conversion_stats(time_metrics)
    
    if len(time_stats) == 0:
        st.warning("No time-to-conversion data available for the selected filters.")
        return
    
    col1, col2 = st.columns(2)
    
    time_columns = [col for col in time_metrics.columns if col.startswith("time_")]
    
    for i, col in enumerate(time_columns):
        metric_name = col.replace("time_", "").replace("_", " → ").title()
        
        with col1 if i % 2 == 0 else col2:
            fig = create_time_distribution_chart(
                time_metrics, 
                col, 
                f"Time: {metric_name}"
            )
            st.plotly_chart(fig, key=f"time_dist_{i}")
    
    st.markdown("#### Time-to-Conversion Statistics (Hours)")
    stats_display = time_stats.copy()
    stats_display.columns = ["Transition", "Users", "Mean", "Median", "Std Dev", "Min", "Max"]
    st.dataframe(stats_display, hide_index=True)


def render_cohort_analysis(df: pd.DataFrame):
    """Render cohort analysis section."""
    st.markdown('<div class="section-header">Cohort Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col2:
        cohort_period = st.selectbox(
            "Cohort Period",
            options=["week", "month"],
            index=0,
            help="Group users by their first visit week or month"
        )
    
    cohort_data = calculate_cohort_analysis(df, cohort_period)
    
    if len(cohort_data) == 0:
        st.warning("No cohort data available for the selected filters.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_cohort_heatmap(cohort_data, "overall_conversion_rate")
        st.plotly_chart(fig, key="cohort_heatmap")
    
    with col2:
        fig = create_cohort_trend_chart(cohort_data)
        st.plotly_chart(fig, key="cohort_trend")
    
    st.markdown("#### Cohort Metrics Table")
    display_df = cohort_data.copy()
    display_df["cohort"] = display_df["cohort"].dt.strftime("%Y-%m-%d")
    display_df["revenue"] = display_df["revenue"].apply(lambda x: f"${x:,.2f}")
    display_df.columns = ["Cohort", "Visits", "Sign-ups", "Activations", "Purchases", "Revenue", 
                          "Users", "Visit→Signup %", "Signup→Activation %", "Activation→Purchase %", "Overall %"]
    st.dataframe(display_df, hide_index=True)


def render_revenue_analytics(user_flags: pd.DataFrame):
    """Render revenue analytics dashboard."""
    st.markdown('<div class="section-header">Revenue Analytics</div>', unsafe_allow_html=True)
    
    revenue_metrics = calculate_revenue_metrics(user_flags)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Revenue", f"${revenue_metrics['total_revenue']:,.2f}")
    with col2:
        st.metric("ARPU", f"${revenue_metrics['arpu']:.2f}", help="Average Revenue Per User")
    with col3:
        st.metric("ARPPU", f"${revenue_metrics['arppu']:.2f}", help="Average Revenue Per Paying User")
    with col4:
        st.metric("Conversion to Paid", f"{revenue_metrics['conversion_to_paid']:.1f}%")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_revenue_bar_chart(revenue_metrics["ltv_by_source"], "traffic_source")
        st.plotly_chart(fig, key="ltv_source")
    
    with col2:
        fig = create_revenue_bar_chart(revenue_metrics["ltv_by_device"], "device")
        st.plotly_chart(fig, key="ltv_device")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_revenue_bar_chart(revenue_metrics["ltv_by_country"], "country")
        st.plotly_chart(fig, key="ltv_country")
    
    with col2:
        fig = create_revenue_distribution_chart(user_flags)
        st.plotly_chart(fig, key="revenue_dist")
    
    st.markdown("#### LTV by Traffic Source")
    ltv_source = revenue_metrics["ltv_by_source"].copy()
    ltv_source["total_revenue"] = ltv_source["total_revenue"].apply(lambda x: f"${x:,.2f}")
    ltv_source["avg_revenue"] = ltv_source["avg_revenue"].apply(lambda x: f"${x:.2f}")
    ltv_source["ltv"] = ltv_source["ltv"].apply(lambda x: f"${x:.2f}")
    ltv_source.columns = ["Traffic Source", "Total Revenue", "Avg Revenue", "Users", "Purchasers", "LTV", "Conversion %"]
    st.dataframe(ltv_source, hide_index=True)


def render_user_journeys(df: pd.DataFrame):
    """Render user journey exploration."""
    st.markdown('<div class="section-header">User Journey Exploration</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        journey_limit = st.selectbox(
            "Number of Users",
            options=[25, 50, 100, 200],
            index=2,
            help="Number of user journeys to display"
        )
    
    with col2:
        sort_by = st.selectbox(
            "Sort By",
            options=["revenue", "event_count", "journey_duration_hours"],
            format_func=lambda x: {"revenue": "Revenue", "event_count": "Event Count", "journey_duration_hours": "Journey Duration"}[x]
        )
    
    journeys = get_user_journeys(df, limit=journey_limit)
    
    if len(journeys) == 0:
        st.warning("No journey data available.")
        return
    
    journeys = journeys.sort_values(sort_by, ascending=False)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Journeys", len(journeys))
    with col2:
        completed = len(journeys[journeys["final_stage"] == "purchase"])
        st.metric("Completed Purchases", completed)
    with col3:
        avg_events = journeys["event_count"].mean()
        st.metric("Avg Events/User", f"{avg_events:.1f}")
    with col4:
        avg_duration = journeys["journey_duration_hours"].mean()
        st.metric("Avg Journey (hrs)", f"{avg_duration:.1f}")
    
    stage_counts = journeys["final_stage"].value_counts()
    stage_labels = {"purchase": "Purchased", "activation": "Activated", "signup": "Signed Up", "visit": "Visited Only"}
    
    st.markdown("#### Final Stage Distribution")
    stage_col1, stage_col2, stage_col3, stage_col4 = st.columns(4)
    cols = [stage_col1, stage_col2, stage_col3, stage_col4]
    
    for i, (stage, label) in enumerate(stage_labels.items()):
        count = stage_counts.get(stage, 0)
        with cols[i]:
            st.metric(label, count)
    
    st.markdown("#### User Journeys")
    display_df = journeys.copy()
    display_df["first_event"] = display_df["first_event"].dt.strftime("%Y-%m-%d %H:%M")
    display_df["last_event"] = display_df["last_event"].dt.strftime("%Y-%m-%d %H:%M")
    display_df["revenue"] = display_df["revenue"].apply(lambda x: f"${x:,.2f}")
    display_df["journey_duration_hours"] = display_df["journey_duration_hours"].apply(lambda x: f"{x:.1f}")
    display_df.columns = ["User ID", "Journey Path", "First Event", "Last Event", "Events", 
                          "Traffic Source", "Device", "Country", "Revenue", "Duration (hrs)", "Final Stage"]
    
    st.dataframe(display_df, hide_index=True, height=400)


def main():
    """Main application entry point."""
    st.markdown('<div class="main-header">Marketing Funnel Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Analyze your marketing funnel performance with interactive visualizations</div>', unsafe_allow_html=True)
    
    synthetic_df = load_synthetic_data()
    
    filters = render_sidebar(synthetic_df)
    
    if filters["data_source"] == "Upload Your Data" and filters["uploaded_df"] is not None:
        active_df = filters["uploaded_df"]
    else:
        active_df = synthetic_df
    
    filtered_df = filter_events(
        active_df,
        traffic_sources=filters["traffic_sources"],
        devices=filters["devices"],
        countries=filters["countries"],
        start_date=filters["start_date"],
        end_date=filters["end_date"]
    )
    
    if len(filtered_df) == 0:
        st.warning("No data matches the selected filters. Please adjust your filter criteria.")
        return
    
    unique_users = filtered_df["user_id"].nunique()
    total_events = len(filtered_df)
    date_range = f"{filtered_df['event_timestamp'].min().strftime('%Y-%m-%d')} to {filtered_df['event_timestamp'].max().strftime('%Y-%m-%d')}"
    
    st.info(f"**Analyzing:** {unique_users:,} users | {total_events:,} events | Date range: {date_range}")
    
    processed_data = process_funnel_data(filtered_df)
    
    render_kpis(processed_data["conversion_rates"], processed_data["user_flags"])
    
    main_tab1, main_tab2, main_tab3, main_tab4 = st.tabs([
        "Funnel Analysis", "Cohort Analysis", "Revenue Analytics", "User Journeys"
    ])
    
    with main_tab1:
        render_funnel_section(processed_data["funnel_counts"], processed_data["conversion_rates"])
        render_dropoff_section(processed_data["conversion_rates"])
        breakdowns = get_breakdown_data(processed_data["user_flags"])
        render_breakdown_section(breakdowns)
        render_time_analysis(processed_data["time_metrics"])
    
    with main_tab2:
        render_cohort_analysis(filtered_df)
    
    with main_tab3:
        render_revenue_analytics(processed_data["user_flags"])
    
    with main_tab4:
        render_user_journeys(filtered_df)
    
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #9CA3AF; font-size: 0.875rem;">
            Marketing Funnel Analysis Dashboard | Built with Streamlit
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

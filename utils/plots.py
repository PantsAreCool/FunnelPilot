"""
Plotting Utilities for Marketing Funnel Analysis

Creates interactive Plotly visualizations for funnel analysis.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


COLORS = {
    "primary": "#4F46E5",
    "secondary": "#7C3AED",
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "info": "#3B82F6"
}

FUNNEL_COLORS = ["#4F46E5", "#7C3AED", "#A855F7", "#EC4899"]


def create_funnel_chart(funnel_df: pd.DataFrame) -> go.Figure:
    """
    Create an interactive funnel chart.
    
    Args:
        funnel_df: DataFrame with stage and count columns
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure(go.Funnel(
        y=funnel_df["stage"].str.title(),
        x=funnel_df["count"],
        textposition="inside",
        textinfo="value+percent initial",
        marker=dict(
            color=FUNNEL_COLORS[:len(funnel_df)],
            line=dict(width=2, color="white")
        ),
        connector=dict(line=dict(color="rgba(0,0,0,0.1)", width=1))
    ))
    
    fig.update_layout(
        title=dict(
            text="Marketing Funnel",
            font=dict(size=20)
        ),
        font=dict(size=14),
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig


def create_conversion_rate_chart(conversion_df: pd.DataFrame) -> go.Figure:
    """
    Create a bar chart showing step conversion rates.
    
    Args:
        conversion_df: DataFrame with conversion rate data
    
    Returns:
        Plotly Figure object
    """
    df = conversion_df[conversion_df["order"] > 0].copy()
    df["stage_transition"] = df["stage"].apply(
        lambda x: f"→ {x.title()}"
    )
    
    fig = go.Figure(go.Bar(
        x=df["stage_transition"],
        y=df["step_conversion_rate"],
        text=df["step_conversion_rate"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside",
        marker_color=FUNNEL_COLORS[1:len(df)+1],
        hovertemplate="<b>%{x}</b><br>Conversion Rate: %{y:.1f}%<extra></extra>"
    ))
    
    fig.update_layout(
        title="Step Conversion Rates",
        xaxis_title="Funnel Step",
        yaxis_title="Conversion Rate (%)",
        yaxis_range=[0, 100],
        height=350,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig


def create_dropoff_chart(conversion_df: pd.DataFrame) -> go.Figure:
    """
    Create a waterfall-style drop-off analysis chart.
    
    Args:
        conversion_df: DataFrame with dropoff data
    
    Returns:
        Plotly Figure object
    """
    df = conversion_df[conversion_df["order"] > 0].copy()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df["stage"].str.title(),
        y=df["dropoff_count"],
        name="Users Lost",
        marker_color=COLORS["danger"],
        text=df["dropoff_count"].apply(lambda x: f"{int(x):,}"),
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Users Lost: %{y:,}<extra></extra>"
    ))
    
    fig.add_trace(go.Scatter(
        x=df["stage"].str.title(),
        y=df["dropoff_rate"],
        name="Drop-off Rate",
        yaxis="y2",
        mode="lines+markers",
        line=dict(color=COLORS["warning"], width=3),
        marker=dict(size=10),
        hovertemplate="<b>%{x}</b><br>Drop-off Rate: %{y:.1f}%<extra></extra>"
    ))
    
    fig.update_layout(
        title="Drop-off Analysis",
        xaxis_title="Funnel Stage",
        yaxis=dict(title="Users Lost", side="left"),
        yaxis2=dict(
            title="Drop-off Rate (%)",
            side="right",
            overlaying="y",
            range=[0, 100]
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        height=400,
        margin=dict(l=20, r=60, t=80, b=20)
    )
    
    return fig


def create_breakdown_bar_chart(
    breakdown_df: pd.DataFrame,
    dimension: str,
    metric: str = "overall_conversion_rate"
) -> go.Figure:
    """
    Create a bar chart for dimension breakdown.
    
    Args:
        breakdown_df: DataFrame with breakdown metrics
        dimension: Name of the dimension (for labeling)
        metric: Metric to visualize
    
    Returns:
        Plotly Figure object
    """
    df = breakdown_df.head(10)
    
    metric_labels = {
        "overall_conversion_rate": "Overall Conversion Rate (%)",
        "visits": "Total Visits",
        "purchases": "Total Purchases",
        "revenue": "Total Revenue ($)"
    }
    
    is_rate = "rate" in metric.lower()
    
    fig = go.Figure(go.Bar(
        x=df[dimension],
        y=df[metric],
        text=df[metric].apply(lambda x: f"{x:.1f}%" if is_rate else f"{x:,.0f}"),
        textposition="outside",
        marker_color=COLORS["primary"],
        hovertemplate=f"<b>%{{x}}</b><br>{metric_labels.get(metric, metric)}: %{{y:,.1f}}<extra></extra>"
    ))
    
    fig.update_layout(
        title=f"{metric_labels.get(metric, metric)} by {dimension.replace('_', ' ').title()}",
        xaxis_title=dimension.replace("_", " ").title(),
        yaxis_title=metric_labels.get(metric, metric),
        height=350,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    if is_rate:
        fig.update_layout(yaxis_range=[0, max(100, df[metric].max() * 1.1)])
    
    return fig


def create_time_distribution_chart(
    time_df: pd.DataFrame,
    column: str,
    title: str = "Time Distribution"
) -> go.Figure:
    """
    Create a histogram/box plot for time-to-conversion distribution.
    
    Args:
        time_df: DataFrame with time data
        column: Column name to plot
        title: Chart title
    
    Returns:
        Plotly Figure object
    """
    data = time_df[column].dropna()
    
    if len(data) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(height=300)
        return fig
    
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.3, 0.7],
        vertical_spacing=0.05,
        shared_xaxes=True
    )
    
    fig.add_trace(
        go.Box(
            x=data,
            name="",
            marker_color=COLORS["primary"],
            boxpoints="outliers",
            hovertemplate="Hours: %{x:.1f}<extra></extra>"
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Histogram(
            x=data,
            name="",
            marker_color=COLORS["primary"],
            opacity=0.7,
            hovertemplate="Hours: %{x:.1f}<br>Count: %{y}<extra></extra>"
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=title,
        showlegend=False,
        height=350,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    fig.update_xaxes(title_text="Hours", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    
    return fig


def create_multi_metric_breakdown(
    breakdown_df: pd.DataFrame,
    dimension: str
) -> go.Figure:
    """
    Create a grouped bar chart showing multiple metrics by dimension.
    
    Args:
        breakdown_df: DataFrame with breakdown metrics
        dimension: Dimension column name
    
    Returns:
        Plotly Figure object
    """
    df = breakdown_df.head(8)
    
    fig = go.Figure()
    
    metrics = [
        ("visit_to_signup_rate", "Visit → Signup", FUNNEL_COLORS[1]),
        ("signup_to_activation_rate", "Signup → Activation", FUNNEL_COLORS[2]),
        ("activation_to_purchase_rate", "Activation → Purchase", FUNNEL_COLORS[3])
    ]
    
    for metric, name, color in metrics:
        fig.add_trace(go.Bar(
            x=df[dimension],
            y=df[metric],
            name=name,
            marker_color=color,
            text=df[metric].apply(lambda x: f"{x:.0f}%"),
            textposition="outside",
            hovertemplate=f"<b>%{{x}}</b><br>{name}: %{{y:.1f}}%<extra></extra>"
        ))
    
    fig.update_layout(
        title=f"Conversion Rates by {dimension.replace('_', ' ').title()}",
        xaxis_title=dimension.replace("_", " ").title(),
        yaxis_title="Conversion Rate (%)",
        yaxis_range=[0, 100],
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        height=400,
        margin=dict(l=20, r=20, t=100, b=20)
    )
    
    return fig


def create_kpi_metric(value: float, label: str, format_str: str = "{:,.0f}") -> dict:
    """
    Create a dictionary for KPI display.
    
    Args:
        value: Metric value
        label: Metric label
        format_str: Format string for the value
    
    Returns:
        Dictionary with formatted metric info
    """
    return {
        "value": format_str.format(value),
        "label": label
    }


def create_cohort_heatmap(cohort_df: pd.DataFrame, metric: str = "overall_conversion_rate") -> go.Figure:
    """
    Create a heatmap for cohort analysis.
    
    Args:
        cohort_df: DataFrame with cohort metrics
        metric: Metric to visualize
    
    Returns:
        Plotly Figure object
    """
    if len(cohort_df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No cohort data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    df = cohort_df.copy()
    df["cohort_label"] = df["cohort"].dt.strftime("%Y-%m-%d")
    
    metric_labels = {
        "overall_conversion_rate": "Overall Conversion Rate (%)",
        "visit_to_signup_rate": "Visit → Signup Rate (%)",
        "signup_to_activation_rate": "Signup → Activation Rate (%)",
        "activation_to_purchase_rate": "Activation → Purchase Rate (%)"
    }
    
    fig = go.Figure(go.Bar(
        x=df["cohort_label"],
        y=df[metric],
        text=df[metric].apply(lambda x: f"{x:.1f}%"),
        textposition="outside",
        marker=dict(
            color=df[metric],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="%")
        ),
        hovertemplate="<b>%{x}</b><br>" + metric_labels.get(metric, metric) + ": %{y:.1f}%<extra></extra>"
    ))
    
    fig.update_layout(
        title=f"Cohort Analysis: {metric_labels.get(metric, metric)}",
        xaxis_title="Cohort (First Visit Date)",
        yaxis_title=metric_labels.get(metric, metric),
        height=400,
        margin=dict(l=20, r=20, t=60, b=80),
        xaxis_tickangle=-45
    )
    
    return fig


def create_cohort_trend_chart(cohort_df: pd.DataFrame) -> go.Figure:
    """
    Create a line chart showing cohort trends over time.
    
    Args:
        cohort_df: DataFrame with cohort metrics
    
    Returns:
        Plotly Figure object
    """
    if len(cohort_df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No cohort data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    df = cohort_df.copy()
    df["cohort_label"] = df["cohort"].dt.strftime("%Y-%m-%d")
    
    fig = go.Figure()
    
    metrics = [
        ("visit_to_signup_rate", "Visit → Signup", FUNNEL_COLORS[1]),
        ("signup_to_activation_rate", "Signup → Activation", FUNNEL_COLORS[2]),
        ("activation_to_purchase_rate", "Activation → Purchase", FUNNEL_COLORS[3])
    ]
    
    for metric, name, color in metrics:
        fig.add_trace(go.Scatter(
            x=df["cohort_label"],
            y=df[metric],
            name=name,
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=8),
            hovertemplate=f"<b>%{{x}}</b><br>{name}: %{{y:.1f}}%<extra></extra>"
        ))
    
    fig.update_layout(
        title="Conversion Rate Trends by Cohort",
        xaxis_title="Cohort (First Visit Date)",
        yaxis_title="Conversion Rate (%)",
        yaxis_range=[0, 100],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        height=400,
        margin=dict(l=20, r=20, t=80, b=80),
        xaxis_tickangle=-45
    )
    
    return fig


def create_revenue_bar_chart(ltv_df: pd.DataFrame, dimension: str) -> go.Figure:
    """
    Create a bar chart for LTV by dimension.
    
    Args:
        ltv_df: DataFrame with LTV metrics
        dimension: Dimension name
    
    Returns:
        Plotly Figure object
    """
    df = ltv_df.head(10)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df[dimension],
        y=df["ltv"],
        name="LTV",
        marker_color=COLORS["success"],
        text=df["ltv"].apply(lambda x: f"${x:.2f}"),
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>LTV: $%{y:.2f}<extra></extra>"
    ))
    
    fig.update_layout(
        title=f"Lifetime Value (LTV) by {dimension.replace('_', ' ').title()}",
        xaxis_title=dimension.replace("_", " ").title(),
        yaxis_title="LTV ($)",
        height=350,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig


def create_revenue_distribution_chart(user_flags: pd.DataFrame) -> go.Figure:
    """
    Create a histogram of revenue distribution.
    
    Args:
        user_flags: DataFrame with user data including revenue
    
    Returns:
        Plotly Figure object
    """
    paying_users = user_flags[user_flags["revenue"] > 0]["revenue"]
    
    if len(paying_users) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No revenue data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=350)
        return fig
    
    fig = go.Figure(go.Histogram(
        x=paying_users,
        nbinsx=30,
        marker_color=COLORS["success"],
        opacity=0.8,
        hovertemplate="Revenue: $%{x:.2f}<br>Count: %{y}<extra></extra>"
    ))
    
    fig.update_layout(
        title="Revenue Distribution (Paying Users)",
        xaxis_title="Revenue ($)",
        yaxis_title="Number of Users",
        height=350,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

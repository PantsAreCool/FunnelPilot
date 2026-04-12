import pandas as pd


def _safe_stage_count(counts: pd.DataFrame, stage: str) -> int:
    rows = counts.loc[counts["stage"] == stage, "count"]
    return int(rows.values[0]) if len(rows) > 0 else 0


def compute_baseline_metrics(funnel_counts: pd.DataFrame, user_flags: pd.DataFrame) -> dict:
    counts = funnel_counts.sort_values("order")
    visited = _safe_stage_count(counts, "visit")
    signed_up = _safe_stage_count(counts, "signup")
    activated = _safe_stage_count(counts, "activation")
    purchased = _safe_stage_count(counts, "purchase")

    conv1 = signed_up / visited if visited > 0 else 0
    conv2 = activated / signed_up if signed_up > 0 else 0
    conv3 = purchased / activated if activated > 0 else 0
    overall = purchased / visited if visited > 0 else 0

    total_revenue = float(user_flags["revenue"].sum())
    revenue_per_user = total_revenue / purchased if purchased > 0 else 0

    return {
        "visited": visited,
        "signed_up": signed_up,
        "activated": activated,
        "purchased": purchased,
        "conv1": conv1,
        "conv2": conv2,
        "conv3": conv3,
        "overall": overall,
        "total_revenue": total_revenue,
        "revenue_per_user": revenue_per_user,
    }


def simulate_funnel_impact(baseline: dict, lift1: float, lift2: float, lift3: float) -> dict:
    new_conv1 = min(baseline["conv1"] * (1 + lift1), 1.0)
    new_conv2 = min(baseline["conv2"] * (1 + lift2), 1.0)
    new_conv3 = min(baseline["conv3"] * (1 + lift3), 1.0)

    visited = baseline["visited"]
    new_signed_up = visited * new_conv1
    new_activated = new_signed_up * new_conv2
    new_purchased = new_activated * new_conv3
    new_overall = new_purchased / visited if visited > 0 else 0

    new_revenue = new_purchased * baseline["revenue_per_user"]

    return {
        "visited": visited,
        "signed_up": new_signed_up,
        "activated": new_activated,
        "purchased": new_purchased,
        "conv1": new_conv1,
        "conv2": new_conv2,
        "conv3": new_conv3,
        "overall": new_overall,
        "total_revenue": new_revenue,
    }


def compute_deltas(baseline: dict, simulated: dict) -> dict:
    delta_signups = simulated["signed_up"] - baseline["signed_up"]
    delta_activations = simulated["activated"] - baseline["activated"]
    delta_purchases = simulated["purchased"] - baseline["purchased"]
    delta_revenue = simulated["total_revenue"] - baseline["total_revenue"]

    overall_lift_pct = (
        (simulated["overall"] - baseline["overall"]) / baseline["overall"] * 100
        if baseline["overall"] > 0
        else 0
    )

    return {
        "delta_signups": delta_signups,
        "delta_activations": delta_activations,
        "delta_purchases": delta_purchases,
        "delta_revenue": delta_revenue,
        "overall_lift_pct": overall_lift_pct,
    }


def generate_insight(baseline: dict, simulated: dict, lift1: float, lift2: float, lift3: float) -> str:
    lifts = {
        "signup": lift1,
        "activation": lift2,
        "purchase": lift3,
    }

    if all(v == 0 for v in lifts.values()):
        return "Adjust the sliders above to simulate improvements at each funnel stage and see the projected impact."

    stage_labels = {
        "signup": "visit-to-signup",
        "activation": "signup-to-activation",
        "purchase": "activation-to-purchase",
    }

    stage_impact = {}
    for stage, lift_val in lifts.items():
        if lift_val > 0:
            solo = simulate_funnel_impact(baseline, 
                lift_val if stage == "signup" else 0,
                lift_val if stage == "activation" else 0,
                lift_val if stage == "purchase" else 0)
            stage_impact[stage] = solo["purchased"] - baseline["purchased"]

    if not stage_impact:
        return "Adjust the sliders above to simulate improvements at each funnel stage and see the projected impact."

    best_stage = max(stage_impact, key=stage_impact.get)
    best_lift_pct = int(lifts[best_stage] * 100)
    best_label = stage_labels[best_stage]

    total_purchase_delta = simulated["purchased"] - baseline["purchased"]
    purchase_pct_increase = (
        total_purchase_delta / baseline["purchased"] * 100 if baseline["purchased"] > 0 else 0
    )

    parts = []

    if len(stage_impact) > 1:
        parts.append(
            f"Improving {best_label} conversion by {best_lift_pct}% has the highest leverage, "
            f"contributing ~{stage_impact[best_stage]:,.0f} additional purchases on its own."
        )
    elif len(stage_impact) == 1:
        parts.append(
            f"A {best_lift_pct}% improvement in {best_label} conversion would generate "
            f"~{stage_impact[best_stage]:,.0f} additional purchases."
        )

    active_lifts = [s for s, v in lifts.items() if v > 0]
    if len(active_lifts) > 1:
        parts.append(
            f"Combined, the simulated changes produce ~{total_purchase_delta:,.0f} additional purchases "
            f"({purchase_pct_increase:,.1f}% increase) due to compounding effects across stages."
        )

    revenue_delta = simulated["total_revenue"] - baseline["total_revenue"]
    if revenue_delta > 0:
        parts.append(f"This translates to an estimated ${revenue_delta:,.0f} in additional revenue.")

    return " ".join(parts)

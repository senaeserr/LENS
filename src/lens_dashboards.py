import re
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from html import escape

from lens_core import *
from lens_core import (_activation_event, _find_churn_column, _funnel_steps, _money_prefix, _normalize_column_name, _paid_event, _payment_attempt_failure_counts, _safe_datetime, _session_analysis_cache_get, _successful_revenue_events, _user_dimension_at_event, _user_dimension_snapshot)

ANALYSIS_COLORS = {
    "turquoise": "#46E1DA",
    "turquoise_2": "#73F3EC",
    "magenta": "#FF4FA3",
    "magenta_2": "#FF79C6",
    "purple": "#8B5CF6",
    "purple_2": "#6D5DFB",
    "blue": "#4F8CFF",
    "ink": "#F7F8FF",
    "muted": "#A6AECA",
}

# One ranking palette for every categorical dashboard chart.
# Highest value is always magenta; the scale then moves through purple/blue
# and ends in turquoise for the lowest value.
RANK_PALETTE = [
    "#FF4FA3",
    "#E75AC8",
    "#C45DE2",
    "#9B63F4",
    "#7868FF",
    "#5C83F7",
    "#4AA7EC",
    "#46C9E2",
    "#46E1DA",
]


def _interpolate_hex(start_hex, end_hex, count):
    if count <= 1:
        return [start_hex]

    def rgb(value):
        value = value.lstrip("#")
        return tuple(int(value[i:i+2], 16) for i in (0, 2, 4))

    a = rgb(start_hex)
    b = rgb(end_hex)
    out = []
    for i in range(count):
        t = i / (count - 1)
        vals = tuple(round(a[j] + (b[j] - a[j]) * t) for j in range(3))
        out.append("#%02X%02X%02X" % vals)
    return out


def _series_colors(n):
    """High -> low palette used consistently across every ranked bar chart."""
    if n <= 0:
        return []
    if n <= len(RANK_PALETTE):
        # Sample across the full palette so even 2–5 bars preserve the
        # magenta-high / turquoise-low meaning.
        if n == 1:
            return [RANK_PALETTE[0]]
        idx = np.linspace(0, len(RANK_PALETTE) - 1, n).round().astype(int)
        return [RANK_PALETTE[i] for i in idx]
    return _interpolate_hex(RANK_PALETTE[0], RANK_PALETTE[-1], n)


def _rank_colors(values):
    """Return colors in original row order, ranked high -> low by value."""
    ser = pd.Series(values).astype(float)
    order = ser.rank(method="first", ascending=False).astype(int) - 1
    palette = _series_colors(len(ser))
    return [palette[i] for i in order]


def _plotly_style(fig, height=360, y_percent=False):
    # Explicit empty title prevents Plotly/Streamlit from rendering the
    # literal word "undefined" above charts on some versions.
    fig.update_layout(
        height=height,
        margin=dict(l=38, r=24, t=18, b=38),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(9,13,42,0.74)",
        font=dict(
            family="Inter, Segoe UI, sans-serif",
            color="#E9ECF8",
            size=12,
        ),
        title=dict(text="", x=0, y=1),
        legend_title_text="",
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#D8DDF1"),
        ),
        hoverlabel=dict(
            bgcolor="#101635",
            bordercolor="#46E1DA",
            font=dict(color="#FFFFFF", family="Inter"),
        ),
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor="rgba(255,255,255,0.12)",
        tickfont=dict(color="#B8C0DA"),
        title_font=dict(color="#8C96B5", size=12),
        automargin=True,
    )
    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.08)",
        zeroline=False,
        linecolor="rgba(255,255,255,0.08)",
        tickfont=dict(color="#B8C0DA"),
        title_font=dict(color="#8C96B5", size=12),
        automargin=True,
    )
    if y_percent:
        fig.update_yaxes(ticksuffix="%")
    return fig


def render_dashboard_header(title, description, eyebrow="Live dashboard"):
    st.html(
        f"""
<div class="analysis-dashboard-head">
    <div class="analysis-dashboard-copy">
        <div class="analysis-dashboard-eyebrow">{eyebrow}</div>
        <div class="analysis-dashboard-title">{title}</div>
        <div class="analysis-dashboard-desc">{description}</div>
    </div>
    <div class="analysis-dashboard-orbit">
        <span></span><span></span><span></span>
    </div>
</div>
"""
    )


def render_kpis(items):
    cols = st.columns(len(items))
    for idx, (col, item) in enumerate(zip(cols, items)):
        label, value, help_text = item
        accent_class = ["turquoise", "magenta", "purple", "blue", "turquoise"][idx % 5]
        with col:
            st.html(
                f"""
<div class="analysis-kpi-card {accent_class}">
    <div class="analysis-kpi-label">{label}</div>
    <div class="analysis-kpi-value">{value}</div>
    <div class="analysis-kpi-note">{help_text}</div>
</div>
"""
            )


def render_chart_heading(title, subtitle=None):
    subtitle_html = f'<div class="analysis-chart-subtitle">{subtitle}</div>' if subtitle else ""
    st.html(
        f"""
<div class="analysis-chart-heading">
    <div class="analysis-chart-title">{title}</div>
    {subtitle_html}
</div>
"""
    )


def render_acquisition_dashboard(df):
    user_col = st.session_state.user_col
    event_col = st.session_state.event_col
    time_col = st.session_state.timestamp_col
    channel_col = find_column(
        df,
        ["acquisition_channel", "marketing_channel", "channel", "source", "traffic_source", "utm_source"],
    )

    render_dashboard_header(
        "Acquisition",
        "See which sources bring users in — and whether those users actually reach product value.",
    )

    if not channel_col:
        st.warning("Acquisition dashboard needs a channel/source field.")
        return

    tmp = df[[user_col, event_col, time_col, channel_col]].copy()
    tmp[time_col] = _safe_datetime(tmp[time_col])
    first_users = _user_dimension_snapshot(df, channel_col, include_missing=True)
    user_mix = (
        first_users[channel_col]
        .value_counts()
        .rename_axis("Acquisition source")
        .reset_index(name="Users")
    )

    activation_event = _activation_event(df)
    activation_rate = None
    if activation_event:
        activated = set(tmp.loc[tmp[event_col].astype(str) == activation_event, user_col].dropna())
        first_users["Activated"] = first_users[user_col].isin(activated)
        activation_rate = (
            first_users.groupby(channel_col, dropna=False)["Activated"]
            .mean()
            .mul(100)
            .reset_index()
            .rename(columns={channel_col: "Acquisition source", "Activated": "Activation rate"})
        )

    ranked_mix = user_mix[user_mix["Acquisition source"] != MISSING_DIMENSION_LABEL].copy()
    top_channel = str(ranked_mix.iloc[0]["Acquisition source"]) if len(ranked_mix) else "—"
    top_share = (ranked_mix.iloc[0]["Users"] / user_mix["Users"].sum() * 100) if len(ranked_mix) else 0
    best_quality = "—"
    if activation_rate is not None and len(activation_rate):
        ranked_quality = activation_rate[activation_rate["Acquisition source"] != MISSING_DIMENSION_LABEL]
        if len(ranked_quality):
            best_quality = str(ranked_quality.sort_values("Activation rate", ascending=False).iloc[0]["Acquisition source"])

    render_kpis([
        ("Acquisition sources", f"{ranked_mix['Acquisition source'].nunique():,}", "Distinct recorded sources"),
        ("Largest source", top_channel, f"{top_share:.1f}% of acquired users"),
        ("Best activation", best_quality, "Highest activation-rate source"),
    ])

    missing_acq_users = int(user_mix.loc[user_mix["Acquisition source"] == MISSING_DIMENSION_LABEL, "Users"].sum())

    st.markdown("")
    c1, c2 = st.columns(2)
    with c1:
        render_chart_heading("User mix by acquisition source", "Where new users are coming from")
        mix_plot = ranked_mix.head(12).copy()
        fig = px.bar(mix_plot, x="Acquisition source", y="Users")
        fig.update_traces(
            marker_color=_rank_colors(mix_plot["Users"]),
            marker_line_width=0,
            hovertemplate="<b>%{x}</b><br>Users: %{y:,}<extra></extra>",
        )
        fig.update_xaxes(title_text="Acquisition source")
        fig.update_yaxes(title_text="Users")
        st.plotly_chart(_plotly_style(fig), use_container_width=True, config={"displayModeBar": False})
        if missing_acq_users:
            st.html(
                f"<div class='missing-data-note'>{missing_acq_users:,} users have no recorded acquisition source and are excluded from source ranking.</div>"
            )

    with c2:
        if activation_rate is not None:
            render_chart_heading("Activation by acquisition source", "Quality, not just acquisition volume")
            quality = (
                activation_rate[activation_rate["Acquisition source"] != MISSING_DIMENSION_LABEL]
                .sort_values("Activation rate", ascending=False)
                .head(12)
            )
            fig = px.bar(quality, x="Acquisition source", y="Activation rate")
            fig.update_traces(
                marker_color=_rank_colors(quality["Activation rate"]),
                marker_line_width=0,
                hovertemplate="<b>%{x}</b><br>Activation: %{y:.1f}%<extra></extra>",
            )
            fig.update_xaxes(title_text="Acquisition source")
            fig.update_yaxes(title_text="Activation rate")
            st.plotly_chart(_plotly_style(fig, y_percent=True), use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Activation event could not be identified automatically for source-quality comparison.")


def _ordered_funnel_counts(tmp, user_col, event_col, time_col, steps):
    """Count users who reach every step in order, not merely users who fired each event."""
    work = tmp[[user_col, event_col, time_col]].dropna().copy()
    work[time_col] = _safe_datetime(work[time_col])
    work = work.dropna(subset=[time_col]).sort_values([user_col, time_col])
    eligible_users = set(work[user_col].dropna().unique())
    rows = []
    previous_users = eligible_users
    previous_times = None

    for label, event_name in steps:
        ev = work[work[event_col].astype(str).eq(str(event_name))][[user_col, time_col]].copy()
        if previous_times is None:
            first = ev.groupby(user_col)[time_col].min()
            reached = set(first.index) & previous_users
            previous_times = first[first.index.isin(reached)]
        else:
            joined = ev.merge(previous_times.rename("prev_time"), left_on=user_col, right_index=True, how="inner")
            joined = joined[joined[time_col] >= joined["prev_time"]]
            first = joined.groupby(user_col)[time_col].min()
            reached = set(first.index)
            previous_times = first
        previous_users = reached
        rows.append({"Funnel step": label, "Event": event_name, "Users": int(len(reached))})
    return pd.DataFrame(rows)


def _ordered_funnel_counts_cached(df, user_col, event_col, time_col, steps):
    step_key = tuple((str(label), str(event_name)) for label, event_name in steps)
    key = ("ordered_funnel", str(user_col), str(event_col), str(time_col), step_key)
    return _session_analysis_cache_get(
        key,
        lambda: _ordered_funnel_counts(df, user_col, event_col, time_col, steps),
    ).copy()


def render_activation_dashboard(df):
    user_col = st.session_state.user_col
    event_col = st.session_state.event_col
    time_col = st.session_state.timestamp_col
    tmp = df[[user_col, event_col, time_col]].copy()
    steps = _funnel_steps(df)

    render_dashboard_header("Activation & Funnel", "Follow users through the detected journey in timestamp order, locate the biggest drop and measure time to value.")
    if len(steps) < 2:
        st.warning("LENS could not identify enough funnel events automatically. Review the event-role mapping before trusting a funnel.")
        return

    funnel = _ordered_funnel_counts_cached(tmp, user_col, event_col, time_col, steps)
    if funnel.empty:
        st.warning("No ordered funnel could be constructed from the confirmed event roles.")
        return
    base = max(int(funnel.iloc[0]["Users"]), 1)
    funnel["Conversion"] = funnel["Users"] / base * 100
    final_conv = funnel.iloc[-1]["Conversion"]

    largest_drop, drop_label = 0.0, "—"
    for i in range(1, len(funnel)):
        prev = max(funnel.iloc[i - 1]["Users"], 1)
        drop = (1 - funnel.iloc[i]["Users"] / prev) * 100
        if drop > largest_drop:
            largest_drop = drop
            drop_label = f"{funnel.iloc[i-1]['Funnel step']} → {funnel.iloc[i]['Funnel step']}"

    activation_event = _activation_event(df)
    activation_rate = None
    time_to_activation = None
    if activation_event:
        tmp[time_col] = _safe_datetime(tmp[time_col])
        first_ts = tmp.groupby(user_col)[time_col].min()
        act_ts = tmp.loc[tmp[event_col].astype(str).eq(str(activation_event))].groupby(user_col)[time_col].min()
        joined = pd.concat([first_ts.rename("first"), act_ts.rename("act")], axis=1).dropna()
        joined = joined[joined["act"] >= joined["first"]]
        activation_rate = len(joined) / max(tmp[user_col].nunique(), 1) * 100
        if len(joined):
            time_to_activation = ((joined["act"] - joined["first"]).dt.total_seconds() / 3600).median()

    render_kpis([
        ("End-to-end conversion", f"{final_conv:.1f}%", "Reached the final detected step in order"),
        ("Largest drop", f"{largest_drop:.1f}%", drop_label),
        ("Activation rate", f"{activation_rate:.1f}%" if activation_rate is not None else "—", "Reached detected core action"),
        ("Median time to activation", f"{time_to_activation:.1f}h" if time_to_activation is not None else "—", "Median first-event → activation time"),
    ])

    st.markdown("")
    c1, c2 = st.columns([1.15, 1])
    with c1:
        render_chart_heading("Ordered core funnel", "Users must reach each detected step after the prior step")
        fig = go.Figure(go.Funnel(y=funnel["Funnel step"], x=funnel["Users"], textinfo="value+percent initial", marker=dict(color=_series_colors(len(funnel))), connector=dict(line=dict(color="rgba(255,255,255,0.13)", width=1)), hovertemplate="<b>%{y}</b><br>Users: %{x:,}<extra></extra>"))
        st.plotly_chart(_plotly_style(fig, 410), use_container_width=True, config={"displayModeBar": False})
    with c2:
        render_chart_heading("Conversion from first step", "Ordered survival from the original cohort")
        fig = px.bar(funnel, x="Funnel step", y="Conversion")
        fig.update_traces(marker_color=_rank_colors(funnel["Conversion"]), marker_line_width=0, hovertemplate="<b>%{x}</b><br>Conversion: %{y:.1f}%<extra></extra>")
        fig.update_yaxes(range=[0, 105], title_text="Conversion")
        fig.update_xaxes(title_text="Funnel step")
        st.plotly_chart(_plotly_style(fig, 410, y_percent=True), use_container_width=True, config={"displayModeBar": False})


def _retention_suitability_cached(df):
    return _session_analysis_cache_get("retention_suitability", lambda: _retention_suitability(df))


def render_engagement_dashboard(df):
    user_col = st.session_state.user_col
    time_col = st.session_state.timestamp_col
    tmp = df[[user_col, time_col]].copy()
    tmp[time_col] = _safe_datetime(tmp[time_col])
    tmp = tmp.dropna(subset=[time_col, user_col])
    tmp["Date"] = tmp[time_col].dt.floor("D")
    daily = tmp.groupby("Date")[user_col].nunique().reset_index(name="Active users")

    avg_events_per_user = len(df) / max(df[user_col].nunique(), 1)
    suitability = _retention_suitability_cached(df)
    render_dashboard_header("Engagement & Retention", "Track recurring product behavior only when the event log contains enough engagement evidence.")

    if not suitability.get("suitable"):
        active_days = tmp.groupby(user_col)["Date"].nunique() if len(tmp) else pd.Series(dtype=float)
        repeat_users = int((active_days >= 2).sum()) if len(active_days) else 0
        repeat_share = repeat_users / max(len(active_days), 1) * 100 if len(active_days) else 0
        render_kpis([
            ("Events per user", f"{avg_events_per_user:.1f}", "Average event volume per user"),
            ("Users active on 2+ days", f"{repeat_share:.1f}%", f"{repeat_users:,} users with activity on multiple dates"),
        ])
        st.info(
            "Exact-day D1/D7/D30 retention is not shown because this file does not contain enough recurring product-engagement evidence. "
            "Lifecycle, payment and renewal events alone should not be interpreted as product retention."
        )
        st.caption(str(suitability.get("reason", "Retention suitability could not be established.")))
        st.markdown("")
        render_chart_heading("Daily active users", "Unique users with any recorded event by day; this is activity volume, not a retention metric")
        fig = px.area(daily, x="Date", y="Active users")
        fig.update_traces(line=dict(color="#46E1DA", width=2.5), fillcolor="rgba(70,225,218,0.16)", hovertemplate="<b>%{x|%d %b %Y}</b><br>Active users: %{y:,}<extra></extra>")
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Active users")
        st.plotly_chart(_plotly_style(fig, 390), use_container_width=True, config={"displayModeBar": False})
        return

    first = tmp.groupby(user_col)["Date"].min().rename("first_date")
    visits = tmp[[user_col, "Date"]].drop_duplicates().join(first, on=user_col)
    visits["day"] = (visits["Date"] - visits["first_date"]).dt.days
    observation_end = tmp["Date"].max()
    rates = {}
    eligible_counts = {}
    for d in [1, 7, 30]:
        eligible = first[first <= observation_end - pd.Timedelta(days=d)].index
        eligible_counts[d] = len(eligible)
        returned = visits[(visits["day"] == d) & (visits[user_col].isin(eligible))][user_col].nunique()
        rates[d] = returned / max(len(eligible), 1) * 100

    render_kpis([
        ("D1 retention", f"{rates[1]:.1f}%", f"{eligible_counts[1]:,} eligible users"),
        ("D7 retention", f"{rates[7]:.1f}%", f"{eligible_counts[7]:,} eligible users"),
        ("D30 retention", f"{rates[30]:.1f}%", f"{eligible_counts[30]:,} eligible users"),
        ("Events per user", f"{avg_events_per_user:.1f}", "Average event volume per user"),
    ])
    st.markdown("")
    c1, c2 = st.columns([1.4, 1])
    with c1:
        render_chart_heading("Daily active users", "Unique active users by day")
        fig = px.area(daily, x="Date", y="Active users")
        fig.update_traces(line=dict(color="#46E1DA", width=2.5), fillcolor="rgba(70,225,218,0.16)", hovertemplate="<b>%{x|%d %b %Y}</b><br>Active users: %{y:,}<extra></extra>")
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Active users")
        st.plotly_chart(_plotly_style(fig, 380), use_container_width=True, config={"displayModeBar": False})
    with c2:
        render_chart_heading("Retention checkpoints", "Only users old enough to reach each checkpoint are included")
        ret = pd.DataFrame({"Checkpoint": ["D1", "D7", "D30"], "Retention": [rates[1], rates[7], rates[30]]})
        fig = px.bar(ret, x="Checkpoint", y="Retention")
        fig.update_traces(marker_color=_rank_colors(ret["Retention"]), marker_line_width=0, hovertemplate="<b>%{x}</b><br>Retention: %{y:.1f}%<extra></extra>")
        fig.update_yaxes(title_text="Retention")
        fig.update_xaxes(title_text="Checkpoint")
        st.plotly_chart(_plotly_style(fig, 380, y_percent=True), use_container_width=True, config={"displayModeBar": False})



def _plan_column_is_trustworthy(df, plan_col):
    """Be conservative with ambiguous segment/tier fields before calling them plans."""
    if not plan_col or plan_col not in df.columns:
        return False
    name = _normalize_column_name(plan_col)
    tokens = set(name.split("_"))
    if tokens & {"plan", "subscription", "billing", "package"}:
        return True
    # Generic tier/band/membership fields are often customer segments, not products.
    if tokens & {"membership", "band", "segment", "customer"}:
        return False
    return "tier" in tokens

def render_monetization_dashboard(df):
    user_col = st.session_state.user_col
    event_col = st.session_state.event_col
    time_col = st.session_state.timestamp_col
    revenue_col = find_column(df, ["amount_usd", "revenue", "amount", "price", "transaction_value", "order_value"])
    plan_col = find_column(df, ["plan_name", "plan", "subscription_plan", "billing_plan", "tier", "subscription_tier", "package"])
    if plan_col and not _plan_column_is_trustworthy(df, plan_col):
        plan_col = None

    render_dashboard_header(
        "Monetization",
        "Connect product behavior to payments, plans and conversion — without turning the dashboard into a finance warehouse.",
    )

    if not revenue_col:
        st.warning("Monetization dashboard needs a revenue/amount field.")
        return

    tmp = df[[c for c in [user_col, event_col, time_col, revenue_col, plan_col] if c]].copy()
    tmp[time_col] = _safe_datetime(tmp[time_col])
    tmp[revenue_col] = pd.to_numeric(tmp[revenue_col], errors="coerce")
    event_lower = tmp[event_col].astype(str).str.lower()
    revenue_events = _successful_revenue_events(tmp, revenue_col)
    if revenue_events.empty:
        st.warning("Revenue/amount exists, but LENS could not identify a trustworthy successful-payment event. Revenue totals are withheld until the payment mapping is reviewed.")

    total_rev = revenue_events[revenue_col].sum() if len(revenue_events) else 0.0
    paying_users = revenue_events[user_col].nunique()
    arppu = total_rev / max(paying_users, 1)
    paid_event = _paid_event(df)
    paid_rate = None
    if paid_event:
        paid_users = df.loc[df[event_col].astype(str) == paid_event, user_col].nunique()
        paid_rate = paid_users / max(df[user_col].nunique(), 1) * 100

    attempted, failed = _payment_attempt_failure_counts(tmp)
    failure_rate = failed / attempted * 100 if attempted else None
    money_prefix = _money_prefix(revenue_col)

    render_kpis([
        ("Collected revenue", f"{money_prefix}{total_rev:,.0f}", "Successful payment events only; attempts/failures excluded"),
        ("Paying users", f"{paying_users:,}", "Unique users with successful revenue"),
        ("Revenue per payer", f"{money_prefix}{arppu:,.2f}", "Average successful revenue per paying user"),
        ("Paid conversion", f"{paid_rate:.1f}%" if paid_rate is not None else "—", "Share of users reaching paid state"),
        ("Payment failure", f"{failure_rate:.1f}%" if failure_rate is not None else "—", "Failed payment events / attempts"),
    ])

    st.markdown("")
    c1, c2 = st.columns([1.35, 1])
    with c1:
        if len(revenue_events):
            render_chart_heading("Collected revenue over time", "Monthly detected successful revenue")
            monthly = (
                revenue_events.dropna(subset=[time_col])
                .set_index(time_col)
                .resample("MS")[revenue_col]
                .sum()
                .reset_index()
                .rename(columns={time_col: "Month", revenue_col: "Revenue"})
            )
            fig = px.area(monthly, x="Month", y="Revenue")
            fig.update_traces(
                line=dict(color="#FF4FA3", width=2.7),
                fillcolor="rgba(255,79,163,0.15)",
                hovertemplate="<b>%{x|%b %Y}</b><br>Revenue: %{y:,.2f}<extra></extra>",
            )
            fig.update_yaxes(title_text="Revenue")
            fig.update_xaxes(title_text="Month")
            st.plotly_chart(_plotly_style(fig, 380), use_container_width=True, config={"displayModeBar": False})

    with c2:
        if plan_col and revenue_events[plan_col].notna().any():
            render_chart_heading("Revenue by plan", "Which plans contribute collected revenue")
            plans = (
                revenue_events.groupby(plan_col, dropna=True)[revenue_col]
                .sum()
                .reset_index()
                .sort_values(revenue_col, ascending=False)
                .rename(columns={plan_col: "Plan", revenue_col: "Revenue"})
            )
            fig = px.bar(plans, x="Plan", y="Revenue")
            fig.update_traces(
                marker_color=_rank_colors(plans["Revenue"]),
                marker_line_width=0,
                hovertemplate="<b>%{x}</b><br>Revenue: %{y:,.2f}<extra></extra>",
            )
            fig.update_yaxes(title_text="Revenue")
            fig.update_xaxes(title_text="Plan")
            st.plotly_chart(_plotly_style(fig, 380), use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Add a plan/tier field to unlock plan mix and plan-level revenue.")


def render_product_dashboard(df):
    user_col = st.session_state.user_col
    event_col = st.session_state.event_col
    time_col = st.session_state.timestamp_col
    version_col = find_column(df, ["app_version", "version", "release_version", "build_version"])
    device_col = find_column(df, ["device_type", "device", "platform", "os", "operating_system"])
    dimension = version_col or device_col

    render_dashboard_header(
        "Product & Release",
        "Watch release adoption and detect behavior shifts across versions, devices or product surfaces.",
    )

    if not dimension:
        st.warning("Product & Release needs a version, device or platform field.")
        return

    tmp = df[[user_col, event_col, time_col, dimension]].copy()
    tmp[time_col] = _safe_datetime(tmp[time_col])
    first_dim = _user_dimension_at_event(df, dimension, get_event_role("onboarding", df) or get_event_role("entry", df), include_missing=True)
    user_counts = (
        first_dim[dimension]
        .value_counts()
        .rename_axis(nice_label(dimension))
        .reset_index(name="Users")
    )

    activation_event = _activation_event(df)
    activation = None
    if activation_event:
        activated = set(tmp.loc[tmp[event_col].astype(str) == activation_event, user_col].dropna())
        first_dim["Activated"] = first_dim[user_col].isin(activated)
        activation = (
            first_dim.groupby(dimension, dropna=False)["Activated"]
            .mean()
            .mul(100)
            .reset_index()
            .rename(columns={dimension: nice_label(dimension), "Activated": "Activation rate"})
        )

    dim_label = nice_label(dimension)
    ranked_counts = user_counts[user_counts[dim_label] != MISSING_DIMENSION_LABEL].copy()
    largest_variant = str(ranked_counts.iloc[0][dim_label]) if len(ranked_counts) else "—"
    largest_share = ranked_counts.iloc[0]["Users"] / max(user_counts["Users"].sum(), 1) * 100 if len(ranked_counts) else 0

    render_kpis([
        ("Detected dimension", dim_label, "Primary release comparison field"),
        ("Variants", f"{ranked_counts[dim_label].nunique():,}", "Distinct recorded versions or platforms"),
        ("Largest cohort", largest_variant, f"{largest_share:.1f}% of users in the exposure cohort"),
    ])

    missing_variant_users = int(user_counts.loc[user_counts[dim_label] == MISSING_DIMENSION_LABEL, "Users"].sum())

    st.markdown("")
    c1, c2 = st.columns(2)
    with c1:
        render_chart_heading(f"Users by {dim_label.lower()}", "Pre-activation exposure cohort by release context")
        counts = ranked_counts.head(15).copy()
        fig = px.bar(counts, x=dim_label, y="Users")
        fig.update_traces(
            marker_color=_rank_colors(counts["Users"]),
            marker_line_width=0,
            hovertemplate="<b>%{x}</b><br>Users: %{y:,}<extra></extra>",
        )
        fig.update_xaxes(title_text=dim_label)
        fig.update_yaxes(title_text="Users")
        st.plotly_chart(_plotly_style(fig, 390), use_container_width=True, config={"displayModeBar": False})
        if missing_variant_users:
            st.html(
                f"<div class='missing-data-note'>{missing_variant_users:,} users have no recorded {dim_label.lower()} and are excluded from release ranking.</div>"
            )

    with c2:
        if activation is not None:
            render_chart_heading(f"Activation by {dim_label.lower()}", "Activation rate from the same pre-outcome exposure cohorts")
            act = (
                activation[activation[dim_label] != MISSING_DIMENSION_LABEL]
                .sort_values("Activation rate", ascending=False)
                .head(15)
            )
            fig = px.bar(act, x=dim_label, y="Activation rate")
            fig.update_traces(
                marker_color=_rank_colors(act["Activation rate"]),
                marker_line_width=0,
                hovertemplate="<b>%{x}</b><br>Activation: %{y:.1f}%<extra></extra>",
            )
            fig.update_xaxes(title_text=dim_label)
            fig.update_yaxes(title_text="Activation rate")
            st.plotly_chart(_plotly_style(fig, 390, y_percent=True), use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Activation event could not be identified automatically for release comparison.")




def _best_dimension(df):
    for role in ["channel", "plan", "geography", "device", "version"]:
        col = get_optional_column(role)
        if col in df.columns:
            return col
    dims = st.session_state.get("custom_dimensions", []) or []
    return next((c for c in dims if c in df.columns), None)


def render_user_profile_dashboard(df):
    user_col = st.session_state.get("user_col")
    render_dashboard_header("User Profile", "A grain-aware view for one-row-per-user datasets.", eyebrow="USER-LEVEL DATA")
    dim = _best_dimension(df)
    render_kpis([("Rows", f"{len(df):,}", "User records"), ("Unique users", f"{df[user_col].nunique():,}" if user_col in df.columns else f"{len(df):,}", "Detected customer identifiers"), ("Fields", f"{len(df.columns):,}", "Available columns")])
    if dim:
        render_chart_heading(f"Users by {nice_label(dim)}", "Largest available segmentation dimension")
        counts = df[dim].fillna(MISSING_DIMENSION_LABEL).value_counts().head(15).rename_axis(nice_label(dim)).reset_index(name="Users")
        fig = px.bar(counts, x=nice_label(dim), y="Users")
        fig.update_traces(marker_color=_rank_colors(counts["Users"]))
        st.plotly_chart(_plotly_style(fig), use_container_width=True, config={"displayModeBar": False})


def render_segments_dashboard(df):
    dim = _best_dimension(df)
    render_dashboard_header("Segments", "Compare the population using the strongest detected categorical dimension.")
    if not dim:
        st.info("No stable categorical dimension was detected for segmentation.")
        return
    counts = df[dim].fillna(MISSING_DIMENSION_LABEL).value_counts().head(20).rename_axis("Segment").reset_index(name="Rows")
    render_kpis([("Detected dimension", nice_label(dim), "Primary segmentation field"), ("Segments", f"{df[dim].nunique(dropna=True):,}", "Distinct values")])
    fig = px.bar(counts, x="Segment", y="Rows")
    fig.update_traces(marker_color=_rank_colors(counts["Rows"]))
    st.plotly_chart(_plotly_style(fig), use_container_width=True, config={"displayModeBar": False})


def render_user_value_dashboard(df):
    revenue_col = get_optional_column("revenue")
    render_dashboard_header("Value / LTV", "Analyze monetary value without assuming event semantics.", eyebrow="USER-LEVEL DATA")
    if revenue_col not in df.columns:
        st.warning("No monetary field was confirmed.")
        return
    values = pd.to_numeric(df[revenue_col], errors="coerce")
    render_kpis([("Total value", f"${values.sum():,.0f}", nice_label(revenue_col)), ("Average value", f"${values.mean():,.2f}", "Per row/user"), ("Median value", f"${values.median():,.2f}", "Robust central value")])
    dim = _best_dimension(df)
    if dim:
        grouped = df.assign(_value=values).groupby(dim, dropna=False)["_value"].mean().sort_values(ascending=False).head(15).reset_index()
        fig = px.bar(grouped, x=dim, y="_value")
        fig.update_traces(marker_color=_rank_colors(grouped["_value"]))
        fig.update_yaxes(title_text="Average value")
        st.plotly_chart(_plotly_style(fig), use_container_width=True, config={"displayModeBar": False})


def render_churn_dashboard(df):
    churn_col = _find_churn_column(df)
    render_dashboard_header("Churn", "Inspect lifecycle outcomes available directly in a user-level table.", eyebrow="USER-LEVEL DATA")
    if churn_col not in df.columns:
        st.warning("No churn/cancellation/status field was detected.")
        return
    s = df[churn_col]
    counts = s.fillna(MISSING_DIMENSION_LABEL).astype(str).value_counts().head(20).rename_axis("Outcome").reset_index(name="Users")
    render_kpis([("Lifecycle field", nice_label(churn_col), "Detected status/outcome field"), ("Distinct outcomes", f"{s.nunique(dropna=True):,}", "Observed values")])
    fig = px.bar(counts, x="Outcome", y="Users")
    fig.update_traces(marker_color=_rank_colors(counts["Users"]))
    st.plotly_chart(_plotly_style(fig), use_container_width=True, config={"displayModeBar": False})


def _transaction_base(df):
    revenue_col = get_optional_column("revenue")
    user_col = st.session_state.get("user_col")
    time_col = st.session_state.get("timestamp_col")
    return user_col, time_col, revenue_col


def render_transaction_revenue_dashboard(df):
    user_col, time_col, revenue_col = _transaction_base(df)
    render_dashboard_header("Revenue", "Transaction-level monetary performance with no event funnel assumptions.", eyebrow="TRANSACTION DATA")
    if revenue_col not in df.columns:
        st.warning("No amount/revenue field was confirmed.")
        return
    rev = pd.to_numeric(df[revenue_col], errors="coerce")
    render_kpis([("Collected value", f"${rev.sum():,.0f}", nice_label(revenue_col)), ("Average transaction", f"${rev.mean():,.2f}", "Mean transaction value"), ("Median transaction", f"${rev.median():,.2f}", "Median transaction value")])
    if time_col in df.columns:
        tmp = df.assign(_rev=rev, _time=_safe_datetime(df[time_col])).dropna(subset=["_time"])
        monthly = tmp.set_index("_time").resample("MS")["_rev"].sum().reset_index()
        fig = px.area(monthly, x="_time", y="_rev")
        fig.update_traces(line=dict(color="#FF4FA3", width=2.5), fillcolor="rgba(255,79,163,0.14)")
        st.plotly_chart(_plotly_style(fig), use_container_width=True, config={"displayModeBar": False})


def render_transaction_customers_dashboard(df):
    user_col, _, revenue_col = _transaction_base(df)
    render_dashboard_header("Customers", "Measure purchase frequency and repeat behavior from transaction rows.", eyebrow="TRANSACTION DATA")
    if user_col not in df.columns:
        st.warning("No customer identifier was confirmed.")
        return
    freq = df.groupby(user_col).size()
    repeat = (freq >= 2).mean() * 100
    render_kpis([("Customers", f"{freq.size:,}", "Unique customers"), ("Transactions / customer", f"{freq.mean():.2f}", "Average frequency"), ("Repeat customers", f"{repeat:.1f}%", "Customers with 2+ transactions")])
    hist = freq.value_counts().sort_index().head(15).rename_axis("Transactions").reset_index(name="Customers")
    fig = px.bar(hist, x="Transactions", y="Customers")
    fig.update_traces(marker_color=_rank_colors(hist["Customers"]))
    st.plotly_chart(_plotly_style(fig), use_container_width=True, config={"displayModeBar": False})


def render_transaction_orders_dashboard(df):
    _, time_col, _ = _transaction_base(df)
    render_dashboard_header("Transactions", "Track transaction volume and temporal concentration.", eyebrow="TRANSACTION DATA")
    render_kpis([("Transactions", f"{len(df):,}", "Rows in the transaction table"), ("Fields", f"{len(df.columns):,}", "Available transaction attributes")])
    if time_col in df.columns:
        ts = _safe_datetime(df[time_col])
        daily = pd.DataFrame({"Date": ts.dt.floor("D")}).dropna().value_counts("Date").rename("Transactions").reset_index()
        fig = px.area(daily, x="Date", y="Transactions")
        fig.update_traces(line=dict(color="#46E1DA", width=2.5), fillcolor="rgba(70,225,218,0.16)")
        st.plotly_chart(_plotly_style(fig), use_container_width=True, config={"displayModeBar": False})


def render_quality_dashboard(df):
    report = st.session_state.get("data_quality") or build_data_quality_report(df)
    render_dashboard_header("Data Quality", "Understand what LENS trusts, what needs review and why analysis may be limited.", eyebrow="DATA PROFILING")
    render_kpis([("Duplicate rows", f"{report['duplicate_rows']:,}", "Exact duplicates"), ("Issues", f"{len(report['issues']):,}", "Warnings and context")])
    for issue in report["issues"][:12]:
        icon = {"high": "🔴", "warning": "🟠", "context": "◌"}.get(issue["severity"], "•")
        st.markdown(f"{icon} **{issue['title']}** — {issue['detail']}")

# ================================================================
# AUTOMATIC FINDINGS ENGINE
# ================================================================


__all__ = [name for name in globals() if not name.startswith("__")]

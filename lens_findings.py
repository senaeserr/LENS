import re
import pandas as pd
import numpy as np
import streamlit as st
from html import escape

from lens_core import *
from lens_core import (_activation_event, _cohort_floor, _event_matches_any, _funnel_steps, _get_analysis_areas_cached, _lifecycle_coverage_stats, _payment_attempt_failure_counts, _retention_suitability, _safe_datetime, _session_analysis_cache_get, _user_dimension_at_event, _user_dimension_snapshot)
from lens_dashboards import *
from lens_dashboards import (_ordered_funnel_counts)

PRIORITY_ORDER = {"high": 3, "medium": 2, "watch": 1, "context": 0}




def _user_first_dimension(df, dimension_col):
    role_map = st.session_state.get("optional_mappings", {}) or {}
    time_varying = dimension_col in {role_map.get("device"), role_map.get("version")}
    if time_varying:
        # Release/device activation must use a pre-outcome exposure cohort.
        # Onboarding is the best generic activation-opportunity anchor; entry is
        # the fallback when onboarding is not available.
        anchor_event = get_event_role("onboarding", df) or get_event_role("entry", df)
        return _user_dimension_at_event(df, dimension_col, anchor_event, include_missing=False)
    return _user_dimension_snapshot(df, dimension_col, include_missing=False)


def _dimension_activation_table(df, dimension_col):
    activation_event = _activation_event(df)
    if not activation_event or not dimension_col:
        return None
    user_col = st.session_state.user_col
    event_col = st.session_state.event_col
    base = _user_first_dimension(df, dimension_col)
    if base.empty:
        return None
    activated = set(
        df.loc[df[event_col].astype(str) == activation_event, user_col]
        .dropna().astype(str)
    )
    base[user_col] = base[user_col].astype(str)
    base["Activated"] = base[user_col].isin(activated)
    grouped = (
        base.groupby(dimension_col, dropna=False)
        .agg(Users=(user_col, "nunique"), Activation_rate=("Activated", "mean"))
        .reset_index()
    )
    grouped["Activation_rate"] *= 100
    grouped["Share"] = grouped["Users"] / max(grouped["Users"].sum(), 1) * 100
    return grouped


def _overall_activation_rate(df):
    activation_event = _activation_event(df)
    if not activation_event:
        return None
    user_col = st.session_state.user_col
    event_col = st.session_state.event_col
    total = df[user_col].nunique(dropna=True)
    activated = df.loc[df[event_col].astype(str) == activation_event, user_col].nunique(dropna=True)
    return activated / max(total, 1) * 100


def _make_finding(title, category, priority, summary, evidence, next_check, note, area_key, score):
    return {
        "title": title,
        "category": category,
        "priority": priority,
        "summary": summary,
        "evidence": evidence,
        "next_check": next_check,
        "note": note,
        "area_key": area_key,
        "score": float(score),
    }


def detect_automatic_findings(df):
    """Detect cross-metric exceptions using only observable uploaded data."""
    findings = []
    grain = (st.session_state.get("grain_info") or {}).get("grain", "event")
    if grain != "event":
        report = st.session_state.get("data_quality") or build_data_quality_report(df)
        if report.get("issues"):
            findings.append(_make_finding(
                "Data quality needs review", "Data Quality",
                "high" if any(i.get("severity") == "high" for i in report["issues"]) else "medium",
                "LENS found structural or quality conditions that can affect downstream metrics.",
                [i["title"] for i in report["issues"][:3]],
                "Review data quality before treating business differences as product signals.",
                "Quality warnings describe the uploaded table; they are not business-causality claims.",
                "quality" if grain == "ambiguous" else next(iter(get_analysis_areas().keys())), 2.0,
            ))
        return findings
    total_users = max(df[st.session_state.user_col].nunique(dropna=True), 1)
    overall_activation = _overall_activation_rate(df)
    lifecycle = _lifecycle_coverage_stats(df)
    if lifecycle.get("entry_coverage") is not None and lifecycle["entry_coverage"] < SAAS_MVP_ENTRY_COVERAGE_WARN:
        findings.append(_make_finding(
            "Lifecycle coverage is partial", "Data Quality", "context",
            "Not every observed user has an entry event inside the uploaded window, so funnel denominators should be read as observed-entry cohorts rather than all historical users.",
            [f"{lifecycle['entry_coverage']:.1%} of observed users have an entry event"],
            "Use a wider extraction window if you need full-lifecycle conversion rates.",
            "This is a dataset-window limitation, not a product-performance problem.",
            "activation", 0.1,
        ))

    channel_col = find_column(
        df,
        ["acquisition_channel", "marketing_channel", "channel", "source", "traffic_source", "utm_source"],
    )
    version_col = find_column(df, ["app_version", "version", "release_version", "build_version"])
    country_col = find_column(df, ["country", "country_code", "region", "market", "geo"])
    device_col = find_column(df, ["device_type", "device", "platform", "os", "operating_system"])

    acquisition_candidate = None
    if channel_col and overall_activation is not None:
        table = _dimension_activation_table(df, channel_col)
        if table is not None and len(table) >= 2:
            eligible = table[table["Users"] >= _cohort_floor(total_users, share=0.01)].copy()
            if len(eligible):
                eligible["Gap"] = overall_activation - eligible["Activation_rate"]
                eligible["Impact_score"] = eligible["Share"] * eligible["Gap"].clip(lower=0) / 100
                row = eligible.sort_values(["Impact_score", "Share"], ascending=False).iloc[0]
                if row["Gap"] >= 4 and row["Share"] >= 5:
                    acquisition_candidate = {
                        "value": row[channel_col], "share": row["Share"],
                        "rate": row["Activation_rate"], "gap": row["Gap"],
                    }
                    priority = "high" if row["Gap"] >= 8 and row["Share"] >= 15 else "medium"
                    findings.append(_make_finding(
                        "Acquisition quality gap", "Acquisition", priority,
                        f"{row[channel_col]} brings a large share of users, but those users activate well below the product average.",
                        [f"{row['Share']:.1f}% of acquired users", f"{row['Activation_rate']:.1f}% activation", f"{row['Gap']:.1f} pp below overall activation"],
                        "Compare this source by device, release version and signup period.",
                        "High acquisition volume does not imply high downstream quality.",
                        "acquisition", row["Impact_score"] + (1 if priority == "high" else 0),
                    ))

    release_candidate = None
    if version_col and overall_activation is not None:
        table = _dimension_activation_table(df, version_col)
        if table is not None and len(table) >= 2:
            eligible = table[table["Users"] >= _cohort_floor(total_users, share=0.02)].copy()
            if len(eligible):
                eligible["Gap"] = overall_activation - eligible["Activation_rate"]
                eligible["Impact_score"] = eligible["Share"] * eligible["Gap"].clip(lower=0) / 100
                row = eligible.sort_values(["Impact_score", "Share"], ascending=False).iloc[0]
                if row["Gap"] >= 4 and row["Share"] >= 5:
                    release_candidate = {
                        "value": row[version_col], "share": row["Share"],
                        "rate": row["Activation_rate"], "gap": row["Gap"],
                    }
                    priority = "high" if row["Gap"] >= 7 and row["Share"] >= 10 else "medium"
                    findings.append(_make_finding(
                        "Possible release regression", "Product & Release", priority,
                        f"{row[version_col]} represents a meaningful user cohort and is associated with lower activation than the product baseline.",
                        [f"{row['Share']:.1f}% of users", f"{row['Activation_rate']:.1f}% activation", f"{row['Gap']:.1f} pp below overall activation"],
                        "Compare this release against adjacent versions within the same channel and device mix.",
                        "This is a regression suspect, not proof that the release caused the decline.",
                        "product", row["Impact_score"] + (1 if priority == "high" else 0),
                    ))

    if acquisition_candidate and release_candidate and channel_col and version_col and overall_activation is not None:
        user_col = st.session_state.user_col
        event_col = st.session_state.event_col
        channel_users = _user_first_dimension(df, channel_col)
        version_users = _user_first_dimension(df, version_col)
        if not channel_users.empty and not version_users.empty:
            users = channel_users.merge(version_users, on=user_col, how="inner")
            activated = set(
                df.loc[df[event_col].astype(str) == _activation_event(df), user_col].dropna().astype(str)
            )
            users[user_col] = users[user_col].astype(str)
            users["Activated"] = users[user_col].isin(activated)
            mask = (
                users[channel_col].astype(str) == str(acquisition_candidate["value"])
            ) & (
                users[version_col].astype(str) == str(release_candidate["value"])
            )
            combo = users.loc[mask]
            if len(combo) >= _cohort_floor(total_users, share=0.005, floor=80):
                combo_rate = combo["Activated"].mean() * 100
                combo_share = len(combo) / total_users * 100
                combo_gap = overall_activation - combo_rate
                if combo_gap >= 6:
                    findings.append(_make_finding(
                        "Activation weakness is concentrated", "Cross-signal", "high",
                        f"The overlap between {acquisition_candidate['value']} and {release_candidate['value']} underperforms overall activation, linking two dashboard warnings into one investigation.",
                        [f"{combo_share:.1f}% of users in this overlap", f"{combo_rate:.1f}% activation in the overlap", f"{combo_gap:.1f} pp below overall activation"],
                        "Break this overlap down by device and country, then compare before/after the release rollout.",
                        "LENS joined two independent signals. The overlap strengthens the hypothesis but still does not establish causality.",
                        "activation", combo_share * combo_gap / 100 + 2.5,
                    ))

    steps = _funnel_steps(df)
    if len(steps) >= 2:
        event_col = st.session_state.event_col
        user_col = st.session_state.user_col
        time_col = st.session_state.timestamp_col
        funnel_metric = _ordered_funnel_counts(df[[user_col, event_col, time_col]].copy(), user_col, event_col, time_col, steps)
        biggest = None
        if not funnel_metric.empty:
            for i in range(1, len(funnel_metric)):
                previous = max(int(funnel_metric.iloc[i - 1]["Users"]), 1)
                current = int(funnel_metric.iloc[i]["Users"])
                drop = (1 - current / previous) * 100
                if biggest is None or drop > biggest[0]:
                    biggest = (drop, funnel_metric.iloc[i - 1]["Funnel step"], funnel_metric.iloc[i]["Funnel step"], previous, current)
        if biggest and biggest[0] >= 30:
            drop, prev_label, next_label, prev_n, next_n = biggest
            priority = "high" if drop >= 60 else "medium"
            findings.append(_make_finding(
                "Funnel bottleneck detected", "Activation & Funnel", priority,
                f"The largest conversion loss occurs between {prev_label} and {next_label}.",
                [f"{drop:.1f}% step loss", f"{prev_n:,} -> {next_n:,} users", f"{next_n / max(prev_n, 1) * 100:.1f}% step conversion"],
                f"Segment the {prev_label} -> {next_label} step by channel, device, country and release version.",
                "A large drop is a prioritization signal; LENS does not assume the reason for the loss.",
                "activation", drop / 20,
            ))

    event_col = st.session_state.event_col
    time_col = st.session_state.timestamp_col
    attempt_event = get_event_role("payment_attempt", df)
    failure_event = get_event_role("payment_failure", df)
    attempted_mask = _event_matches_any(df[event_col], [attempt_event], r"(?:payment|renewal|charge|transaction).*(?:attempt|probe|submitted)")
    failed_mask = _event_matches_any(df[event_col], [failure_event], r"(?:payment|renewal|charge|transaction).*(?:failed|failure|declined|bounced)")
    attempts, failures = _payment_attempt_failure_counts(df)
    overall_failure = failures / attempts * 100 if attempts else None

    if attempts and overall_failure is not None:
        best_segment = None
        for dim in [country_col, device_col, channel_col]:
            if not dim:
                continue
            attempt_counts = df.loc[attempted_mask].groupby(dim).size().rename("attempts")
            failure_counts = df.loc[failed_mask].groupby(dim).size().rename("failures")
            temp = pd.concat([attempt_counts, failure_counts], axis=1).fillna(0)
            min_attempts = max(40, _cohort_floor(attempts, share=0.01, floor=40))
            temp = temp[temp["attempts"] >= min_attempts]
            if temp.empty:
                continue
            temp["rate"] = temp["failures"] / temp["attempts"] * 100
            temp["gap"] = temp["rate"] - overall_failure
            candidate = temp.sort_values("gap", ascending=False).head(1)
            if len(candidate):
                val = candidate.index[0]
                row = candidate.iloc[0]
                score = row["gap"] * np.log1p(row["attempts"])
                if best_segment is None or score > best_segment["score"]:
                    best_segment = {
                        "dimension": dim, "value": val, "attempts": row["attempts"],
                        "rate": row["rate"], "gap": row["gap"], "score": score,
                    }

        # Time-aware concentration: use conservative monthly windows and require
        # both a jump versus the segment's own history and a gap versus peers in
        # the same month. This reduces false alarms from tiny weekly cohorts.
        best_temporal = None
        payment_rows = df.loc[attempted_mask | failed_mask, [c for c in [event_col, time_col, country_col, device_col, channel_col] if c]].copy()
        if time_col in payment_rows.columns:
            payment_rows[time_col] = _safe_datetime(payment_rows[time_col])
            payment_rows = payment_rows.dropna(subset=[time_col])
            if not payment_rows.empty:
                payment_rows["_period"] = payment_rows[time_col].dt.to_period("M").dt.to_timestamp()
                attempt_regex = r"(?:payment|renewal|charge|transaction).*(?:attempt|probe|submitted)"
                failure_regex = r"(?:payment|renewal|charge|transaction).*(?:failed|failure|declined|bounced)"
                for dim in [country_col, device_col, channel_col]:
                    if not dim or dim not in payment_rows.columns:
                        continue
                    for (val, period), grp in payment_rows.dropna(subset=[dim]).groupby([dim, "_period"]):
                        period_attempts = int(_event_matches_any(grp[event_col], [attempt_event], attempt_regex).sum())
                        period_failures = int(_event_matches_any(grp[event_col], [failure_event], failure_regex).sum())
                        if period_attempts < 100 or period_failures < 8:
                            continue

                        segment_all = payment_rows[payment_rows[dim].astype(str).eq(str(val))]
                        outside = segment_all[segment_all["_period"] != period]
                        out_attempts = int(_event_matches_any(outside[event_col], [attempt_event], attempt_regex).sum())
                        out_failures = int(_event_matches_any(outside[event_col], [failure_event], failure_regex).sum())

                        peers = payment_rows[(payment_rows["_period"] == period) & ~payment_rows[dim].astype(str).eq(str(val))]
                        peer_attempts = int(_event_matches_any(peers[event_col], [attempt_event], attempt_regex).sum())
                        peer_failures = int(_event_matches_any(peers[event_col], [failure_event], failure_regex).sum())
                        if out_attempts < 200 or peer_attempts < 200:
                            continue

                        period_rate = period_failures / period_attempts * 100
                        baseline = out_failures / out_attempts * 100
                        peer_rate = peer_failures / peer_attempts * 100
                        baseline_gap = period_rate - baseline
                        peer_gap = period_rate - peer_rate
                        if baseline_gap < 8 or peer_gap < 8:
                            continue

                        conservative_gap = min(baseline_gap, peer_gap)
                        score = conservative_gap * np.log1p(period_attempts)
                        if best_temporal is None or score > best_temporal["score"]:
                            best_temporal = {
                                "dimension": dim, "value": val, "period": period,
                                "attempts": period_attempts, "failures": period_failures,
                                "rate": period_rate, "baseline": baseline, "peer_rate": peer_rate,
                                "gap": conservative_gap, "score": score,
                            }

        if best_temporal is not None:
            priority = "high" if best_temporal["gap"] >= 15 else "medium"
            period_label = pd.Timestamp(best_temporal["period"]).strftime("%b %Y")
            findings.append(_make_finding(
                "Payment failure spike is time-concentrated", "Monetization", priority,
                f"Payment failures for {best_temporal['value']} within {nice_label(best_temporal['dimension']).lower()} spike during {period_label}.",
                [
                    f"{best_temporal['rate']:.1f}% failure rate in the period",
                    f"{best_temporal['gap']:.1f} pp above both historical baseline and same-period peers",
                    f"{int(best_temporal['attempts']):,} payment attempts",
                ],
                "Inspect provider, payment route and deployment changes during the affected week; separate first-payment from renewal attempts.",
                "This is a temporal concentration signal, not proof of a provider or product defect.",
                "monetization", best_temporal["score"] / 10,
            ))
        elif best_segment and best_segment["gap"] >= 3:
            priority = "high" if best_segment["gap"] >= 8 else "medium"
            findings.append(_make_finding(
                "Payment friction is concentrated", "Monetization", priority,
                f"Payment failures are disproportionately high for {best_segment['value']} within {nice_label(best_segment['dimension']).lower()}.",
                [f"{best_segment['rate']:.1f}% failure rate", f"{best_segment['gap']:.1f} pp above overall", f"{int(best_segment['attempts']):,} payment attempts"],
                "Compare the affected segment over time and separate first-payment from renewal failures.",
                "Failure concentration can reflect provider, geography, payment method or user-mix effects.",
                "monetization", best_segment["score"] / 10,
            ))
        elif overall_failure >= 5:
            findings.append(_make_finding(
                "Payment failures deserve monitoring", "Monetization", "watch",
                "The overall payment-failure rate is high enough to keep on the recurring dashboard.",
                [f"{overall_failure:.1f}% overall payment failure", f"{attempts:,} attempts", f"{failures:,} failures"],
                "Inspect payment failures by market, platform and time period.",
                "This is a monitoring threshold, not evidence of a specific provider issue.",
                "monetization", overall_failure / 10,
            ))


    user_col = st.session_state.user_col
    time_col = st.session_state.timestamp_col
    temp = df[[user_col, time_col]].copy()
    temp[time_col] = _safe_datetime(temp[time_col])
    temp = temp.dropna(subset=[user_col, time_col])
    retention_ok = _retention_suitability(df).get("suitable", False)
    if len(temp) and retention_ok:
        temp["date"] = temp[time_col].dt.floor("D")
        first = temp.groupby(user_col)["date"].min().rename("first")
        visits = temp[[user_col, "date"]].drop_duplicates().join(first, on=user_col)
        visits["day"] = (visits["date"] - visits["first"]).dt.days
        observation_end = temp["date"].max()
        eligible_d30 = first[first <= observation_end - pd.Timedelta(days=30)].index
        returned_d30 = visits[(visits["day"] == 30) & (visits[user_col].isin(eligible_d30))][user_col].nunique()
        d30 = returned_d30 / max(len(eligible_d30), 1) * 100
        if d30 < 15:
            findings.append(_make_finding(
                "Longer-term retention is weak", "Engagement & Retention", "medium",
                "A relatively small share of users return thirty days after first activity.",
                [f"{d30:.1f}% D30 retention"],
                "Compare D30 retention by activation status, acquisition source and release cohort.",
                "Retention benchmarks vary by product category; LENS flags the pattern inside this dataset rather than claiming an external benchmark.",
                "engagement", (15 - d30) / 5,
            ))
        elif d30 >= 20:
            findings.append(_make_finding(
                "Retention is not the strongest top-line alarm", "Engagement & Retention", "context",
                "A meaningful share of users still return at D30, so activation and monetization gaps may deserve earlier investigation.",
                [f"{d30:.1f}% D30 retention"],
                "Compare retention for activated vs non-activated users before treating retention as the root problem.",
                "This is contextual prioritization, not a claim that retention is healthy for every product type.",
                "engagement", 0.2,
            ))

    findings.sort(key=lambda x: (PRIORITY_ORDER[x["priority"]], x["score"]), reverse=True)
    return findings


def _get_automatic_findings_cached(df):
    return _session_analysis_cache_get("automatic_findings", lambda: detect_automatic_findings(df))


def _open_finding_evidence(area_key):
    """Switch the evidence dashboard without reloading the Streamlit session."""
    st.session_state.analysis_area = area_key
    st.session_state.scroll_to_evidence = True


def render_findings_overview(df):
    """Render elegant, fully clickable finding cards without URL navigation."""
    findings = _get_automatic_findings_cached(df)
    areas = _get_analysis_areas_cached()

    investigation = []
    for item in findings:
        if item["area_key"] not in investigation and item["priority"] != "context":
            investigation.append(item["area_key"])

    path = (
        " → ".join(areas[key]["label"] for key in investigation[:4])
        if investigation
        else "Explore dashboards"
    )

    high_count = sum(f["priority"] == "high" for f in findings)
    medium_count = sum(f["priority"] == "medium" for f in findings)

    st.html(
        f"""
<div class="findings-shell">
    <div class="findings-head">
        <div>
            <div class="findings-kicker">AUTOMATIC SIGNALS</div>
            <div class="findings-title">What LENS found</div>
            <div class="findings-copy">LENS connected recurring metrics across acquisition, activation, retention, monetization and product releases — so you do not have to compare every chart manually.</div>
        </div>
        <div class="findings-scorebox">
            <div class="findings-score">{len(findings)}</div>
            <div class="findings-score-label">signals</div>
            <div class="findings-score-meta">{high_count} high · {medium_count} medium</div>
        </div>
    </div>

    <div class="findings-path">
        <span>Suggested investigation path</span>
        <strong>{escape(path)}</strong>
    </div>
</div>
"""
    )

    if findings:
        visible = findings[:5]
        rows = [visible[i:i + 2] for i in range(0, len(visible), 2)]

        for row_index, row in enumerate(rows):
            cols = st.columns(2)
            for col_index, finding in enumerate(row):
                idx = row_index * 2 + col_index
                area_key = finding["area_key"]
                area_label = areas.get(area_key, {}).get("label", finding["category"])
                priority = finding["priority"]

                evidence_html = "".join(
                    f'<span class="finding-evidence-chip">{escape(str(item))}</span>'
                    for item in finding["evidence"]
                )
                note_html = (
                    f'<div class="finding-note">Note · {escape(finding["note"])}</div>'
                    if finding.get("note")
                    else ""
                )

                with cols[col_index]:
                    with st.container(key=f"finding_click_{idx}"):
                        st.html(
                            f"""
<article class="finding-card priority-{escape(priority)}">
    <div class="finding-topline">
        <span class="finding-priority">{escape(priority)}</span>
        <span class="finding-category">{escape(finding["category"])}</span>
    </div>
    <div class="finding-title">{escape(finding["title"])}</div>
    <div class="finding-summary">{escape(finding["summary"])}</div>
    <div class="finding-evidence">{evidence_html}</div>
    <div class="finding-next"><b>Next check →</b> {escape(finding["next_check"])}</div>
    {note_html}
    <div class="finding-open">View {escape(area_label)} evidence ↘</div>
</article>
"""
                        )
                        clicked = st.button(
                            f"Open {area_label} evidence",
                            key=f"finding_button_{idx}",
                            use_container_width=True,
                        )
                        if clicked:
                            st.session_state.analysis_area = area_key
                            st.session_state.scroll_to_evidence = True
                            st.rerun()
    else:
        st.html(
            """
<div class="finding-card priority-context finding-empty-card">
    <div class="finding-topline"><span class="finding-priority">context</span></div>
    <div class="finding-title">No strong exception detected yet</div>
    <div class="finding-summary">The current first-pass rules did not find a material cross-metric anomaly. Explore the recurring dashboards or ask a more specific question.</div>
</div>
"""
        )

    st.html(
        """
<div class="findings-method-note">
    <strong>How to read this:</strong> LENS ranks observable associations and exceptions from the uploaded data. A detected signal is a hypothesis to investigate — not proof of causality. Click a finding to open its supporting dashboard.
</div>
"""
    )

    return findings


def render_analysis_dashboard(area_key, df):
    grain = (st.session_state.get("grain_info") or {}).get("grain", "event")
    if grain == "user":
        if area_key == "profile": render_user_profile_dashboard(df)
        elif area_key == "segments": render_segments_dashboard(df)
        elif area_key == "acquisition": render_segments_dashboard(df)
        elif area_key == "value": render_user_value_dashboard(df)
        elif area_key == "churn": render_churn_dashboard(df)
        return
    if grain == "transaction":
        if area_key == "revenue": render_transaction_revenue_dashboard(df)
        elif area_key == "customers": render_transaction_customers_dashboard(df)
        elif area_key == "orders": render_transaction_orders_dashboard(df)
        elif area_key == "segments": render_segments_dashboard(df)
        elif area_key == "trend": render_transaction_orders_dashboard(df)
        return
    if grain == "ambiguous":
        render_quality_dashboard(df)
        return
    if area_key == "acquisition": render_acquisition_dashboard(df)
    elif area_key == "activation": render_activation_dashboard(df)
    elif area_key == "engagement": render_engagement_dashboard(df)
    elif area_key == "monetization": render_monetization_dashboard(df)
    elif area_key == "product": render_product_dashboard(df)


# ================================================================

__all__ = [name for name in globals() if not name.startswith("__")]

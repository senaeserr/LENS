import re
from io import BytesIO
import pandas as pd
import numpy as np
import streamlit as st
from html import escape

try:
    from schema_mapper import detect_schema, sanitize_duplicate_columns
except ImportError:
    from schema_mapper import detect_schema

    def sanitize_duplicate_columns(df):
        """Compatibility fallback for older schema_mapper.py files."""
        if df is None or not getattr(df.columns, "duplicated", lambda: [])().any():
            return df
        out = df.copy(deep=False)
        seen = {}
        names = []
        for raw in out.columns:
            base = str(raw)
            seen[base] = seen.get(base, 0) + 1
            names.append(base if seen[base] == 1 else f"{base}__{seen[base]}")
        out.columns = names
        return out

DEFAULTS = {
    "workflow_step": "upload",
    "workspace_page": "summary",
    "dataset_df": None,
    "dataset_name": None,
    "user_col": None,
    "event_col": None,
    "timestamp_col": None,
    "mapping_edit_mode": False,
    "confirm_dataset_change": False,
    "analysis_area": "activation",
    "scroll_to_evidence": False,
    "optional_mappings": {},
    "optional_mapping_edit_mode": False,
    "custom_dimensions": [],
    "event_role_mappings": {},
    "event_role_confidence": {},
    "data_profile": {},
    "grain_info": {},
    "data_quality": {},
    "semantic_model": {},
    "analysis_cache": {},
    "timestamp_invalid_count": None,
    "data_quality_finalized": False,
    "inference_sample_rows": 0,
    "upload_size_mb": 0.0,
    "large_dataset_warning": False,
}

def _fresh_default(value):
    """Return a per-session copy for mutable defaults."""
    if isinstance(value, dict):
        return value.copy()
    if isinstance(value, list):
        return value.copy()
    if isinstance(value, set):
        return value.copy()
    return value


def ensure_session_defaults():
    """Initialize required Streamlit session keys for the current session.

    This must be called from the rerun entry path (app.py / lens_ui.py), not only
    at module import time, because Python caches imported modules across multiple
    Streamlit user sessions in the same server process.
    """
    for key, default in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = _fresh_default(default)

# ================================================================
# SAAS MVP GUARDRAILS
# ================================================================

SAAS_MVP_MIN_COHORT = 100
SAAS_MVP_MIN_COHORT_SHARE = 0.01
SAAS_MVP_ENTRY_COVERAGE_WARN = 0.80
SAAS_MVP_FIRST_EVENT_ENTRY_WARN = 0.65

# Performance: structure inference uses a bounded representative sample.
INFERENCE_SAMPLE_MAX_ROWS = 30000



def _cohort_floor(total_users, share=SAAS_MVP_MIN_COHORT_SHARE, floor=SAAS_MVP_MIN_COHORT):
    return max(int(floor), int(max(total_users, 1) * float(share)))


# ================================================================
# HELPERS
# ================================================================

def reset_dataset():
    for key, default in DEFAULTS.items():
        st.session_state[key] = _fresh_default(default)

    for key in [
        "dataset_uploader",
        "user_guess",
        "event_guess",
        "time_guess",
    ]:
        st.session_state.pop(key, None)

def _analysis_cache_signature():
    """Stable key for expensive analysis results within the current dataset/mapping."""
    grain = (st.session_state.get("grain_info") or {}).get("grain", "ambiguous")
    optional = tuple(sorted((k, str(v)) for k, v in (st.session_state.get("optional_mappings") or {}).items()))
    event_roles = tuple(sorted((k, str(v)) for k, v in (st.session_state.get("event_role_mappings") or {}).items()))
    return (
        str(st.session_state.get("dataset_name") or ""),
        int(len(st.session_state.dataset_df)) if st.session_state.get("dataset_df") is not None else 0,
        str(st.session_state.get("user_col") or ""),
        str(st.session_state.get("event_col") or ""),
        str(st.session_state.get("timestamp_col") or ""),
        grain,
        optional,
        event_roles,
    )


def _session_analysis_cache_get(key, builder):
    """Memoize expensive deterministic calculations for the active dataset session."""
    cache = st.session_state.setdefault("analysis_cache", {})
    signature = _analysis_cache_signature()
    cache_key = (signature, key)
    if cache_key not in cache:
        cache[cache_key] = builder()
        # Keep the session cache bounded if many mappings are reviewed.
        if len(cache) > 40:
            current_items = {k: v for k, v in cache.items() if k[0] == signature}
            st.session_state.analysis_cache = current_items
            cache = st.session_state.analysis_cache
    return cache[cache_key]


def _invalidate_analysis_cache():
    st.session_state.analysis_cache = {}


def _safe_datetime(series):
    """Parse heterogeneous timestamp strings without losing valid rows.

    Pandas can infer one format for an entire Series and coerce otherwise-valid
    mixed-precision timestamps to NaT. `format="mixed"` handles row-wise formats
    on modern pandas; the fallback keeps compatibility with older versions.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        return pd.to_datetime(series, errors="coerce")


def _numeric_ratio(series):
    if len(series) == 0:
        return 0.0
    return pd.to_numeric(series, errors="coerce").notna().mean()


def _inference_sample(df, max_rows=INFERENCE_SAMPLE_MAX_ROWS):
    """Deterministic representative sample used only for mapping/inference."""
    if len(df) <= max_rows:
        return df
    positions = np.linspace(0, len(df) - 1, max_rows, dtype=np.int64)
    return df.iloc[positions]


def _initial_quality_report():
    """Cheap placeholder; the exact full scan runs once after confirmation."""
    return {
        "health_score": None,
        "issues": [],
        "duplicate_rows": None,
        "timestamp_parse_rate": None,
        "pending_full_scan": True,
    }


def _finalize_dataset_for_workspace(df):
    """Parse once and run full quality checks once after mapping confirmation."""
    if st.session_state.get("data_quality_finalized"):
        return

    time_col = st.session_state.get("timestamp_col")
    if time_col in df.columns:
        raw = df[time_col]
        if pd.api.types.is_datetime64_any_dtype(raw):
            invalid = int(st.session_state.get("timestamp_invalid_count") or 0)
        else:
            raw_missing = int(raw.isna().sum())
            parsed = _safe_datetime(raw)
            invalid = max(int(parsed.isna().sum()) - raw_missing, 0)
            df[time_col] = parsed
        st.session_state.timestamp_invalid_count = invalid

    profile = st.session_state.get("data_profile") or {}
    profile["rows"] = int(len(df))
    profile["duplicate_rows"] = int(df.duplicated().sum())
    st.session_state.data_profile = profile

    st.session_state.data_quality = build_data_quality_report(df)
    st.session_state.data_quality["pending_full_scan"] = False
    st.session_state.data_quality_finalized = True
    _invalidate_analysis_cache()


def profile_dataset(df):
    """Create a lightweight, deterministic profile before any analytics are chosen."""
    profile = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "column_profiles": {},
    }
    for col in df.columns:
        s = df[col]
        non_null = s.dropna()
        n = max(len(non_null), 1)
        nunique = int(non_null.nunique(dropna=True)) if len(non_null) else 0
        sample = non_null.astype(str).head(300)
        parsed = _safe_datetime(sample) if len(sample) else pd.Series(dtype="datetime64[ns]")
        profile["column_profiles"][col] = {
            "null_rate": float(s.isna().mean()),
            "nunique": nunique,
            "unique_ratio": float(nunique / n),
            "numeric_ratio": float(_numeric_ratio(non_null)) if len(non_null) else 0.0,
            "datetime_ratio": float(parsed.notna().mean()) if len(sample) else 0.0,
            "dtype": str(s.dtype),
        }
    return profile


def _name_tokens(value):
    return set(_normalize_column_name(value).split("_"))


def _looks_like_transaction_id(col, stats):
    tokens = _name_tokens(col)
    return (
        bool(tokens & {"transaction", "order", "purchase", "invoice", "payment", "receipt", "txn"})
        and stats.get("unique_ratio", 0) >= 0.80
    )


def detect_data_grain(df, schema_results=None, profile=None):
    """Infer whether rows represent events, users, transactions, or remain ambiguous.

    The detector deliberately exposes evidence and confidence. Low confidence is allowed;
    LENS should ask for review rather than force an event model onto every CSV.
    """
    profile = profile or profile_dataset(df)
    schema_results = schema_results or detect_schema(df)
    cps = profile["column_profiles"]

    user_guess = schema_results.get("user_id", {})
    event_guess = schema_results.get("event_name", {})
    time_guess = schema_results.get("event_timestamp", {})
    user_col = user_guess.get("best_match")
    event_col = event_guess.get("best_match")
    time_col = time_guess.get("best_match")

    user_conf = float(user_guess.get("confidence", 0) or 0)
    event_conf = float(event_guess.get("confidence", 0) or 0)
    time_conf = float(time_guess.get("confidence", 0) or 0)
    user_unique = cps.get(user_col, {}).get("unique_ratio", 1.0)
    event_unique = cps.get(event_col, {}).get("unique_ratio", 1.0)

    event_score = 0.0
    user_score = 0.0
    transaction_score = 0.0
    evidence = []

    # Event evidence: repeated users + event vocabulary + time.
    if user_conf >= 55:
        event_score += 18
    if event_conf >= 55:
        event_score += 30
    if time_conf >= 60:
        event_score += 22
    if user_unique < 0.90:
        event_score += 16
    if event_unique < 0.25:
        event_score += 14

    # User-level evidence: one row per user and no convincing event vocabulary.
    if user_conf >= 55:
        user_score += 30
    if user_unique >= 0.95:
        user_score += 45
    elif user_unique >= 0.85:
        user_score += 22
    if event_conf < 55 or event_unique > 0.50:
        user_score += 20

    # Transaction evidence: repeated customer + timestamp + amount + transaction-like key.
    if user_conf >= 50 and user_unique < 0.95:
        transaction_score += 20
    if time_conf >= 55:
        transaction_score += 20
    amount_candidates = []
    txn_ids = []
    for col, stats in cps.items():
        name = _normalize_column_name(col)
        if stats.get("numeric_ratio", 0) >= 0.85 and any(k in name for k in ["amount", "revenue", "price", "value", "sales", "total"]):
            amount_candidates.append(col)
        if _looks_like_transaction_id(col, stats):
            txn_ids.append(col)
    if amount_candidates:
        transaction_score += 28
    if txn_ids:
        transaction_score += 28
    if event_conf >= 70 and event_unique < 0.20:
        transaction_score -= 18

    scores = {"event": event_score, "user": user_score, "transaction": transaction_score}
    grain = max(scores, key=scores.get)
    best = scores[grain]
    runner_up = sorted(scores.values(), reverse=True)[1]
    margin = best - runner_up

    if best < 55 or margin < 8:
        grain = "ambiguous"
        confidence = min(0.74, max(0.35, best / 100))
    else:
        confidence = min(0.98, 0.55 + best / 220 + min(margin, 30) / 200)

    if user_col:
        evidence.append(f"User field: {user_col}")
    if grain == "event":
        evidence += [
            f"Event field: {event_col}",
            f"Time field: {time_col}",
        ]
    elif grain == "user":
        evidence += ["Rows look close to one-record-per-user."]
    elif grain == "transaction":
        if amount_candidates:
            evidence.append(f"Amount-like field: {amount_candidates[0]}")
        if txn_ids:
            evidence.append(f"Transaction-like identifier: {txn_ids[0]}")
    else:
        evidence.append("Signals conflict; manual review is safer than forcing a grain.")

    return {
        "grain": grain,
        "confidence": round(float(confidence), 2),
        "scores": {k: round(float(v), 1) for k, v in scores.items()},
        "evidence": evidence,
        "amount_candidates": amount_candidates,
        "transaction_id_candidates": txn_ids,
    }


def _confidence_status(score, high=80, review=60):
    score = float(score or 0)
    if score >= high:
        return "High confidence"
    if score >= review:
        return "Review suggested"
    return "Low confidence"


def _lifecycle_coverage_stats(df):
    """Measure how complete the observable SaaS lifecycle is inside the file.

    This is intentionally conservative. A dataset can start mid-lifecycle, so the
    first observed event is not automatically treated as a real signup.
    """
    user_col = st.session_state.get("user_col")
    event_col = st.session_state.get("event_col")
    time_col = st.session_state.get("timestamp_col")
    entry_event = get_event_role("entry", df) if st.session_state.get("event_role_mappings") else None
    if not entry_event or not all(c in df.columns for c in [user_col, event_col, time_col]):
        return {"entry_event": entry_event, "total_users": 0, "entry_users": 0, "entry_coverage": None, "first_event_entry_share": None}

    tmp = df[[user_col, event_col, time_col]].dropna(subset=[user_col, event_col]).copy()
    tmp[time_col] = _safe_datetime(tmp[time_col])
    tmp = tmp.dropna(subset=[time_col]).sort_values([user_col, time_col])
    if tmp.empty:
        return {"entry_event": entry_event, "total_users": 0, "entry_users": 0, "entry_coverage": None, "first_event_entry_share": None}

    total_users = int(tmp[user_col].nunique())
    entry_users = int(tmp.loc[tmp[event_col].astype(str).eq(str(entry_event)), user_col].nunique())
    first_events = tmp.drop_duplicates(user_col, keep="first")
    first_share = float(first_events[event_col].astype(str).eq(str(entry_event)).mean()) if len(first_events) else None
    return {
        "entry_event": entry_event,
        "total_users": total_users,
        "entry_users": entry_users,
        "entry_coverage": entry_users / max(total_users, 1),
        "first_event_entry_share": first_share,
    }


def _event_matches_any(series, exact_values=None, regex=None):
    text = series.astype(str)
    mask = pd.Series(False, index=series.index)
    for value in exact_values or []:
        if value:
            mask |= text.eq(str(value))
    if regex:
        mask |= text.str.contains(regex, case=False, regex=True, na=False)
    return mask


def _dedupe_event_rows(df, subset):
    subset = [c for c in subset if c in df.columns]
    return df.drop_duplicates(subset=subset, keep="first") if subset else df.drop_duplicates()


def _successful_revenue_events(df, revenue_col):
    """Return only observable successful-capture events, deduplicated.

    Attempts, failures and conversion markers are deliberately not counted as
    revenue when a payment-success event exists. Repeated renewals remain valid
    revenue because their timestamps differ.
    """
    user_col = st.session_state.get("user_col")
    event_col = st.session_state.get("event_col")
    time_col = st.session_state.get("timestamp_col")
    if revenue_col not in df.columns or event_col not in df.columns:
        return pd.DataFrame(columns=df.columns)
    work = df.copy()
    work[revenue_col] = pd.to_numeric(work[revenue_col], errors="coerce")
    success_role = get_event_role("payment_success", df)
    success_regex = r"(?:payment|renewal|charge|transaction).*(?:succeeded|success|cleared|captured)|(?:succeeded|success|cleared|captured).*(?:payment|renewal|charge|transaction)"
    mask = _event_matches_any(work[event_col], [success_role], success_regex)

    if not mask.any():
        conversion_role = get_event_role("conversion", df)
        if conversion_role:
            mask = work[event_col].astype(str).eq(str(conversion_role))
        else:
            mask = pd.Series(False, index=work.index)

    out = work.loc[mask & work[revenue_col].notna()].copy()
    return _dedupe_event_rows(out, [user_col, event_col, time_col, revenue_col])


def _payment_attempt_failure_counts(df):
    event_col = st.session_state.get("event_col")
    user_col = st.session_state.get("user_col")
    time_col = st.session_state.get("timestamp_col")
    if event_col not in df.columns:
        return 0, 0
    attempt_role = get_event_role("payment_attempt", df)
    failure_role = get_event_role("payment_failure", df)
    attempts = _event_matches_any(df[event_col], [attempt_role], r"(?:payment|renewal|charge|transaction).*(?:attempt|probe|submitted)")
    failures = _event_matches_any(df[event_col], [failure_role], r"(?:payment|renewal|charge|transaction).*(?:failed|failure|declined|bounced)")
    dedup_cols = [user_col, event_col, time_col]
    attempt_n = len(_dedupe_event_rows(df.loc[attempts], dedup_cols))
    failure_n = len(_dedupe_event_rows(df.loc[failures], dedup_cols))
    return int(attempt_n), int(failure_n)


def _money_prefix(revenue_col):
    name = str(revenue_col or "").lower()
    if "usd" in name or "dollar" in name:
        return "$"
    if "eur" in name:
        return "€"
    if "gbp" in name or "pound" in name:
        return "£"
    return ""


def build_data_quality_report(df):
    profile = st.session_state.get("data_profile") or profile_dataset(df)
    grain = (st.session_state.get("grain_info") or {}).get("grain", "ambiguous")
    issues = []

    dup = int(profile.get("duplicate_rows", 0))
    if dup:
        issues.append({"severity": "warning", "title": "Duplicate rows", "detail": f"{dup:,} exact duplicate rows detected."})

    user_col = st.session_state.get("user_col")
    event_col = st.session_state.get("event_col")
    time_col = st.session_state.get("timestamp_col")

    if user_col in df.columns:
        rate = float(df[user_col].isna().mean())
        if rate > 0:
            sev = "high" if rate >= 0.05 else "warning"
            issues.append({"severity": sev, "title": "Missing user identifiers", "detail": f"{rate:.1%} of rows have no user identifier."})

    if grain == "event" and event_col in df.columns:
        rate = float(df[event_col].isna().mean())
        if rate > 0:
            issues.append({"severity": "warning", "title": "Missing event names", "detail": f"{rate:.1%} of rows have no event value."})

    if time_col in df.columns:
        parsed = _safe_datetime(df[time_col])
        parse_rate = float(parsed.notna().mean())
        stored_invalid = st.session_state.get("timestamp_invalid_count")
        if stored_invalid is not None and pd.api.types.is_datetime64_any_dtype(df[time_col]):
            invalid_timestamps = int(stored_invalid)
        else:
            invalid_timestamps = int(parsed.isna().sum() - df[time_col].isna().sum())
        if parse_rate < 0.995:
            sev = "high" if parse_rate < 0.95 else "warning"
            issues.append({"severity": sev, "title": "Timestamp parsing", "detail": f"{invalid_timestamps:,} non-empty timestamp value(s) could not be parsed."})
        elif invalid_timestamps > 0:
            issues.append({"severity": "context", "title": "Timestamp parsing", "detail": f"{invalid_timestamps:,} non-empty timestamp value(s) could not be parsed; valid rows remain usable."})
    else:
        parse_rate = None

    revenue_col = get_optional_column("revenue") if st.session_state.get("optional_mappings") else None
    if revenue_col in df.columns:
        revenue = pd.to_numeric(df[revenue_col], errors="coerce")
        negatives = int((revenue < 0).sum())
        if negatives:
            issues.append({"severity": "warning", "title": "Negative amounts", "detail": f"{negatives:,} negative values found in {revenue_col}; verify whether these are refunds/credits."})

    for col, stats in profile.get("column_profiles", {}).items():
        if stats.get("null_rate", 0) >= 0.50:
            issues.append({"severity": "context", "title": f"Sparse field: {col}", "detail": f"{stats['null_rate']:.1%} null values."})

    # Partial-lifecycle awareness: the file may begin after many users actually signed up.
    if grain == "event" and st.session_state.get("event_role_mappings"):
        lifecycle = _lifecycle_coverage_stats(df)
        entry_cov = lifecycle.get("entry_coverage")
        first_share = lifecycle.get("first_event_entry_share")
        if entry_cov is not None and (entry_cov < SAAS_MVP_ENTRY_COVERAGE_WARN or (first_share is not None and first_share < SAAS_MVP_FIRST_EVENT_ENTRY_WARN)):
            issues.append({
                "severity": "warning",
                "title": "Partial lifecycle coverage",
                "detail": "A meaningful share of users do not have a clearly observed entry event in this file. Funnel rates remain valid for observed entrants, but first-event should not be interpreted as signup for every user.",
            })

    # Detect extreme event concentration that can distort volume-based metrics.
    if grain == "event" and user_col in df.columns:
        counts = df[user_col].dropna().value_counts()
        if len(counts) >= 20:
            top_share = float(counts.iloc[0] / max(counts.sum(), 1))
            median_events = float(counts.median())
            if top_share >= 0.05 or (median_events > 0 and counts.iloc[0] >= median_events * 100):
                issues.append({
                    "severity": "warning",
                    "title": "Extreme event concentration",
                    "detail": "A very small number of users generate unusually large event volume. User-level rates are safer than raw event-volume comparisons for this dataset.",
                })

    health = 100
    for issue in issues:
        health -= {"high": 18, "warning": 7, "context": 2}.get(issue["severity"], 0)
    health = max(0, min(100, health))

    return {
        "health_score": health,
        "issues": issues,
        "duplicate_rows": dup,
        "timestamp_parse_rate": parse_rate,
    }


def build_semantic_model(df):
    return {
        "grain": (st.session_state.get("grain_info") or {}).get("grain", "ambiguous"),
        "core": {
            "user": st.session_state.get("user_col"),
            "event": st.session_state.get("event_col"),
            "timestamp": st.session_state.get("timestamp_col"),
        },
        "dimensions": dict(st.session_state.get("optional_mappings", {}) or {}),
        "custom_dimensions": list(st.session_state.get("custom_dimensions", []) or []),
        "events": dict(st.session_state.get("event_role_mappings", {}) or {}),
    }


def get_dataset_summary():
    df = st.session_state.dataset_df
    grain = (st.session_state.get("grain_info") or {}).get("grain", "event")
    user_col = st.session_state.get("user_col")
    event_col = st.session_state.get("event_col")
    time_col = st.session_state.get("timestamp_col")

    coverage = "Unknown"
    if time_col in df.columns:
        valid_dates = _safe_datetime(df[time_col]).dropna()
        if len(valid_dates):
            coverage = f"{valid_dates.min().strftime('%b %Y')} → {valid_dates.max().strftime('%b %Y')}"

    users = df[user_col].nunique(dropna=True) if user_col in df.columns else len(df)
    event_types = df[event_col].nunique(dropna=True) if event_col in df.columns else None

    return {
        "rows": len(df),
        "events": len(df) if grain == "event" else None,
        "users": int(users),
        "event_types": int(event_types) if event_types is not None else None,
        "coverage": coverage,
        "grain": grain,
    }


OPTIONAL_ROLE_ALIASES = {
    "channel": [
        "acquisition_channel", "marketing_channel", "channel", "source",
        "traffic_source", "utm_source", "campaign_source", "acquisition_source", "entry_origin", "origin", "referrer",
        "acq_src", "acq_source", "acquisition_src",
    ],
    "device": [
        "device_type", "device", "platform", "os", "operating_system",
        "client_platform", "device_category", "client_surface", "surface",
    ],
    "geography": [
        "country", "country_code", "region", "market", "geo", "geography",
        "location", "territory", "geo_zone", "country_iso",
    ],
    "version": [
        "app_version", "version", "release_version", "build_version",
        "application_version", "release", "build", "build_tag", "client_version",
    ],
    "revenue": [
        "amount_usd", "revenue", "amount", "price", "transaction_value",
        "order_value", "revenue_usd", "payment_amount", "value", "cash_value", "sales_amount", "gmv", "ltv",
    ],
    "plan": [
        "plan_name", "plan", "subscription_plan", "billing_plan", "tier",
        "subscription_tier", "package",
    ],
    "experiment": [
        "experiment", "experiment_id", "test_id", "ab_test", "variant",
        "experiment_variant", "ab_variant", "test_group",
    ],
    "session": [
        "session_id", "session", "visit_id", "visit", "session_key", "journey_token", "journey_id",
    ],
}

OPTIONAL_ROLE_LABELS = {
    "channel": "Acquisition source",
    "device": "Device / platform",
    "geography": "Geography",
    "version": "App / product version",
    "revenue": "Revenue / amount",
    "plan": "Plan / subscription",
    "experiment": "Experiment / variant",
    "session": "Session",
}



EVENT_ROLE_LABELS = {
    "entry": "Entry / signup",
    "onboarding": "Onboarding",
    "activation": "Activation / core value",
    "engagement": "Engagement",
    "conversion": "Conversion / paid",
    "churn": "Churn / cancel",
    "payment_attempt": "Payment attempt",
    "payment_success": "Payment success",
    "payment_failure": "Payment failure",
}

EVENT_ROLE_ALIASES = {
    "entry": ["signup", "sign_up", "register", "registration", "account_created", "user_created", "session_started", "identity_opened", "account_opened", "member_created"],
    "onboarding": ["onboarding", "welcome_completed", "profile_completed", "workspace_created", "workspace_seeded", "setup_completed", "setup_done", "project_initialized"],
    "activation": ["first_core_action", "core_action", "activated", "activation", "first_value", "first_project", "first_report", "first_document", "first_artifact", "artifact_live", "first_artifact_live", "first_workspace_action", "workspace_first_action", "created", "generated", "exported", "uploaded"],
    "engagement": ["feature_used", "project_created", "document_created", "report_created", "share_clicked", "invite_sent", "search", "viewed", "opened"],
    "conversion": ["subscription_started", "upgrade", "upgraded", "purchase_completed", "purchase", "checkout_completed", "paid", "plan_started", "seat_converted", "converted"],
    "churn": ["cancel", "cancelled", "canceled", "churn", "subscription_ended", "downgrade", "access_closed", "account_closed"],
    "payment_attempt": ["payment_attempted", "payment_attempt", "checkout_started", "transaction_attempted", "charge_probe", "charge_attempted"],
    "payment_success": ["payment_succeeded", "payment_success", "renewal_payment_succeeded", "purchase_completed", "transaction_completed", "charge_cleared", "charge_succeeded"],
    "payment_failure": ["payment_failed", "payment_failure", "renewal_payment_failed", "transaction_failed", "charge_bounced", "charge_declined"],
}


def _normalize_event_name(value):
    return str(value).lower().strip().replace(" ", "_").replace("-", "_")


def _event_behavior_stats(df):
    """Lightweight event context used when names alone are not enough."""
    user_col = st.session_state.user_col
    event_col = st.session_state.event_col
    time_col = st.session_state.timestamp_col
    if not user_col or not event_col or not time_col:
        return pd.DataFrame()
    tmp = df[[user_col, event_col, time_col]].dropna(subset=[user_col, event_col]).copy()
    tmp[time_col] = _safe_datetime(tmp[time_col])
    tmp = tmp.dropna(subset=[time_col])
    if tmp.empty:
        return pd.DataFrame()
    first_user = tmp.groupby(user_col)[time_col].min().rename("user_first")
    tmp = tmp.join(first_user, on=user_col)
    tmp["hours_from_entry"] = (tmp[time_col] - tmp["user_first"]).dt.total_seconds() / 3600
    total_users = max(tmp[user_col].nunique(), 1)
    stats = (tmp.groupby(event_col)
        .agg(users=(user_col, "nunique"), median_hours=("hours_from_entry", "median"), events=(user_col, "size"))
        .reset_index())
    stats["user_share"] = stats["users"] / total_users
    stats["events_per_user"] = stats["events"] / stats["users"].clip(lower=1)
    return stats


def detect_event_roles(df):
    """Infer semantic event roles from names plus observed timing/reach.

    The result is intentionally reviewable: LENS suggests roles; the user can override them.
    """
    event_col = st.session_state.event_col
    values = [str(v) for v in df[event_col].dropna().astype(str).unique()]
    normalized = {v: _normalize_event_name(v) for v in values}
    stats = _event_behavior_stats(df)
    stat_lookup = {} if stats.empty else stats.set_index(event_col).to_dict("index")

    roles, confidence = {}, {}
    used = set()
    for role, aliases in EVENT_ROLE_ALIASES.items():
        best, best_score = None, -1.0
        for original, clean in normalized.items():
            if original in used:
                continue
            score = 0.0
            for alias in aliases:
                alias = _normalize_event_name(alias)
                if clean == alias:
                    score = max(score, 1.0)
                elif alias in clean or clean in alias:
                    score = max(score, 0.82)
            context = stat_lookup.get(original, {})
            share = float(context.get("user_share", 0) or 0)
            median_h = float(context.get("median_hours", 0) or 0)
            events_per_user = float(context.get("events_per_user", 0) or 0)
            explicit_name_score = score

            if role == "entry" and share >= 0.5 and median_h <= 1:
                score = max(score, 0.64)
            if role == "onboarding" and share >= 0.25 and 0 <= median_h <= 24:
                score = max(score, 0.52 if score == 0 else score)
            if role == "activation" and share >= 0.08 and 0 <= median_h <= 168:
                # Activation/core-value events are usually milestone-like, not events that
                # fire repeatedly on every visit. Behaviour-only guesses are therefore
                # allowed only when repetition is modest. Explicit activation names win.
                if explicit_name_score >= 0.80 or events_per_user <= 2.5:
                    score = max(score, 0.48 if score == 0 else score)

                generic_repeat_event = any(
                    token in clean
                    for token in ["open", "view", "session", "heartbeat", "screen", "page"]
                )
                if explicit_name_score < 0.80 and (events_per_user > 2.5 or generic_repeat_event):
                    score *= 0.40

            if score > best_score:
                best, best_score = original, score
        if best is not None and best_score >= 0.48:
            roles[role] = best
            confidence[role] = "High" if best_score >= 0.8 else "Medium"
            used.add(best)
        else:
            roles[role] = None
            confidence[role] = "Not detected"

    # If entry is still unknown, the earliest high-reach event is the safest structural guess.
    if not roles.get("entry") and not stats.empty:
        candidate = stats.loc[stats["user_share"] >= 0.35].sort_values(["median_hours", "user_share"], ascending=[True, False])
        if not candidate.empty:
            roles["entry"] = str(candidate.iloc[0][event_col])
            confidence["entry"] = "Medium"

    return roles, confidence


def get_event_role(role, df=None, mappings=None):
    """Resolve a semantic event role against the DataFrame being analysed.

    ``df`` is explicit so filtered/subset analyses do not silently validate roles
    against the full session dataset. ``mappings`` can also be injected by tests.
    """
    if df is None:
        df = st.session_state.get("dataset_df")
    if mappings is None:
        mappings = st.session_state.get("event_role_mappings", {}) or {}
    value = mappings.get(role)
    if df is None:
        return None
    if value in _event_values(df):
        return value
    return None

def _normalize_column_name(value):
    return str(value).lower().strip().replace(" ", "_").replace("-", "_")


def _raw_find_column(df, keywords, exclude=None):
    """Name-based matcher used before optional mappings are confirmed."""
    exclude = set(exclude or [])
    normalized = {
        col: _normalize_column_name(col)
        for col in df.columns
        if col not in exclude
    }

    for keyword in keywords:
        key = _normalize_column_name(keyword)
        for original, clean in normalized.items():
            if clean == key:
                return original

    for keyword in keywords:
        key = _normalize_column_name(keyword)
        for original, clean in normalized.items():
            if key in clean or clean in key:
                return original

    return None


def _optional_role_score(df, column, role):
    name = _normalize_column_name(column)
    tokens = set(name.split("_"))
    aliases = OPTIONAL_ROLE_ALIASES[role]
    name_score = 0.0
    for alias in aliases:
        key = _normalize_column_name(alias)
        if name == key:
            name_score = max(name_score, 100)
        elif key in name or name in key:
            name_score = max(name_score, 82)
        else:
            overlap = len(tokens & set(key.split("_")))
            if overlap:
                name_score = max(name_score, 55 + 10 * overlap)

    s = df[column]
    non_null = s.dropna()
    if non_null.empty:
        return 0.0
    sample = non_null.astype(str).head(500)
    unique_count = non_null.nunique(dropna=True)
    unique_ratio = unique_count / max(len(non_null), 1)
    numeric_ratio = _numeric_ratio(non_null)
    values = {str(v).strip().lower() for v in non_null.drop_duplicates().head(200)}
    value_score = 0.0

    if role == "device":
        common = {"ios", "android", "web", "mobile", "desktop", "tablet", "windows", "macos", "linux"}
        if values and len(values & common) / max(min(len(values), len(common)), 1) >= 0.25:
            value_score = 95
        elif unique_count <= 15 and pd.api.types.is_object_dtype(s):
            value_score = 35
    elif role == "geography":
        upper = [str(v).strip() for v in non_null.drop_duplicates().head(100)]
        code_rate = sum(bool(re.fullmatch(r"[A-Za-z]{2,3}", v)) for v in upper) / max(len(upper), 1)
        if code_rate >= 0.70 and 2 <= unique_count <= 300:
            value_score = 80
        elif 2 <= unique_count <= 300 and pd.api.types.is_object_dtype(s):
            value_score = 30
    elif role == "version":
        pattern_rate = sample.str.match(r"^(?:v|r|build[-_ ]?)?\d+(?:[._-]\d+)+(?:[-_a-z0-9]*)?$", case=False).mean()
        if pattern_rate >= 0.60:
            value_score = 95
        elif unique_count <= 50:
            value_score = 30
    elif role == "revenue":
        nums = pd.to_numeric(non_null, errors="coerce")
        if numeric_ratio >= 0.90 and nums.notna().any():
            non_negative = (nums.dropna() >= 0).mean()
            value_score = 80 if non_negative >= 0.95 else 60
    elif role == "channel":
        if pd.api.types.is_object_dtype(s) and 2 <= unique_count <= 50:
            value_score = 45
        channel_words = {"organic", "referral", "direct", "email", "social", "paid", "google", "meta", "tiktok", "affiliate"}
        if any(any(w in v for w in channel_words) for v in values):
            value_score = max(value_score, 85)
    elif role == "plan":
        if pd.api.types.is_object_dtype(s) and 2 <= unique_count <= 30:
            value_score = 45
        plan_words = {"free", "basic", "pro", "premium", "enterprise", "monthly", "annual", "starter", "mid_market"}
        if any(any(w in v for w in plan_words) for v in values):
            value_score = max(value_score, 80)
    elif role == "experiment":
        if pd.api.types.is_object_dtype(s) and 2 <= unique_count <= 20:
            if any(v in {"a", "b", "control", "treatment", "variant_a", "variant_b"} for v in values):
                value_score = 85
            else:
                value_score = 35
    elif role == "session":
        if pd.api.types.is_object_dtype(s) and unique_count >= 20 and unique_ratio < 0.95:
            value_score = 55
        if any(k in name for k in ["session", "journey", "visit"]):
            value_score = max(value_score, 80)

    # Names matter, but values can rescue unfamiliar schemas.
    score = 0.62 * name_score + 0.38 * value_score
    if name_score >= 100:
        score = max(score, 95)
    return round(float(score), 2)


def detect_optional_mappings(df):
    """Detect optional semantic roles using column names plus value patterns."""
    core = {
        st.session_state.get("user_col"),
        st.session_state.get("event_col"),
        st.session_state.get("timestamp_col"),
    }
    core.discard(None)
    mappings = {}
    used = set(core)
    role_order = ["session", "experiment", "version", "channel", "device", "geography", "plan", "revenue"]

    for role in role_order:
        candidates = []
        for col in df.columns:
            if col in used:
                continue
            score = _optional_role_score(df, col, role)
            candidates.append((score, col))
        candidates.sort(reverse=True)
        best_score, best_col = candidates[0] if candidates else (0, None)
        # Conservative threshold: uncertain roles stay unassigned for review.
        mappings[role] = best_col if best_score >= 55 else None
        if mappings[role] is not None:
            used.add(mappings[role])
    return mappings


def detect_custom_dimensions(df, optional_mappings=None):
    """Return useful unmapped categorical dimensions for segmentation."""
    optional_mappings = optional_mappings or {}
    excluded = {
        st.session_state.get("user_col"),
        st.session_state.get("event_col"),
        st.session_state.get("timestamp_col"),
        *[value for value in optional_mappings.values() if value],
    }
    excluded.discard(None)

    dimensions = []
    row_count = max(len(df), 1)
    for col in df.columns:
        if col in excluded:
            continue

        nunique = df[col].nunique(dropna=True)
        if nunique < 2:
            continue

        is_text_like = (
            pd.api.types.is_object_dtype(df[col])
            or pd.api.types.is_string_dtype(df[col])
            or pd.api.types.is_bool_dtype(df[col])
            or isinstance(df[col].dtype, pd.CategoricalDtype)
        )
        manageable = nunique <= min(100, max(20, int(row_count * 0.05)))

        if is_text_like and manageable:
            dimensions.append(col)

    return dimensions[:12]


def get_optional_column(role):
    """Return the user-confirmed optional column for a canonical role."""
    mappings = st.session_state.get("optional_mappings", {}) or {}
    col = mappings.get(role)
    if col in getattr(st.session_state.get("dataset_df"), "columns", []):
        return col
    return None


def find_column(df, keywords):
    """
    Backwards-compatible lookup used by the existing dashboards.
    Confirmed optional mappings always win; raw alias detection is fallback only.
    """
    keyword_set = {_normalize_column_name(k) for k in keywords}
    for role, aliases in OPTIONAL_ROLE_ALIASES.items():
        alias_set = {_normalize_column_name(a) for a in aliases}
        if keyword_set & alias_set:
            mapped = get_optional_column(role)
            if mapped is not None:
                return mapped

    return _raw_find_column(df, keywords)




# ================================================================
# SHARED COHORT / RETENTION HELPERS
# ================================================================

MISSING_DIMENSION_LABEL = "Missing / not recorded"

def _user_dimension_snapshot(df, dimension_col, include_missing=True):
    """Return one stable dimension value per user using the first non-null value."""
    if not dimension_col or dimension_col not in df.columns:
        return pd.DataFrame()
    user_col = st.session_state.get("user_col")
    time_col = st.session_state.get("timestamp_col")
    if not user_col or user_col not in df.columns:
        return pd.DataFrame()
    all_users = df[[user_col]].dropna().drop_duplicates().copy()
    cols = [user_col, dimension_col] + ([time_col] if time_col in df.columns else [])
    temp = df[cols].dropna(subset=[user_col, dimension_col]).copy()
    if time_col in temp.columns:
        temp[time_col] = _safe_datetime(temp[time_col])
        temp = temp.sort_values([user_col, time_col], na_position="last")
    first = temp.drop_duplicates(user_col)[[user_col, dimension_col]]
    out = all_users.merge(first, on=user_col, how="left")
    if include_missing:
        out[dimension_col] = out[dimension_col].fillna(MISSING_DIMENSION_LABEL)
    else:
        out = out.dropna(subset=[dimension_col])
    return out

def _user_dimension_at_event(df, dimension_col, anchor_event, include_missing=False):
    """Use the latest known dimension at/before a pre-outcome anchor event."""
    if not dimension_col or dimension_col not in df.columns:
        return pd.DataFrame()
    user_col = st.session_state.get("user_col")
    event_col = st.session_state.get("event_col")
    time_col = st.session_state.get("timestamp_col")
    if not anchor_event or not all(c and c in df.columns for c in [user_col, event_col, time_col]):
        return _user_dimension_snapshot(df, dimension_col, include_missing=include_missing)

    work = df[[user_col, event_col, time_col, dimension_col]].dropna(subset=[user_col]).copy()
    work[time_col] = _safe_datetime(work[time_col])
    anchors = (work.loc[work[event_col].astype(str).eq(str(anchor_event))]
               .dropna(subset=[time_col])
               .sort_values(time_col)
               .drop_duplicates(user_col)[[user_col, time_col]]
               .rename(columns={time_col: "_anchor_time"}))
    if anchors.empty:
        return _user_dimension_snapshot(df, dimension_col, include_missing=include_missing)

    dim_rows = work.dropna(subset=[dimension_col, time_col]).sort_values([user_col, time_col])
    joined = dim_rows.merge(anchors, on=user_col, how="inner")
    joined = joined[joined[time_col] <= joined["_anchor_time"]]
    picked = (joined.sort_values([user_col, time_col])
              .drop_duplicates(user_col, keep="last")[[user_col, dimension_col]])

    all_users = df[[user_col]].dropna().drop_duplicates()
    out = all_users.merge(picked, on=user_col, how="left")
    fallback = _user_dimension_snapshot(df, dimension_col, include_missing=False)
    if not fallback.empty:
        out = out.merge(fallback, on=user_col, how="left", suffixes=("", "_fallback"))
        fb = f"{dimension_col}_fallback"
        out[dimension_col] = out[dimension_col].where(out[dimension_col].notna(), out[fb])
        out = out.drop(columns=[fb])
    if include_missing:
        out[dimension_col] = out[dimension_col].fillna(MISSING_DIMENSION_LABEL)
    else:
        out = out.dropna(subset=[dimension_col])
    return out

def _retention_suitability(df):
    """Return whether exact-day retention is meaningful for this event dataset."""
    user_col = st.session_state.get("user_col")
    event_col = st.session_state.get("event_col")
    time_col = st.session_state.get("timestamp_col")
    if not all(c and c in df.columns for c in [user_col, event_col, time_col]):
        return {"suitable": False, "reason": "Needs user, event and timestamp fields."}

    engagement_event = get_event_role("engagement", df)
    if engagement_event:
        engaged_users = df.loc[df[event_col].astype(str).eq(str(engagement_event)), user_col].dropna().nunique()
        total_users = max(df[user_col].dropna().nunique(), 1)
        share = engaged_users / total_users
        if share >= 0.03:
            return {
                "suitable": True,
                "reason": f"Recurring engagement event detected for {share:.1%} of users.",
                "engaged_share": share,
            }

    tmp = df[[user_col, event_col, time_col]].dropna(subset=[user_col, time_col]).copy()
    tmp[time_col] = _safe_datetime(tmp[time_col])
    tmp = tmp.dropna(subset=[time_col])
    if tmp.empty:
        return {"suitable": False, "reason": "No usable timestamps for recurring-activity checks."}

    lifecycle_events = {
        get_event_role("entry", df), get_event_role("onboarding", df), get_event_role("activation", df),
        get_event_role("conversion", df), get_event_role("churn", df), get_event_role("payment_attempt", df),
        get_event_role("payment_success", df), get_event_role("payment_failure", df),
    }
    lifecycle_events = {str(x) for x in lifecycle_events if x}
    lifecycle_regex = r"(?:payment|renewal|subscription|billing|charge|trial)"
    non_lifecycle = (~tmp[event_col].astype(str).isin(lifecycle_events) &
                     ~tmp[event_col].astype(str).str.contains(lifecycle_regex, case=False, regex=True, na=False))
    activity = tmp.loc[non_lifecycle].copy()
    if activity.empty:
        return {"suitable": False, "reason": "The file mostly contains lifecycle/payment events rather than recurring product activity."}

    activity["_active_date"] = activity[time_col].dt.floor("D")
    active_days = activity.groupby(user_col)["_active_date"].nunique()
    total_users = max(tmp[user_col].nunique(), 1)
    recurring_share = float((active_days >= 2).sum() / total_users)
    if recurring_share >= 0.10:
        return {
            "suitable": True,
            "reason": f"{recurring_share:.1%} of users show recurring non-lifecycle activity across multiple days.",
            "recurring_share": recurring_share,
        }
    return {
        "suitable": False,
        "reason": "Too little recurring non-lifecycle product activity is present to interpret exact-day retention reliably.",
        "recurring_share": recurring_share,
    }

def get_analysis_capabilities():
    """
    Decide which analysis modules the current dataset can support.

    Core user/event/time fields have already been confirmed during mapping.
    Optional modules are unlocked only when the required dimensions exist.
    """

    df = st.session_state.dataset_df

    channel_col = find_column(
        df,
        [
            "acquisition_channel",
            "marketing_channel",
            "channel",
            "source",
            "traffic_source",
            "utm_source",
        ],
    )

    device_col = find_column(
        df,
        [
            "device_type",
            "device",
            "platform",
            "os",
            "operating_system",
        ],
    )

    country_col = find_column(
        df,
        [
            "country",
            "country_code",
            "region",
            "market",
            "geo",
        ],
    )

    version_col = find_column(
        df,
        [
            "app_version",
            "version",
            "release_version",
            "build_version",
        ],
    )

    revenue_col = find_column(
        df,
        [
            "amount_usd",
            "revenue",
            "amount",
            "price",
            "transaction_value",
            "order_value",
        ],
    )

    plan_col = find_column(
        df,
        [
            "plan_name",
            "plan",
            "subscription_plan",
            "billing_plan",
            "tier",
        ],
    )

    experiment_col = find_column(
        df,
        [
            "experiment",
            "experiment_id",
            "test_id",
            "ab_test",
            "variant",
            "experiment_variant",
        ],
    )

    segment_dimensions = [
        col
        for col in [
            channel_col,
            device_col,
            country_col,
            version_col,
            plan_col,
        ]
        if col is not None
    ]

    capabilities = [
        {
            "name": "Behavior over time",
            "description": "Track event volume and product activity over time.",
            "status": "ready",
            "reason": None,
        },
        {
            "name": "Funnel analysis",
            "description": "Measure how users move through key product steps.",
            "status": "ready" if len(_funnel_steps(df)) >= 2 else "limited",
            "reason": None if len(_funnel_steps(df)) >= 2 else "Confirm at least two event roles such as Entry, Activation or Conversion.",
        },
        {
            "name": "Retention",
            "description": "Measure whether users return after their first activity.",
            "status": "ready" if _retention_suitability(df).get("suitable") else "limited",
            "reason": None if _retention_suitability(df).get("suitable") else "Needs recurring product-engagement evidence; payment and lifecycle events alone are not enough for reliable exact-day retention.",
        },
        {
            "name": "Cohort analysis",
            "description": "Compare user groups based on when they entered the product.",
            "status": "ready",
            "reason": None,
        },
        {
            "name": "Segmentation",
            "description": (
                f"Break product behavior down by "
                f"{len(segment_dimensions)} detected dimensions."
                if segment_dimensions
                else "Compare product behavior across user segments."
            ),
            "status": "ready" if segment_dimensions else "limited",
            "reason": (
                None
                if segment_dimensions
                else "Add fields such as country, device, channel or plan."
            ),
        },
        {
            "name": "Acquisition analysis",
            "description": "Compare product quality and conversion by acquisition source.",
            "status": "ready" if channel_col else "missing",
            "reason": (
                None
                if channel_col
                else "Needs an acquisition, source or channel column."
            ),
        },
        {
            "name": "Release / version analysis",
            "description": "Detect behavior changes across product or app versions.",
            "status": "ready" if version_col else "missing",
            "reason": (
                None
                if version_col
                else "Needs an app or product version column."
            ),
        },
        {
            "name": "Revenue & subscription",
            "description": "Connect product behavior to payments, plans and revenue.",
            "status": (
                "ready"
                if revenue_col and (plan_col or _paid_event(df))
                else "limited"
                if revenue_col or _paid_event(df)
                else "missing"
            ),
            "reason": (
                None
                if revenue_col and (plan_col or _paid_event(df))
                else "Add a revenue field and confirm a paid/conversion event for the full module."
                if revenue_col or _paid_event(df)
                else "Needs revenue data or a confirmed paid/conversion event."
            ),
        },
        {
            "name": "Experiment analysis",
            "description": "Compare behavior and outcomes across experiment variants.",
            "status": "ready" if experiment_col else "missing",
            "reason": (
                None
                if experiment_col
                else "Needs an experiment or variant column."
            ),
        },
    ]

    return capabilities


def _get_analysis_areas_cached():
    return _session_analysis_cache_get("analysis_areas", get_analysis_areas)


def _find_churn_column(df):
    candidates = []
    for col in df.columns:
        name = _normalize_column_name(col)
        if any(k in name for k in ["churn", "cancel", "active", "status", "retained"]):
            candidates.append(col)
    return candidates[0] if candidates else None


def get_analysis_areas():
    """Return dashboards appropriate to the detected row grain."""
    df = st.session_state.dataset_df
    grain = (st.session_state.get("grain_info") or {}).get("grain", "event")

    channel_col = find_column(df, ["acquisition_channel", "marketing_channel", "channel", "source", "traffic_source", "utm_source", "entry_origin"])
    version_col = find_column(df, ["app_version", "version", "release_version", "build_version", "build_tag"])
    device_col = find_column(df, ["device_type", "device", "platform", "os", "operating_system", "client_surface"])
    revenue_col = find_column(df, ["amount_usd", "revenue", "amount", "price", "transaction_value", "order_value", "cash_value", "ltv", "gmv"])
    plan_col = find_column(df, ["plan_name", "plan", "subscription_plan", "billing_plan", "tier", "subscription_tier", "package"])

    if grain == "user":
        churn_col = _find_churn_column(df)
        return {
            "profile": {"label": "User Profile", "status": "ready", "purpose": "Understand the shape of the customer base and important dimensions.", "missing": None},
            "segments": {"label": "Segments", "status": "ready" if (channel_col or plan_col or st.session_state.get("custom_dimensions")) else "limited", "purpose": "Compare user groups across available categorical dimensions.", "missing": None},
            "acquisition": {"label": "Acquisition", "status": "ready" if channel_col else "missing", "purpose": "Understand how the user base is distributed by acquisition source.", "missing": None if channel_col else "Needs an acquisition/source field."},
            "value": {"label": "Value / LTV", "status": "ready" if revenue_col else "missing", "purpose": "Compare customer value when a monetary field is available.", "missing": None if revenue_col else "Needs a revenue, value or LTV field."},
            "churn": {"label": "Churn", "status": "ready" if churn_col else "missing", "purpose": "Analyze churn/active status when a lifecycle outcome exists.", "missing": None if churn_col else "Needs a churn, cancellation or status field."},
        }

    if grain == "transaction":
        time_col = st.session_state.get("timestamp_col")
        return {
            "revenue": {"label": "Revenue", "status": "ready" if revenue_col else "missing", "purpose": "Measure transaction value and revenue trends.", "missing": None if revenue_col else "Needs an amount/revenue field."},
            "customers": {"label": "Customers", "status": "ready" if st.session_state.get("user_col") else "limited", "purpose": "Measure customer frequency and repeat purchasing.", "missing": None},
            "orders": {"label": "Transactions", "status": "ready", "purpose": "Understand transaction volume and timing.", "missing": None},
            "segments": {"label": "Segments", "status": "ready" if (channel_col or plan_col or st.session_state.get("custom_dimensions")) else "limited", "purpose": "Break transaction performance down by available dimensions.", "missing": None},
            "trend": {"label": "Trend", "status": "ready" if time_col in df.columns else "missing", "purpose": "Track transaction activity over time.", "missing": None if time_col in df.columns else "Needs a timestamp/date field."},
        }

    if grain == "ambiguous":
        return {
            "quality": {"label": "Data Quality", "status": "ready", "purpose": "Inspect the dataset before selecting an analytical model.", "missing": None},
        }

    monetization_status = "ready" if revenue_col and (plan_col or _paid_event(df)) else "limited" if revenue_col or plan_col or _paid_event(df) else "missing"
    product_status = "ready" if version_col or device_col else "limited"
    return {
        "acquisition": {"label": "Acquisition", "status": "ready" if channel_col else "missing", "purpose": "Understand where users come from and which sources bring the strongest users.", "missing": "Needs an acquisition, source or channel column." if not channel_col else None},
        "activation": {"label": "Activation & Funnel", "status": "ready" if len(_funnel_steps(df)) >= 2 else "limited", "purpose": "See where users progress, where they drop and how quickly they reach value.", "missing": None if len(_funnel_steps(df)) >= 2 else "Confirm at least two event roles to build a reliable funnel."},
        "engagement": {
            "label": "Engagement & Retention",
            "status": "ready" if _retention_suitability(df).get("suitable") else "limited",
            "purpose": "Understand whether users return, how often they engage and which behaviors are sticky.",
            "missing": None if _retention_suitability(df).get("suitable") else "Exact-day retention needs recurring product-engagement evidence; lifecycle/payment-only activity is not enough.",
        },
        "monetization": {"label": "Monetization", "status": monetization_status, "purpose": "Connect product behavior to conversion, revenue, renewals and churn.", "missing": None if monetization_status == "ready" else "Full view needs revenue and a confirmed paid/conversion semantic."},
        "product": {"label": "Product & Release", "status": product_status, "purpose": "Spot regressions and behavior changes across releases, devices and product surfaces.", "missing": None if product_status == "ready" else "Add a version, device or platform field for a stronger release view."},
    }


def _event_values(df):
    event_col = st.session_state.get("event_col")
    if not event_col or event_col not in df.columns:
        return []
    return [str(x) for x in df[event_col].dropna().astype(str).unique()]


def _find_event(df, aliases):
    values = _event_values(df)
    normalized = {v: v.lower().strip().replace(" ", "_") for v in values}
    for alias in aliases:
        for original, clean in normalized.items():
            if clean == alias:
                return original
    for alias in aliases:
        for original, clean in normalized.items():
            if alias in clean:
                return original
    return None


def _activation_event(df):
    return get_event_role("activation", df) or get_event_role("onboarding", df)


def _paid_event(df):
    return get_event_role("conversion", df) or get_event_role("payment_success", df)


def _funnel_steps(df):
    """Build the funnel from confirmed semantic event roles, not NOVA-specific names."""
    ordered_roles = [
        ("Entry", "entry"),
        ("Onboarding", "onboarding"),
        ("Core value", "activation"),
        ("Conversion", "conversion"),
    ]
    steps, used = [], set()
    for label, role in ordered_roles:
        event_name = get_event_role(role, df)
        if event_name and event_name not in used:
            steps.append((label, event_name))
            used.add(event_name)
    return steps


def nice_label(value):
    if value is None:
        return "—"
    text = str(value).replace("_", " ").replace("-", " ").strip()
    return text[:1].upper() + text[1:]



__all__ = [name for name in globals() if not name.startswith("__")]

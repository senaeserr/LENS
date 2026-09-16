import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from rapidfuzz import fuzz


# ================================================================
# CANONICAL CORE ROLE ALIASES
# ================================================================

ALIASES = {
    "user_id": [
        "user_id", "userid", "user_key", "user_uuid", "uid", "user",
        "customer_id", "customerid", "customer_key", "customer_uuid", "customer",
        "cust_id", "cust", "client_id", "client_key", "client",
        "member_id", "member_key", "member", "account_id", "account_key", "account",
        "profile_id", "profile_key", "profile", "visitor_id", "visitor_key", "visitor",
        "player_id", "player_key", "player", "consumer_id", "person_id", "contact_id",
        "subscriber_id", "buyer_id", "shopper_id",
    ],
    "event_name": [
        "event_name", "event", "event_type", "event_code", "event_key",
        "action", "action_name", "action_type", "action_code",
        "activity", "activity_name", "activity_type", "activity_code",
        "interaction", "interaction_name", "interaction_type",
        "behavior", "behaviour", "behavior_type", "behaviour_type",
        "event_label", "event_category", "operation", "operation_type",
        "screen_event", "track_name", "tracking_event",
    ],
    "event_timestamp": [
        "event_timestamp", "timestamp", "time", "datetime", "date",
        "event_time", "event_date", "created_at", "created_on", "occurred_at",
        "occurred_on", "happened_at", "action_time", "activity_time", "logged_at",
        "recorded_at", "tracked_at", "ts", "event_ts", "time_stamp",
    ],
}


# ================================================================
# NAME NORMALIZATION
# ================================================================

def normalize_name(name) -> str:
    """Normalize a raw column name without destroying camelCase information."""
    name = str(name).strip()
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    name = name.lower()
    name = name.replace("-", "_").replace(" ", "_").replace(".", "_")
    name = re.sub(r"[^a-z0-9_]", "", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def _tokens(name: str) -> set:
    return {t for t in normalize_name(name).split("_") if t}

def make_unique_column_names(columns) -> List[str]:
    """Return deterministic unique column labels without dropping any columns.

    Duplicate headers are legal in pandas DataFrames and can appear in messy exports.
    Downstream code expects df[col] to be a Series, so duplicate names are suffixed
    as ``__2``, ``__3`` ... while preserving the first occurrence unchanged.
    """
    seen = {}
    output = []
    for raw in columns:
        base = str(raw)
        count = seen.get(base, 0) + 1
        seen[base] = count
        output.append(base if count == 1 else f"{base}__{count}")
    return output


def sanitize_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a shallow copy with unique column labels when duplicates exist."""
    if df is None or not getattr(df.columns, "duplicated", lambda: [])().any():
        return df
    out = df.copy(deep=False)
    out.columns = make_unique_column_names(out.columns)
    return out



# ================================================================
# NAME SIMILARITY
# ================================================================

def alias_score(column_name, canonical_role) -> float:
    normalized = normalize_name(column_name)
    aliases = ALIASES[canonical_role]

    best_score = 0.0
    for alias in aliases:
        alias_norm = normalize_name(alias)
        if normalized == alias_norm:
            return 100.0

        ratio = fuzz.ratio(normalized, alias_norm)
        partial = fuzz.partial_ratio(normalized, alias_norm)
        token_ratio = fuzz.token_set_ratio(
            normalized.replace("_", " "),
            alias_norm.replace("_", " "),
        )
        score = 0.55 * ratio + 0.25 * partial + 0.20 * token_ratio

        # Exact informative token overlap gives a small, bounded boost.
        overlap = _tokens(normalized) & _tokens(alias_norm)
        if overlap:
            score = min(100.0, score + 5.0)

        best_score = max(best_score, score)

    return round(best_score, 2)


# ================================================================
# SAFE SERIES HELPERS
# ================================================================

def _sample_non_null(series: pd.Series, n: int = 1000) -> pd.Series:
    values = series.dropna()
    if len(values) <= n:
        return values
    # Deterministic evenly-spaced sample; avoids only looking at the first rows.
    idx = np.linspace(0, len(values) - 1, n).astype(int)
    return values.iloc[idx]


def _string_like(series: pd.Series) -> bool:
    return (
        pd.api.types.is_string_dtype(series)
        or pd.api.types.is_object_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


def _integer_like(series: pd.Series) -> bool:
    return pd.api.types.is_integer_dtype(series)


# ================================================================
# TIMESTAMP VALUE SCORE
# ================================================================

def _parse_datetime_sample(sample: pd.Series) -> pd.Series:
    """Parse mixed strings plus unix seconds/ms/us/ns without trusting one format."""
    text = sample.astype(str).str.strip()
    try:
        parsed = pd.to_datetime(text, errors="coerce", format="mixed", utc=False)
    except (TypeError, ValueError):
        parsed = pd.to_datetime(text, errors="coerce", utc=False)

    missing = parsed.isna()
    if missing.any():
        numeric = pd.to_numeric(text[missing], errors="coerce")
        for unit, low, high in [
            ("s", 1e8, 4e9),
            ("ms", 1e11, 4e12),
            ("us", 1e14, 4e15),
            ("ns", 1e17, 4e18),
        ]:
            mask = numeric.abs().between(low, high, inclusive="both")
            if mask.any():
                converted = pd.to_datetime(numeric[mask], errors="coerce", unit=unit, utc=False)
                parsed.loc[converted.index] = converted
    return parsed


def timestamp_value_score(series: pd.Series) -> float:
    sample = _sample_non_null(series, 1000)
    if len(sample) == 0:
        return 0.0

    # Native datetime is strong evidence.
    if pd.api.types.is_datetime64_any_dtype(series):
        return 100.0

    parsed = _parse_datetime_sample(sample)
    success_rate = float(parsed.notna().mean())

    # Avoid mistaking plain IDs / years / small integers for timestamps.
    numeric_ratio = pd.to_numeric(sample, errors="coerce").notna().mean()
    score = success_rate * 100
    if numeric_ratio > 0.98 and not _string_like(series):
        # Unix timestamps can still be numeric, but arbitrary integer IDs often parse too easily.
        med = pd.to_numeric(sample, errors="coerce").dropna().median()
        looks_unix = med is not None and (
            1e8 <= abs(float(med)) <= 4e9
            or 1e11 <= abs(float(med)) <= 4e12
            or 1e14 <= abs(float(med)) <= 4e15
            or 1e17 <= abs(float(med)) <= 4e18
        )
        if not looks_unix:
            score *= 0.40

    return round(min(score, 100.0), 2)


# ================================================================
# USER-ID VALUE SCORE
# ================================================================

def user_id_value_score(series: pd.Series) -> float:
    non_null = series.dropna()
    if len(non_null) == 0:
        return 0.0

    unique_count = int(non_null.nunique(dropna=True))
    row_count = len(non_null)
    unique_ratio = unique_count / max(row_count, 1)
    null_rate = float(series.isna().mean())

    score = 0.0

    # IDs usually have many distinct values, but in event tables each user repeats.
    if unique_count >= 1000:
        score += 35
    elif unique_count >= 100:
        score += 30
    elif unique_count >= 20:
        score += 18
    elif unique_count >= 2:
        score += 5

    if 0.01 <= unique_ratio <= 0.90:
        score += 35
    elif 0.90 < unique_ratio <= 1.00:
        # User-level tables often have one row per user.
        score += 26
    elif unique_ratio < 0.01:
        score += 8

    if _string_like(series) or _integer_like(series):
        score += 20

    if null_rate < 0.01:
        score += 10
    elif null_rate < 0.05:
        score += 6

    return round(min(score, 100.0), 2)


# ================================================================
# EVENT VALUE SCORE
# ================================================================

def event_value_score(series: pd.Series) -> float:
    non_null = series.dropna()
    if len(non_null) == 0:
        return 0.0

    unique_count = int(non_null.nunique(dropna=True))
    unique_ratio = unique_count / max(len(non_null), 1)
    null_rate = float(series.isna().mean())

    score = 0.0

    # Event vocabularies are usually bounded but not necessarily tiny.
    if 2 <= unique_count <= 100:
        score += 45
    elif 100 < unique_count <= 500:
        score += 34
    elif 500 < unique_count <= 2000:
        score += 15

    if unique_ratio <= 0.05:
        score += 30
    elif unique_ratio <= 0.20:
        score += 24
    elif unique_ratio <= 0.50:
        score += 10

    if _string_like(series):
        score += 20

    if null_rate < 0.02:
        score += 5
    elif null_rate < 0.10:
        score += 2

    # Values that resemble a repeated categorical vocabulary are stronger evidence.
    sample = _sample_non_null(series.astype(str), 500)
    if len(sample):
        avg_len = float(sample.str.len().mean())
        alpha_share = float(sample.str.contains(r"[A-Za-z]", regex=True).mean())
        if 2 <= avg_len <= 80 and alpha_share >= 0.60:
            score += 5

    return round(min(score, 100.0), 2)


# ================================================================
# ROLE-SPECIFIC PENALTIES / BOOSTS
# ================================================================

def _role_adjustment(column: str, series: pd.Series, role: str) -> float:
    """Small guardrails that reduce common false positives."""
    name = normalize_name(column)
    tokens = _tokens(name)
    adjustment = 0.0

    # Obvious role words.
    if role == "user_id" and tokens & {"user", "customer", "client", "member", "account", "visitor", "player", "person"}:
        adjustment += 8
    if role == "event_name" and tokens & {"event", "action", "activity", "interaction", "behavior", "behaviour", "operation"}:
        adjustment += 8
    if role == "event_timestamp" and tokens & {"time", "timestamp", "date", "datetime", "created", "occurred", "happened", "logged", "recorded", "tracked", "ts"}:
        adjustment += 8

    # Penalize other well-known concepts when they masquerade as a core role.
    if role == "user_id" and tokens & {"session", "visit", "journey", "request", "transaction", "order", "invoice", "event"}:
        adjustment -= 22
    if role == "user_id":
        # Session/visit keys are often nearly unique per row and should not beat a repeated customer key.
        non_null = series.dropna()
        if len(non_null):
            ratio = non_null.nunique(dropna=True) / max(len(non_null), 1)
            if ratio >= 0.97 and tokens & {"session", "visit", "journey", "request"}:
                adjustment -= 18
    if role == "event_name" and tokens & {"user", "customer", "session", "transaction", "order", "amount", "price", "revenue"}:
        adjustment -= 10
    if role == "event_timestamp" and tokens & {"amount", "price", "revenue", "score", "count", "quantity"}:
        adjustment -= 12
    if role == "event_timestamp" and tokens & {"id", "key", "order", "invoice", "phone", "sku", "code", "number", "barcode"}:
        # Numeric identifiers frequently overlap Unix-epoch magnitudes. A timestamp
        # value pattern without any time-like naming evidence must not override
        # a clearly identifier-like column name.
        adjustment -= 35

    return adjustment


# ================================================================
# COMBINED ROLE SCORE
# ================================================================

def score_column(df: pd.DataFrame, column: str, canonical_role: str) -> float:
    name_score = alias_score(column, canonical_role)
    series = df[column]

    if canonical_role == "user_id":
        value_score = user_id_value_score(series)
        final_score = 0.62 * name_score + 0.38 * value_score
    elif canonical_role == "event_name":
        value_score = event_value_score(series)
        final_score = 0.62 * name_score + 0.38 * value_score
    elif canonical_role == "event_timestamp":
        value_score = timestamp_value_score(series)
        final_score = 0.50 * name_score + 0.50 * value_score

        # A purely numeric column can accidentally look like Unix epoch data.
        # Require at least weak time-like naming evidence before allowing the raw
        # value score to make it a strong timestamp candidate.
        sample = _sample_non_null(series, 500)
        numeric_ratio = pd.to_numeric(sample, errors="coerce").notna().mean() if len(sample) else 0.0
        time_tokens = {"time", "timestamp", "date", "datetime", "created", "occurred", "happened", "logged", "recorded", "tracked", "ts"}
        if numeric_ratio > 0.98 and not (_tokens(column) & time_tokens) and name_score < 45:
            final_score = min(final_score, 42.0)
    else:
        final_score = name_score

    final_score += _role_adjustment(column, series, canonical_role)
    return round(max(0.0, min(final_score, 100.0)), 2)


# ================================================================
# CONFIDENCE LABELS
# ================================================================

def confidence_label(score: float, margin: float = 0.0) -> str:
    """Human-readable review status. The app may apply stricter thresholds by grain."""
    score = float(score or 0)
    if score >= 80 and margin >= 8:
        return "high"
    if score >= 60 and margin >= 4:
        return "review"
    if score >= 50:
        return "low"
    return "not_detected"


# ================================================================
# FIND BEST MATCHES
# ================================================================

def detect_schema(df: pd.DataFrame) -> Dict[str, dict]:
    """Score every column for the three core roles.

    Important: this function always returns ranked candidates and a best_match so the
    app can reason about grain even when confidence is low. The Streamlit layer decides
    whether to trust, review, or leave a role unassigned.
    """
    roles = ["user_id", "event_name", "event_timestamp"]
    results: Dict[str, dict] = {}

    if df is not None:
        df = sanitize_duplicate_columns(df)

    if df is None or len(df.columns) == 0:
        for role in roles:
            results[role] = {
                "best_match": None,
                "confidence": 0.0,
                "status": "not_detected",
                "margin": 0.0,
                "candidates": [],
            }
        return results

    for role in roles:
        candidates: List[dict] = []
        for column in df.columns:
            score = score_column(df, column, role)
            candidates.append({"column": column, "score": score})

        candidates.sort(key=lambda x: x["score"], reverse=True)
        best = candidates[0]
        second_score = candidates[1]["score"] if len(candidates) > 1 else 0.0
        margin = round(float(best["score"] - second_score), 2)

        # A nearly unique session-like key is a dangerous user-id false positive.
        # Prefer review status unless the name itself strongly identifies a user/customer.
        if role == "user_id":
            col = best["column"]
            non_null = df[col].dropna()
            ratio = non_null.nunique(dropna=True) / max(len(non_null), 1) if len(non_null) else 1.0
            tokens = _tokens(normalize_name(col))
            if ratio >= 0.97 and tokens & {"session", "visit", "journey", "request"}:
                best = dict(best)
                best["score"] = min(float(best["score"]), 59.0)
                margin = min(margin, 3.0)

        results[role] = {
            "best_match": best["column"],
            "confidence": float(best["score"]),
            "status": confidence_label(best["score"], margin),
            "margin": margin,
            "candidates": candidates,
        }

    return results

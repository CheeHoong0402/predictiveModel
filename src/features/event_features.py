"""
Event Features Module.
Generates public holiday lead/lags, major Malaysian festival proximity windows,
school holiday duration dynamics, government salary cycles, and government cash transfers.
"""

import datetime
from typing import Dict, List, Tuple
import polars as pl
from src.features.feature_registry import FeatureDefinition


# Historical and calendar anchor dates for major Malaysian festivals
CNY_DATES = [
    datetime.date(2021, 2, 12),
    datetime.date(2022, 2, 1),
    datetime.date(2023, 1, 22),
    datetime.date(2024, 2, 10),
    datetime.date(2025, 1, 29),
]

RAYA_DATES = [
    datetime.date(2020, 5, 24),
    datetime.date(2021, 5, 13),
    datetime.date(2022, 5, 2),
    datetime.date(2023, 4, 21),
    datetime.date(2024, 4, 10),
    datetime.date(2025, 3, 31),
]

DEEPAVALI_DATES = [
    datetime.date(2020, 11, 14),
    datetime.date(2021, 11, 4),
    datetime.date(2022, 10, 24),
    datetime.date(2023, 11, 12),
    datetime.date(2024, 10, 31),
    datetime.date(2025, 10, 20),
]

CHRISTMAS_DATES = [
    datetime.date(2020, 12, 25),
    datetime.date(2021, 12, 25),
    datetime.date(2022, 12, 25),
    datetime.date(2023, 12, 25),
    datetime.date(2024, 12, 25),
    datetime.date(2025, 12, 25),
]


def _calc_festival_proximity(dt: datetime.date, anchor_dates: List[datetime.date]) -> Tuple[int, int]:
    """Calculate days until next festival and days since previous festival."""
    fut = [d for d in anchor_dates if d >= dt]
    past = [d for d in anchor_dates if d <= dt]
    days_until = (min(fut) - dt).days if fut else 999
    days_since = (dt - max(past)).days if past else 999
    return days_until, days_since


def _calc_salary_proximity(dt: datetime.date, salary_dom: int = 23) -> Tuple[int, int]:
    """Calculate days until government pay day (23rd) and days since."""
    # Current month salary date
    cur_salary = datetime.date(dt.year, dt.month, salary_dom)
    if dt < cur_salary:
        # Prior salary was last month
        if dt.month == 1:
            prev_salary = datetime.date(dt.year - 1, 12, salary_dom)
        else:
            prev_salary = datetime.date(dt.year, dt.month - 1, salary_dom)
        next_salary = cur_salary
    else:
        # Prior salary is this month
        prev_salary = cur_salary
        if dt.month == 12:
            next_salary = datetime.date(dt.year + 1, 1, salary_dom)
        else:
            next_salary = datetime.date(dt.year, dt.month + 1, salary_dom)

    days_until = (next_salary - dt).days
    days_since = (dt - prev_salary).days
    return days_until, days_since


def generate_event_features(
    df: pl.DataFrame,
) -> Tuple[pl.DataFrame, List[FeatureDefinition]]:
    """
    Generate event features including public holidays, festival proximities,
    school holiday dynamics, salary cycles, and government assistance disbursements.
    """
    definitions: List[FeatureDefinition] = []

    # 1. Unique calendar date level lookup for deterministic calendar events
    unique_dates = df.select("date").unique().sort("date")["date"].to_list()
    date_records = []

    # Check if public holiday flags are present
    pub_hol_dates = set(
        df.filter(pl.col("public_holiday_flag") == 1)
        .select("date")
        .unique()["date"]
        .to_list()
    )

    gov_assist_dates = set(
        df.filter(pl.col("government_assistance_event") == 1)
        .select("date")
        .unique()["date"]
        .to_list()
    )

    last_gov_dt = None
    sorted_dts = [datetime.datetime.strptime(d, "%Y-%m-%d").date() for d in unique_dates]
    gov_dts_set = {datetime.datetime.strptime(d, "%Y-%m-%d").date() for d in gov_assist_dates}

    for dt in sorted_dts:
        dt_str = dt.strftime("%Y-%m-%d")

        # Festival proximities
        cny_until, cny_since = _calc_festival_proximity(dt, CNY_DATES)
        raya_until, raya_since = _calc_festival_proximity(dt, RAYA_DATES)
        deep_until, deep_since = _calc_festival_proximity(dt, DEEPAVALI_DATES)
        xmas_until, xmas_since = _calc_festival_proximity(dt, CHRISTMAS_DATES)

        # Salary proximity
        sal_until, sal_since = _calc_salary_proximity(dt, salary_dom=23)

        # Public holiday leads and lags (within -3 to +7)
        hol_lead_1 = 1.0 if (dt + datetime.timedelta(days=1)).strftime("%Y-%m-%d") in pub_hol_dates else 0.0
        hol_lead_2 = 1.0 if (dt + datetime.timedelta(days=2)).strftime("%Y-%m-%d") in pub_hol_dates else 0.0
        hol_lead_3 = 1.0 if (dt + datetime.timedelta(days=3)).strftime("%Y-%m-%d") in pub_hol_dates else 0.0
        hol_lead_7 = 1.0 if (dt + datetime.timedelta(days=7)).strftime("%Y-%m-%d") in pub_hol_dates else 0.0

        hol_lag_1 = 1.0 if (dt - datetime.timedelta(days=1)).strftime("%Y-%m-%d") in pub_hol_dates else 0.0
        hol_lag_2 = 1.0 if (dt - datetime.timedelta(days=2)).strftime("%Y-%m-%d") in pub_hol_dates else 0.0
        hol_lag_3 = 1.0 if (dt - datetime.timedelta(days=3)).strftime("%Y-%m-%d") in pub_hol_dates else 0.0
        hol_lag_7 = 1.0 if (dt - datetime.timedelta(days=7)).strftime("%Y-%m-%d") in pub_hol_dates else 0.0

        # Government assistance tracking
        if dt in gov_dts_set:
            last_gov_dt = dt
        days_since_gov = (dt - last_gov_dt).days if last_gov_dt else 999

        # Major festival window active (within 14d before or 7d after any of the 4 festivals)
        is_fest_active = 1.0 if (
            (0 <= cny_until <= 14 or 0 <= cny_since <= 7) or
            (0 <= raya_until <= 14 or 0 <= raya_since <= 7) or
            (0 <= deep_until <= 14 or 0 <= deep_since <= 7) or
            (0 <= xmas_until <= 14 or 0 <= xmas_since <= 7)
        ) else 0.0

        date_records.append({
            "date": dt_str,
            # Holiday leads/lags
            "holiday_lead_1d": hol_lead_1,
            "holiday_lead_2d": hol_lead_2,
            "holiday_lead_3d": hol_lead_3,
            "holiday_lead_7d": hol_lead_7,
            "holiday_lag_1d": hol_lag_1,
            "holiday_lag_2d": hol_lag_2,
            "holiday_lag_3d": hol_lag_3,
            "holiday_lag_7d": hol_lag_7,
            # Festival CNY
            "days_until_cny": float(cny_until),
            "days_since_cny": float(cny_since),
            "is_cny_window_prior_7d": 1.0 if 0 <= cny_until <= 7 else 0.0,
            "is_cny_window_prior_14d": 1.0 if 0 <= cny_until <= 14 else 0.0,
            "is_cny_window_prior_30d": 1.0 if 0 <= cny_until <= 30 else 0.0,
            "is_cny_window_post_7d": 1.0 if 0 <= cny_since <= 7 else 0.0,
            # Festival Raya
            "days_until_raya": float(raya_until),
            "days_since_raya": float(raya_since),
            "is_raya_window_prior_7d": 1.0 if 0 <= raya_until <= 7 else 0.0,
            "is_raya_window_prior_14d": 1.0 if 0 <= raya_until <= 14 else 0.0,
            "is_raya_window_prior_30d": 1.0 if 0 <= raya_until <= 30 else 0.0,
            "is_raya_window_post_7d": 1.0 if 0 <= raya_since <= 7 else 0.0,
            # Festival Deepavali
            "days_until_deepavali": float(deep_until),
            "days_since_deepavali": float(deep_since),
            "is_deepavali_window_prior_7d": 1.0 if 0 <= deep_until <= 7 else 0.0,
            "is_deepavali_window_prior_14d": 1.0 if 0 <= deep_until <= 14 else 0.0,
            "is_deepavali_window_prior_30d": 1.0 if 0 <= deep_until <= 30 else 0.0,
            "is_deepavali_window_post_7d": 1.0 if 0 <= deep_since <= 7 else 0.0,
            # Festival Christmas
            "days_until_christmas": float(xmas_until),
            "days_since_christmas": float(xmas_since),
            "is_christmas_window_prior_7d": 1.0 if 0 <= xmas_until <= 7 else 0.0,
            "is_christmas_window_prior_14d": 1.0 if 0 <= xmas_until <= 14 else 0.0,
            "is_christmas_window_prior_30d": 1.0 if 0 <= xmas_until <= 30 else 0.0,
            "is_christmas_window_post_7d": 1.0 if 0 <= xmas_since <= 7 else 0.0,
            # Summary festival
            "major_festival_window_active": is_fest_active,
            # Salary cycle
            "is_govt_salary_day": 1.0 if dt.day == 23 else 0.0,
            "days_until_salary_day": float(sal_until),
            "days_since_salary_day": float(sal_since),
            "is_post_salary_3d": 1.0 if 0 <= sal_since <= 3 else 0.0,
            "is_post_salary_7d": 1.0 if 0 <= sal_since <= 7 else 0.0,
            "is_month_end_salary_window": 1.0 if dt.day >= 25 else 0.0,
            # Government cash transfers
            "days_since_last_gov_assist": float(days_since_gov),
            "is_post_gov_assist_3d": 1.0 if 0 <= days_since_gov <= 3 else 0.0,
            "is_post_gov_assist_7d": 1.0 if 0 <= days_since_gov <= 7 else 0.0,
        })

    df_event_lookup = pl.DataFrame(date_records)
    df = df.join(df_event_lookup, on="date", how="left")

    # Document event features
    for col in df_event_lookup.columns:
        if col == "date":
            continue
        family = "festival_proximity" if "cny" in col or "raya" in col or "deepavali" in col or "christmas" in col or "festival" in col \
            else "holiday_proximity" if "holiday" in col \
            else "salary_cycle" if "salary" in col \
            else "government_assistance"
        definitions.append(
            FeatureDefinition(
                name=col,
                description=f"Deterministic event feature: {col}",
                source_columns="date, public_holidays, government_assistance",
                calculation="Calendar astronomical/gazetted date distance lookup",
                data_type="float64",
                feature_family=family,
                known_at_prediction_time=True,
                expected_missingness="none",
                target_model_applicability="both" if "salary" in col or "fest" in col else "demand",
                leakage_risk="none",
            )
        )

    # 2. School Holiday Durations (within state/store series)
    df = df.sort(["store_id", "product_id", "date"])
    df = df.with_columns(
        (pl.col("school_holiday_flag") != pl.col("school_holiday_flag").shift(1).over(["store_id", "product_id"]))
        .fill_null(True)
        .cum_sum()
        .over(["store_id", "product_id"])
        .alias("__sh_block_id"),
        pl.int_range(0, pl.len()).over(["store_id", "product_id"]).alias("__t_idx"),
    )

    df = df.with_columns([
        pl.when(pl.col("school_holiday_flag") == 1)
        .then(pl.col("__t_idx") - pl.col("__t_idx").min().over(["store_id", "product_id", "__sh_block_id"]) + 1)
        .otherwise(0)
        .cast(pl.Float64)
        .alias("school_holiday_days_since_start"),
        pl.when(pl.col("school_holiday_flag") == 1)
        .then(pl.col("__t_idx").max().over(["store_id", "product_id", "__sh_block_id"]) - pl.col("__t_idx"))
        .otherwise(0)
        .cast(pl.Float64)
        .alias("school_holiday_days_until_end"),
        pl.when(pl.col("school_holiday_flag") == 1)
        .then(pl.col("__t_idx").max().over(["store_id", "product_id", "__sh_block_id"]) - pl.col("__t_idx").min().over(["store_id", "product_id", "__sh_block_id"]) + 1)
        .otherwise(0)
        .cast(pl.Float64)
        .alias("school_holiday_duration"),
    ])
    df = df.with_columns([
        (pl.col("school_holiday_days_since_start") <= 7.0).cast(pl.Float64).alias("is_school_holiday_first_week"),
        (pl.col("school_holiday_days_until_end") <= 7.0).cast(pl.Float64).alias("is_school_holiday_final_week"),
    ])

    sh_cols = [
        "school_holiday_days_since_start",
        "school_holiday_days_until_end",
        "school_holiday_duration",
        "is_school_holiday_first_week",
        "is_school_holiday_final_week",
    ]
    for c in sh_cols:
        definitions.append(
            FeatureDefinition(
                name=c,
                description=f"School holiday dynamics metric: {c}",
                source_columns="school_holiday_flag",
                calculation="Contiguous holiday break length and relative position",
                data_type="float64",
                feature_family="school_holiday",
                known_at_prediction_time=True,
                expected_missingness="low",
                target_model_applicability="demand",
                leakage_risk="none",
            )
        )

    df = df.drop(["__sh_block_id", "__t_idx"])

    return df, definitions

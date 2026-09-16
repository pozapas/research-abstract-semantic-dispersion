from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

RNG_SEED = 20260424
COHORT_START_YEAR = 2018
COHORT_END_YEAR = 2025
CAP_PER_CATEGORY_MONTH = 50
MIN_CELL_N_FOR_MONTHLY_DID = 15
MIN_PRE_MONTHS = 24
MIN_POST_MONTHS = 18
FIRST_POST_MONTH = "2022-12"

RAW_COLUMNS = ["id", "categories", "title", "abstract", "update_date"]


def normalize_arxiv_id(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def parse_submission_month(ids: pd.Series) -> pd.DataFrame:
    extracted = ids.astype(str).str.extract(r"^(?P<yy>\d{2})(?P<mm>\d{2})\.")
    yy = pd.to_numeric(extracted["yy"], errors="coerce")
    mm = pd.to_numeric(extracted["mm"], errors="coerce")
    year = 2000 + yy
    valid = year.between(COHORT_START_YEAR, COHORT_END_YEAR) & mm.between(1, 12)
    month = pd.Series(pd.NA, index=ids.index, dtype="object")
    month.loc[valid] = (
        year.loc[valid].astype(int).astype(str)
        + "-"
        + mm.loc[valid].astype(int).astype(str).str.zfill(2)
    )
    return pd.DataFrame({"submission_year": year, "submission_month": month, "valid_month": valid})


def deterministic_key(arxiv_id: str) -> float:
    payload = f"{RNG_SEED}:{arxiv_id}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    integer = int.from_bytes(digest, byteorder="big", signed=False)
    return integer / 2**64


def current_scope_macros() -> list[str]:
    current = pd.read_csv(DATA / "q1_arxiv_analysis_cohort.csv", usecols=["categories"])
    primary = current["categories"].astype(str).str.split().str[0]
    macros = sorted(primary.str.split(".").str[0].dropna().unique().tolist())
    return macros


def update_reservoir(reservoir: dict[tuple[str, str], pd.DataFrame], frame: pd.DataFrame) -> None:
    for key, group in frame.groupby(["primary_category", "submission_month"], observed=True):
        old = reservoir.get(key)
        merged = group if old is None else pd.concat([old, group], ignore_index=True)
        reservoir[key] = merged.nsmallest(CAP_PER_CATEGORY_MONTH, "sample_key").reset_index(drop=True)


def eligible_category_summary(cell_counts: pd.DataFrame, count_column: str) -> pd.DataFrame:
    counts = cell_counts.loc[cell_counts[count_column] >= MIN_CELL_N_FOR_MONTHLY_DID].copy()
    counts["post"] = counts["submission_month"] >= FIRST_POST_MONTH
    summary = counts.groupby("primary_category")["post"].agg(
        pre_months=lambda s: int((s == 0).sum()),
        post_months=lambda s: int((s == 1).sum()),
        usable_months="size",
    )
    summary["eligible_monthly_did"] = (
        (summary["pre_months"] >= MIN_PRE_MONTHS) & (summary["post_months"] >= MIN_POST_MONTHS)
    )
    return summary.reset_index()


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    allowed_macros = current_scope_macros()
    reservoir: dict[tuple[str, str], pd.DataFrame] = {}
    raw_total = 0
    eligible_total = 0
    month_pattern_rows = 0

    for chunk in pd.read_csv(DATA / "arxiv_meta.csv", usecols=RAW_COLUMNS, dtype={"id": str}, chunksize=100_000):
        raw_total += len(chunk)
        chunk["id"] = chunk["id"].map(normalize_arxiv_id)
        parsed = parse_submission_month(chunk["id"])
        month_pattern_rows += int(parsed["valid_month"].sum())
        chunk = pd.concat([chunk, parsed], axis=1)
        chunk = chunk.loc[chunk["valid_month"]].copy()
        if chunk.empty:
            continue

        chunk["primary_category"] = chunk["categories"].astype(str).str.split().str[0]
        chunk["macro_field"] = chunk["primary_category"].str.split(".").str[0]
        chunk = chunk.loc[chunk["macro_field"].isin(allowed_macros)].copy()
        if chunk.empty:
            continue

        eligible_total += len(chunk)
        chunk["quarter"] = pd.PeriodIndex(chunk["submission_month"], freq="M").to_timestamp().to_period("Q").astype(str)
        chunk["submitted_date_proxy"] = chunk["submission_month"] + "-01"
        chunk["source_date_policy"] = "arxiv_id_month_proxy_for_2018_2025_monthly_panel"
        chunk["sample_cap_per_category_month"] = CAP_PER_CATEGORY_MONTH
        chunk["sample_key"] = chunk["id"].map(deterministic_key)
        chunk["row_id_expanded"] = np.arange(len(chunk))
        keep_columns = [
            "id",
            "categories",
            "primary_category",
            "macro_field",
            "title",
            "abstract",
            "update_date",
            "submission_month",
            "quarter",
            "submitted_date_proxy",
            "source_date_policy",
            "sample_cap_per_category_month",
            "sample_key",
        ]
        update_reservoir(reservoir, chunk[keep_columns])

    expanded = pd.concat(reservoir.values(), ignore_index=True)
    expanded = expanded.sort_values(["primary_category", "submission_month", "sample_key"]).reset_index(drop=True)
    expanded["expanded_row_id"] = np.arange(len(expanded))

    out_path = DATA / "q1_rq2_expanded_stem_sample.csv"
    expanded.to_csv(out_path, index=False)

    selected_counts = (
        expanded.groupby(["primary_category", "macro_field", "submission_month"])
        .size()
        .rename("selected_n")
        .reset_index()
    )
    raw_counts = []
    for chunk in pd.read_csv(DATA / "arxiv_meta.csv", usecols=["id", "categories"], dtype={"id": str}, chunksize=200_000):
        chunk["id"] = chunk["id"].map(normalize_arxiv_id)
        parsed = parse_submission_month(chunk["id"])
        chunk = pd.concat([chunk, parsed], axis=1)
        chunk = chunk.loc[chunk["valid_month"]].copy()
        if chunk.empty:
            continue
        chunk["primary_category"] = chunk["categories"].astype(str).str.split().str[0]
        chunk["macro_field"] = chunk["primary_category"].str.split(".").str[0]
        chunk = chunk.loc[chunk["macro_field"].isin(allowed_macros)]
        if not chunk.empty:
            raw_counts.append(
                chunk.groupby(["primary_category", "macro_field", "submission_month"])
                .size()
                .rename("raw_n")
                .reset_index()
            )
    raw_counts_df = pd.concat(raw_counts, ignore_index=True)
    raw_counts_df = raw_counts_df.groupby(["primary_category", "macro_field", "submission_month"], as_index=False)[
        "raw_n"
    ].sum()
    cell_counts = raw_counts_df.merge(
        selected_counts,
        on=["primary_category", "macro_field", "submission_month"],
        how="left",
    )
    cell_counts["selected_n"] = cell_counts["selected_n"].fillna(0).astype(int)
    cell_counts.to_csv(RESULTS / "q1_rq2_expanded_cell_counts.csv", index=False)

    eligible_raw = eligible_category_summary(cell_counts, "raw_n")
    eligible_selected = eligible_category_summary(cell_counts, "selected_n")
    eligible = eligible_raw.merge(
        eligible_selected,
        on="primary_category",
        how="outer",
        suffixes=("_raw", "_selected"),
    )
    eligible.to_csv(RESULTS / "q1_rq2_expanded_eligibility_by_category.csv", index=False)

    manifest = {
        "script": "q1_prepare_expanded_rq2_cohort.py",
        "date": "2026-04-24",
        "paper_scope_policy": "preserve current Q1 arXiv STEM macro-field scope",
        "allowed_macro_fields": allowed_macros,
        "date_policy": "parse submission month from 2018-2025 arXiv IDs; exact day not used for monthly/quarterly panel",
        "raw_rows_scanned": int(raw_total),
        "rows_with_2018_2025_month_id": int(month_pattern_rows),
        "eligible_scope_rows_2018_2025": int(eligible_total),
        "selected_rows": int(len(expanded)),
        "selected_primary_categories": int(expanded["primary_category"].nunique()),
        "selected_macro_fields": int(expanded["macro_field"].nunique()),
        "category_month_cells_raw": int(len(cell_counts)),
        "category_month_cells_selected": int(len(selected_counts)),
        "cap_per_category_month": CAP_PER_CATEGORY_MONTH,
        "min_cell_n_for_monthly_did": MIN_CELL_N_FOR_MONTHLY_DID,
        "eligible_monthly_did_categories_raw": int(eligible_raw["eligible_monthly_did"].sum()),
        "eligible_monthly_did_categories_selected": int(eligible_selected["eligible_monthly_did"].sum()),
        "output_sample": str(out_path.relative_to(ROOT)),
        "output_cell_counts": "results/q1_rq2_expanded_cell_counts.csv",
        "output_eligibility": "results/q1_rq2_expanded_eligibility_by_category.csv",
        "rq1_guard": "expanded cohort must reproduce RQ1 descriptive direction before replacing current RQ1 outputs",
        "rq2_use": "intended for stronger monthly exposure-gradient event-study after embeddings/adoption measures are built",
        "rq3_guard": "does not alter DI-CE theory; only empirical calibration/examples can change",
    }
    (RESULTS / "q1_rq2_expanded_sample_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

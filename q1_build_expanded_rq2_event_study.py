"""Build the expanded monthly RQ2 event-study package.

The expanded RQ2 run uses the guarded STEM sample and Colab embeddings. Its
primary exposure is an external broad-field GenAI adoption measure from the
Nature survey mapped onto arXiv macro fields. Pre-cutoff text task-fit and
observed GenAI markers are retained as diagnostics, not as the main treatment.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"

RNG_SEED = 20260424
FIRST_POST_MONTH = pd.Period("2022-12", freq="M")
REFERENCE_EVENT_TIME = -1
EXPECTED_ROWS = 425_728
EXPECTED_DIM = 384
AI_HEAVY_CATEGORIES = {"cs.AI", "cs.CL", "cs.CV", "cs.LG", "stat.ML"}

TERM_DICTIONARIES = {
    "text_writing": [
        "language",
        "text",
        "natural language",
        "document",
        "summarization",
        "translation",
        "writing",
        "prompt",
        "abstract",
        "question answering",
        "dialogue",
    ],
    "code_computation": [
        "code",
        "program",
        "software",
        "algorithm",
        "simulation",
        "computation",
        "python",
        "repository",
        "package",
        "optimization",
        "data",
    ],
    "ai_ml_method": [
        "ai",
        "artificial intelligence",
        "machine learning",
        "deep learning",
        "neural",
        "transformer",
        "foundation model",
        "large language model",
        "language model",
        "natural language processing",
        "computer vision",
        "reinforcement learning",
        "generative model",
        "diffusion model",
    ],
    "literature_synthesis": [
        "survey",
        "review",
        "literature",
        "bibliometric",
        "meta-analysis",
        "systematic review",
        "scientometric",
    ],
}

GENAI_MARKERS = [
    "chatgpt",
    "gpt-3",
    "gpt-4",
    "gpt4",
    "openai",
    "large language model",
    "large language models",
    "llm",
    "llms",
    "generative ai",
    "generative artificial intelligence",
    "foundation model",
    "foundation models",
    "prompt engineering",
    "prompting",
]


@dataclass(frozen=True)
class ModelResult:
    fit: object
    design: pd.DataFrame
    frame: pd.DataFrame
    outcome: str
    weights: str


def ensure_dirs() -> None:
    for path in (RESULTS, TABLES, FIGURES):
        path.mkdir(exist_ok=True)


def compile_terms(terms: list[str]) -> re.Pattern[str]:
    parts = []
    for term in terms:
        escaped = re.escape(term.lower()).replace(r"\ ", r"\s+")
        if term in {"ai", "llm", "llms"}:
            parts.append(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])")
        else:
            parts.append(rf"\b{escaped}\b")
    return re.compile("|".join(parts), flags=re.IGNORECASE)


TERM_PATTERNS = {name: compile_terms(terms) for name, terms in TERM_DICTIONARIES.items()}
GENAI_PATTERN = compile_terms(GENAI_MARKERS)


def zscore(series: pd.Series, reference_mask: pd.Series | None = None) -> pd.Series:
    values = series.astype(float)
    reference = values.loc[reference_mask] if reference_mask is not None else values
    mean = reference.mean()
    sd = reference.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=series.index)
    return (values - mean) / sd


def load_sample_and_embeddings() -> tuple[pd.DataFrame, np.ndarray]:
    sample_path = DATA / "q1_rq2_expanded_stem_sample.csv"
    embedding_path = DATA / "q1_rq2_expanded_embeddings.npy"
    usecols = [
        "id",
        "primary_category",
        "macro_field",
        "title",
        "abstract",
        "submission_month",
        "expanded_row_id",
    ]
    sample = pd.read_csv(sample_path, usecols=usecols, dtype={"id": str})
    sample = sample.sort_values("expanded_row_id").reset_index(drop=True)
    sample["month_period"] = pd.PeriodIndex(sample["submission_month"], freq="M")
    sample["month_start"] = sample["month_period"].dt.to_timestamp()
    sample["event_time"] = sample["month_period"].map(lambda p: p.ordinal - FIRST_POST_MONTH.ordinal).astype(int)
    sample["post_gpt"] = (sample["month_period"] >= FIRST_POST_MONTH).astype(int)
    sample["log_text"] = (
        sample["title"].fillna("").astype(str) + " " + sample["abstract"].fillna("").astype(str)
    ).str.lower()
    sample["genai_marker"] = sample["log_text"].map(lambda text: bool(GENAI_PATTERN.search(text)))

    row_ids = sample["expanded_row_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(row_ids, np.arange(len(sample), dtype=np.int64)):
        raise ValueError("expanded_row_id is not contiguous after sorting.")

    embeddings = np.load(embedding_path, mmap_mode="r")
    if embeddings.shape != (EXPECTED_ROWS, EXPECTED_DIM):
        raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")
    if len(sample) != embeddings.shape[0]:
        raise ValueError("Expanded sample and embeddings are not row-aligned.")
    return sample, embeddings


def build_external_nature_exposure() -> pd.DataFrame:
    path = RESULTS / "q1_nature_arxiv_macro_exposure_validation.csv"
    if not path.exists():
        raise FileNotFoundError("Run q1_build_genai_adoption_validation.py before expanded RQ2.")
    raw = pd.read_csv(path)
    numeric = raw.select_dtypes(include=[np.number]).columns.tolist()
    nature = raw.groupby("arxiv_macro_field", as_index=False)[numeric].mean()
    nature["nature_frequency_z"] = zscore(nature["mean_frequency_score"])
    nature["nature_any_use_z"] = zscore(nature["share_any_use"])
    nature["nature_weekly_z"] = zscore(nature["share_weekly_or_daily"])
    nature["nature_research_writing_raw"] = nature[
        ["share_use_research", "share_use_manuscripts", "share_use_literature", "share_use_brainstorm"]
    ].mean(axis=1)
    nature["nature_research_writing_z"] = zscore(nature["nature_research_writing_raw"])
    return nature


def build_pre_task_fit(sample: pd.DataFrame) -> pd.DataFrame:
    pre = sample.loc[sample["month_period"] < FIRST_POST_MONTH].copy()
    for name, pattern in TERM_PATTERNS.items():
        pre[f"hit_{name}"] = pre["log_text"].map(lambda text: bool(pattern.search(text)))
    grouped = pre.groupby("primary_category")
    out = grouped.size().rename("n_pre_papers").to_frame()
    for name in TERM_DICTIONARIES:
        out[f"{name}_share"] = grouped[f"hit_{name}"].mean()
    out["pre_task_supported"] = out["n_pre_papers"] >= 20
    for name in TERM_DICTIONARIES:
        out[f"{name}_z"] = zscore(out[f"{name}_share"], out["pre_task_supported"])
    out["pre_task_fit_raw"] = (
        out["text_writing_z"] + out["code_computation_z"] + out["literature_synthesis_z"]
    )
    out["pre_task_fit_z"] = zscore(out["pre_task_fit_raw"], out["pre_task_supported"])
    out["pre_ai_ml_z"] = zscore(out["ai_ml_method_share"], out["pre_task_supported"])
    return out.reset_index()


def build_exposure_table(sample: pd.DataFrame) -> pd.DataFrame:
    category = sample.groupby("primary_category", as_index=False).agg(
        macro_field=("macro_field", "first"),
        rows=("id", "size"),
        pre_rows=("post_gpt", lambda s: int((s == 0).sum())),
        post_rows=("post_gpt", lambda s: int((s == 1).sum())),
        observed_genai_marker_share=("genai_marker", "mean"),
    )
    nature = build_external_nature_exposure()
    pre_task = build_pre_task_fit(sample)
    exposure = category.merge(nature, left_on="macro_field", right_on="arxiv_macro_field", how="left")
    exposure = exposure.merge(pre_task, on="primary_category", how="left")
    exposure["ai_heavy_category"] = exposure["primary_category"].isin(AI_HEAVY_CATEGORIES)
    exposure["exposure_claim_role"] = np.where(
        exposure["nature_frequency_z"].notna(),
        "primary_external_adoption",
        "missing_external_mapping",
    )
    exposure.to_csv(RESULTS / "q1_rq2_expanded_exposure_table.csv", index=False)
    return exposure


def _cell_metrics(indices: np.ndarray, embeddings: np.ndarray) -> tuple[float, float]:
    if len(indices) < 2:
        return np.nan, np.nan
    values = np.asarray(embeddings[indices], dtype=np.float32)
    semantic_distance = float(np.mean(pdist(values, metric="euclidean")))
    centroid = values.mean(axis=0)
    centroid_dispersion = float(np.mean(np.linalg.norm(values - centroid, axis=1)))
    return semantic_distance, centroid_dispersion


def build_month_panel(sample: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    rows = []
    for (category, month), group in sample.groupby(["primary_category", "month_period"], observed=True):
        idx = group.index.to_numpy()
        semantic_distance, centroid_dispersion = _cell_metrics(idx, embeddings)
        event_time = int(month.ordinal - FIRST_POST_MONTH.ordinal)
        rows.append(
            {
                "primary_category": category,
                "macro_field": group["macro_field"].iloc[0],
                "month": str(month),
                "month_start": month.to_timestamp(),
                "event_time": event_time,
                "event_bin": event_bin(event_time),
                "post_gpt": int(month >= FIRST_POST_MONTH),
                "n_papers": int(len(group)),
                "log_n_papers": float(np.log1p(len(group))),
                "semantic_distance": semantic_distance,
                "centroid_dispersion": centroid_dispersion,
                "genai_marker_share": float(group["genai_marker"].mean()),
            }
        )
    panel = pd.DataFrame(rows).sort_values(["primary_category", "month_start"])
    panel.to_csv(RESULTS / "q1_rq2_expanded_month_panel.csv", index=False)
    return panel


def event_bin(event_time: int) -> str:
    if event_time <= -49:
        return "lead_49_60"
    if event_time <= -37:
        return "lead_37_48"
    if event_time <= -25:
        return "lead_25_36"
    if event_time <= -13:
        return "lead_13_24"
    if event_time <= -7:
        return "lead_07_12"
    if event_time <= -4:
        return "lead_04_06"
    if event_time <= -2:
        return "lead_02_03"
    if event_time == -1:
        return "reference_m01"
    if event_time == 0:
        return "lag_00"
    if event_time <= 2:
        return "lag_01_02"
    if event_time <= 6:
        return "lag_03_06"
    if event_time <= 12:
        return "lag_07_12"
    if event_time <= 24:
        return "lag_13_24"
    return "lag_25_36"


EVENT_BIN_ORDER = [
    "lead_49_60",
    "lead_37_48",
    "lead_25_36",
    "lead_13_24",
    "lead_07_12",
    "lead_04_06",
    "lead_02_03",
    "reference_m01",
    "lag_00",
    "lag_01_02",
    "lag_03_06",
    "lag_07_12",
    "lag_13_24",
    "lag_25_36",
]

EVENT_BIN_MIDPOINT = {
    "lead_49_60": -54.5,
    "lead_37_48": -42.5,
    "lead_25_36": -30.5,
    "lead_13_24": -18.5,
    "lead_07_12": -9.5,
    "lead_04_06": -5.0,
    "lead_02_03": -2.5,
    "reference_m01": -1.0,
    "lag_00": 0.0,
    "lag_01_02": 1.5,
    "lag_03_06": 4.5,
    "lag_07_12": 9.5,
    "lag_13_24": 18.5,
    "lag_25_36": 30.5,
}


def eligible_panel(panel: pd.DataFrame, exposure: pd.DataFrame, min_n: int = 15) -> pd.DataFrame:
    frame = panel.merge(exposure, on=["primary_category", "macro_field"], how="left")
    frame = frame.loc[
        (frame["n_papers"] >= min_n)
        & frame["semantic_distance"].notna()
        & frame["nature_frequency_z"].notna()
    ].copy()
    counts = frame.groupby("primary_category")["post_gpt"].agg(
        pre_periods=lambda s: int((s == 0).sum()),
        post_periods=lambda s: int((s == 1).sum()),
    )
    keep = counts.loc[(counts["pre_periods"] >= 24) & (counts["post_periods"] >= 18)].index
    frame = frame.loc[frame["primary_category"].isin(keep)].copy()
    return frame


def safe_col(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(value))


def make_design(
    frame: pd.DataFrame,
    target_columns: list[str],
    period_column: str = "month",
    controls: list[str] | None = None,
) -> pd.DataFrame:
    controls = controls or ["log_n_papers"]
    pieces = [frame[target_columns + controls].astype(float).reset_index(drop=True)]
    field_fe = pd.get_dummies(frame["primary_category"], prefix="field", drop_first=True, dtype=float)
    time_fe = pd.get_dummies(frame[period_column].astype(str), prefix="time", drop_first=True, dtype=float)
    pieces.extend([field_fe.reset_index(drop=True), time_fe.reset_index(drop=True)])
    design = pd.concat(pieces, axis=1)
    design = design.loc[:, design.nunique(dropna=False) > 1]
    design.insert(0, "const", 1.0)
    return design.astype(float)


def fit_model(
    frame: pd.DataFrame,
    outcome: str,
    target_columns: list[str],
    weights: str = "n_papers",
) -> ModelResult:
    import statsmodels.api as sm

    working = frame.loc[frame[outcome].notna()].copy()
    design = make_design(working, target_columns=target_columns)
    y = working[outcome].astype(float).reset_index(drop=True)
    w = None if weights == "unweighted" else working[weights].astype(float).clip(lower=1.0).reset_index(drop=True)
    model = sm.OLS(y, design) if w is None else sm.WLS(y, design, weights=w)
    fit = model.fit(cov_type="cluster", cov_kwds={"groups": working["primary_category"]})
    return ModelResult(fit=fit, design=design, frame=working, outcome=outcome, weights=weights)


def coefficient_row(result: ModelResult, term: str, model_name: str, exposure_name: str) -> dict[str, object]:
    fit = result.fit
    if term not in fit.params.index:
        return {}
    ci = fit.conf_int().loc[term]
    return {
        "outcome": result.outcome,
        "exposure_name": exposure_name,
        "model_name": model_name,
        "term": term,
        "coef": float(fit.params.loc[term]),
        "se_cluster": float(fit.bse.loc[term]),
        "p_cluster": float(fit.pvalues.loc[term]),
        "ci_low": float(ci.iloc[0]),
        "ci_high": float(ci.iloc[1]),
        "n_fields": int(result.frame["primary_category"].nunique()),
        "n_field_months": int(len(result.frame)),
        "weighting": result.weights,
        "controls": "primary-category FE; month FE; log_n_papers",
    }


def run_did(
    frame: pd.DataFrame,
    outcome: str = "semantic_distance",
    exposure_column: str = "nature_frequency_z",
    exposure_name: str = "nature_frequency",
    weights: str = "n_papers",
    model_name: str = "expanded_month_did",
) -> tuple[ModelResult, dict[str, object]]:
    working = frame.copy()
    working["did_target"] = working[exposure_column].astype(float) * working["post_gpt"].astype(float)
    result = fit_model(working, outcome, ["did_target"], weights=weights)
    return result, coefficient_row(result, "did_target", model_name, exposure_name)


def run_event_study(
    frame: pd.DataFrame,
    outcome: str = "semantic_distance",
    exposure_column: str = "nature_frequency_z",
    exposure_name: str = "nature_frequency",
) -> tuple[ModelResult, pd.DataFrame, dict[str, object]]:
    working = frame.copy()
    event_cols = []
    for label in EVENT_BIN_ORDER:
        if label == "reference_m01":
            continue
        col = f"event_{safe_col(label)}"
        working[col] = working[exposure_column].astype(float) * (working["event_bin"] == label).astype(float)
        event_cols.append(col)

    result = fit_model(working, outcome, event_cols)
    rows = []
    for label in EVENT_BIN_ORDER:
        if label == "reference_m01":
            rows.append(
                {
                    "outcome": outcome,
                    "exposure_name": exposure_name,
                    "event_bin": label,
                    "event_midpoint_month": EVENT_BIN_MIDPOINT[label],
                    "coef": 0.0,
                    "se_cluster": np.nan,
                    "p_cluster": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "n_fields": int(result.frame["primary_category"].nunique()),
                    "n_field_months": int(len(result.frame)),
                    "weighting": result.weights,
                    "controls": "primary-category FE; month FE; log_n_papers",
                    "reference": True,
                }
            )
            continue
        col = f"event_{safe_col(label)}"
        row = coefficient_row(result, col, "expanded_month_event_study", exposure_name)
        if row:
            row["event_bin"] = label
            row["event_midpoint_month"] = EVENT_BIN_MIDPOINT[label]
            row["reference"] = False
            rows.append(row)
    event = pd.DataFrame(rows).sort_values("event_midpoint_month")

    lead_cols = [f"event_{safe_col(label)}" for label in EVENT_BIN_ORDER if label.startswith("lead")]
    lead_cols = [col for col in lead_cols if col in result.design.columns]
    if lead_cols:
        constraints = np.zeros((len(lead_cols), result.design.shape[1]))
        for i, col in enumerate(lead_cols):
            constraints[i, result.design.columns.get_loc(col)] = 1.0
        try:
            joint_p = float(result.fit.f_test(constraints).pvalue)
        except Exception:
            joint_p = np.nan
    else:
        joint_p = np.nan
    lead_effect = event.loc[event["event_bin"].str.startswith("lead"), "coef"].abs().mean()
    post_effect = event.loc[event["event_bin"].str.startswith("lag"), "coef"].abs().mean()
    diagnostics = {
        "outcome": outcome,
        "exposure_name": exposure_name,
        "joint_lead_p": joint_p,
        "mean_abs_lead": float(lead_effect) if pd.notna(lead_effect) else np.nan,
        "mean_abs_post": float(post_effect) if pd.notna(post_effect) else np.nan,
        "lead_to_post_ratio": float(lead_effect / post_effect) if post_effect and np.isfinite(post_effect) else np.nan,
        "status": "pass"
        if (pd.notna(joint_p) and joint_p >= 0.05 and (not post_effect or lead_effect / post_effect < 0.5))
        else "review",
        "reference_event_bin": "reference_m01",
    }
    return result, event, diagnostics


def run_placebos(frame: pd.DataFrame, main_coef: float) -> pd.DataFrame:
    rows = []
    pre_only = frame.loc[pd.PeriodIndex(frame["month"], freq="M") < FIRST_POST_MONTH].copy()
    for cutoff_text in ["2020-01", "2021-01", "2022-01"]:
        cutoff = pd.Period(cutoff_text, freq="M")
        working = pre_only.copy()
        working["post_gpt"] = (pd.PeriodIndex(working["month"], freq="M") >= cutoff).astype(int)
        try:
            _, row = run_did(
                working,
                outcome="semantic_distance",
                exposure_column="nature_frequency_z",
                exposure_name="nature_frequency",
                model_name=f"pre_cutoff_placebo_{cutoff_text}",
            )
            row["placebo_cutoff"] = cutoff_text
            row["same_sign_as_main"] = bool(np.sign(row["coef"]) == np.sign(main_coef))
            rows.append(row)
        except Exception as exc:
            rows.append(
                {
                    "model_name": f"pre_cutoff_placebo_{cutoff_text}",
                    "placebo_cutoff": cutoff_text,
                    "coef": np.nan,
                    "se_cluster": np.nan,
                    "p_cluster": np.nan,
                    "same_sign_as_main": False,
                    "error": str(exc),
                }
            )
    out = pd.DataFrame(rows)
    if "p_cluster" in out:
        n_tests = int(out["p_cluster"].notna().sum())
        out["holm_like_p"] = np.minimum(pd.to_numeric(out["p_cluster"], errors="coerce") * max(n_tests, 1), 1.0)
        out["status"] = np.where(
            out["same_sign_as_main"] & (out["holm_like_p"] < 0.05),
            "fail_same_sign_pre_cutoff",
            "pass_or_diagnostic",
        )
    out.to_csv(RESULTS / "q1_rq2_expanded_placebo_cutoffs.csv", index=False)
    return out


def run_robustness(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add_variant(
        variant: str,
        data: pd.DataFrame,
        outcome: str,
        exposure_col: str,
        exposure_name: str,
        weights: str = "n_papers",
    ) -> None:
        if data["primary_category"].nunique() < 10 or len(data) < 100:
            rows.append({"variant": variant, "status": "too_sparse", "outcome": outcome, "exposure_name": exposure_name})
            return
        try:
            _, row = run_did(
                data,
                outcome=outcome,
                exposure_column=exposure_col,
                exposure_name=exposure_name,
                weights=weights,
                model_name=variant,
            )
            row["variant"] = variant
            row["status"] = "ok"
            rows.append(row)
        except Exception as exc:
            rows.append(
                {
                    "variant": variant,
                    "status": "error",
                    "outcome": outcome,
                    "exposure_name": exposure_name,
                    "error": str(exc),
                }
            )

    add_variant("primary_nature_frequency_weighted", frame, "semantic_distance", "nature_frequency_z", "nature_frequency")
    add_variant("nature_any_use_weighted", frame, "semantic_distance", "nature_any_use_z", "nature_any_use")
    add_variant("nature_research_writing_weighted", frame, "semantic_distance", "nature_research_writing_z", "nature_research_writing")
    add_variant("pre_task_fit_weighted", frame, "semantic_distance", "pre_task_fit_z", "pre_task_fit")
    add_variant("primary_unweighted", frame, "semantic_distance", "nature_frequency_z", "nature_frequency", weights="unweighted")
    add_variant("centroid_dispersion_primary", frame, "centroid_dispersion", "nature_frequency_z", "nature_frequency")
    add_variant(
        "exclude_ai_heavy_categories",
        frame.loc[~frame["primary_category"].isin(AI_HEAVY_CATEGORIES)].copy(),
        "semantic_distance",
        "nature_frequency_z",
        "nature_frequency",
    )
    add_variant(
        "min_n20_primary",
        frame.loc[frame["n_papers"] >= 20].copy(),
        "semantic_distance",
        "nature_frequency_z",
        "nature_frequency",
    )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "q1_rq2_expanded_robustness_grid.csv", index=False)
    return out


def write_outputs(
    sample: pd.DataFrame,
    panel: pd.DataFrame,
    eligible: pd.DataFrame,
    did: pd.DataFrame,
    event: pd.DataFrame,
    pretrend: pd.DataFrame,
    placebos: pd.DataFrame,
    robustness: pd.DataFrame,
) -> dict[str, object]:
    def fmt(value: object, digits: int = 3) -> str:
        try:
            value = float(value)
        except Exception:
            return "--"
        if not np.isfinite(value):
            return "--"
        if abs(value) < 0.001 and value != 0:
            return f"{value:.2e}"
        return f"{value:.{digits}f}"

    def p_fmt(value: object) -> str:
        try:
            value = float(value)
        except Exception:
            return "--"
        if not np.isfinite(value):
            return "--"
        if value < 0.001:
            return "$<0.001$"
        return f"{value:.3f}"

    main = did.iloc[0]
    lead = pretrend.iloc[0]
    placebo_failures = int((placebos.get("status", pd.Series(dtype=str)) == "fail_same_sign_pre_cutoff").sum())
    robust_ok = robustness.loc[
        (robustness.get("status", "") == "ok")
        & (pd.to_numeric(robustness.get("coef", np.nan), errors="coerce") < 0)
        & (pd.to_numeric(robustness.get("p_cluster", np.nan), errors="coerce") < 0.05)
    ]
    gate = (
        "green_candidate"
        if main["coef"] < 0 and main["p_cluster"] < 0.05 and lead["joint_lead_p"] >= 0.05 and placebo_failures == 0
        else "not_green"
    )
    gate_display = gate.replace("_", " ")

    table = rf"""\begin{{table}}[t]
\centering
\scriptsize
\renewcommand{{\arraystretch}}{{1.16}}
\setlength{{\tabcolsep}}{{3.2pt}}
\caption{{Expanded monthly RQ2 event-study gate using the guarded STEM sample. The treatment gradient is external Nature-survey GenAI adoption mapped to arXiv macro fields; category and month fixed effects absorb persistent field differences and common time shocks.}}
\label{{tab:q1_expanded_monthly_rq2_event_study}}
\begin{{tabularx}}{{\linewidth}}{{>{{\raggedright\arraybackslash}}p{{0.23\linewidth}}rrrrrrr}}
\toprule
\rowcolor{{q1teallight}}
\textbf{{Model gate}} & \textbf{{$\hat\beta$}} & \textbf{{Cluster $p$}} & \textbf{{Lead $p$}} & \textbf{{Lead/Post}} & \textbf{{Placebo fails}} & \textbf{{Sig. robust}} & \textbf{{Decision}} \\
\midrule
Monthly semantic DiD & {fmt(main['coef'])} & {p_fmt(main['p_cluster'])} & {p_fmt(lead['joint_lead_p'])} & {fmt(lead['lead_to_post_ratio'])} & {placebo_failures} & {len(robust_ok)} & {gate_display} \\
\bottomrule
\end{{tabularx}}
\end{{table}}
"""
    (TABLES / "q1_T15_expanded_monthly_rq2_event_study.tex").write_text(table, encoding="utf-8")

    manifest = {
        "script": "q1_build_expanded_rq2_event_study.py",
        "date": "2026-04-24",
        "status": gate,
        "sample_rows": int(len(sample)),
        "panel_cells": int(len(panel)),
        "eligible_field_months": int(len(eligible)),
        "eligible_primary_categories": int(eligible["primary_category"].nunique()),
        "cutoff_policy": "first post month is 2022-12; 2022-11 is the reference bin because daily timing is unavailable in expanded sample",
        "primary_exposure": "Nature survey mean GenAI frequency score mapped to arXiv macro fields and z-scored",
        "main_semantic_did": did.to_dict(orient="records"),
        "pretrend": pretrend.to_dict(orient="records"),
        "placebo_failures": placebo_failures,
        "significant_negative_robustness_count": int(len(robust_ok)),
        "claim_boundary": "A green candidate supports RQ2 only as exposure-gradient evidence, not paper-level observed GenAI use.",
        "outputs": [
            "results/q1_rq2_expanded_month_panel.csv",
            "results/q1_rq2_expanded_exposure_table.csv",
            "results/q1_rq2_expanded_did_coefficients.csv",
            "results/q1_rq2_expanded_event_study_coefficients.csv",
            "results/q1_rq2_expanded_pretrend_diagnostics.csv",
            "results/q1_rq2_expanded_placebo_cutoffs.csv",
            "results/q1_rq2_expanded_robustness_grid.csv",
            "tables/q1_T15_expanded_monthly_rq2_event_study.tex",
        ],
    }
    (RESULTS / "q1_rq2_expanded_event_study_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    memo = f"""# Expanded Monthly RQ2 Event-Study Memo

Date: 2026-04-24

## Decision

`{gate}`

## Primary Model

- Unit: primary arXiv category by month.
- Outcome: mean pairwise SBERT distance within the category-month cell.
- Exposure: Nature-survey mean GenAI frequency score mapped to arXiv macro fields and z-scored.
- Fixed effects: primary category and month.
- Control: log number of papers in the cell.
- Weights: category-month paper count.

## Main Result

- DiD coefficient: {main['coef']:.6g}.
- Cluster p-value: {main['p_cluster']:.6g}.
- Joint pretrend p-value: {lead['joint_lead_p']:.6g}.
- Lead/post ratio: {lead['lead_to_post_ratio']:.3f}.
- Failed same-sign pre-cutoff placebos: {placebo_failures}.
- Significant negative robustness variants: {len(robust_ok)}.

## Interpretation Boundary

This is the first expanded-sample RQ2 result after the Colab embedding job. It
is stronger than the earlier sparse quarterly test because it uses monthly
field cells and an external adoption gradient. It still does not observe
paper-level GenAI use, so manuscript language must say "exposure-gradient
evidence" rather than "observed GenAI adoption caused".
"""
    (RESULTS / "q1_rq2_expanded_event_study_summary.txt").write_text(memo, encoding="utf-8")
    return manifest


def main() -> None:
    ensure_dirs()
    sample, embeddings = load_sample_and_embeddings()
    exposure = build_exposure_table(sample)
    panel = build_month_panel(sample, embeddings)
    eligible = eligible_panel(panel, exposure)
    if eligible["primary_category"].nunique() < 30:
        raise RuntimeError("Expanded monthly RQ2 panel has too few eligible categories.")

    _, did_row = run_did(eligible)
    did = pd.DataFrame([did_row])
    did.to_csv(RESULTS / "q1_rq2_expanded_did_coefficients.csv", index=False)

    _, event, pretrend_row = run_event_study(eligible)
    event.to_csv(RESULTS / "q1_rq2_expanded_event_study_coefficients.csv", index=False)
    pretrend = pd.DataFrame([pretrend_row])
    pretrend.to_csv(RESULTS / "q1_rq2_expanded_pretrend_diagnostics.csv", index=False)

    placebos = run_placebos(eligible, main_coef=float(did_row["coef"]))
    robustness = run_robustness(eligible)
    manifest = write_outputs(sample, panel, eligible, did, event, pretrend, placebos, robustness)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

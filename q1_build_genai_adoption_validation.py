from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

FREQ_COL = (
    "Generative AI This section now asks specifically about generative AI - that is, tools such as ChatGPT, "
    "GPT-4, PALM, MidJourney, LLaMA, or any products built using these tools. How often do you use these tools at work?"
)

PURPOSE_COLS = {
    "use_research": "To help do research:What do you use these tools for? (Choose all that apply).",
    "use_code": "To help write code:What do you use these tools for? (Choose all that apply).",
    "use_manuscripts": "To help write research manuscripts:What do you use these tools for? (Choose all that apply).",
    "use_review": "To help review research manuscripts:What do you use these tools for? (Choose all that apply).",
    "use_literature": "To conduct literature reviews:What do you use these tools for? (Choose all that apply).",
    "use_scientific_search": "Within scientific search engines:What do you use these tools for? (Choose all that apply).",
    "use_brainstorm": "To brainstorm research ideas:What do you use these tools for? (Choose all that apply).",
}

FIELD_COLS = {
    "agriculture": "Agriculture, veterinary or food science:Which best describes your research field?",
    "biological": "Biological sciences:Which best describes your research field?",
    "biomedical": "Biomedical, clinical, or health-related sciences:Which best describes your research field?",
    "chemical": "Chemical sciences:Which best describes your research field?",
    "earth": "Earth sciences:Which best describes your research field?",
    "economics": "Economics:Which best describes your research field?",
    "engineering": "Engineering:Which best describes your research field?",
    "environmental": "Environmental sciences and Ecology:Which best describes your research field?",
    "computing": "Computing or Information sciences:Which best describes your research field?",
    "mathematics": "Mathematics:Which best describes your research field?",
    "physical": "Physical sciences:Which best describes your research field?",
    "psychology": "Psychology:Which best describes your research field?",
    "social": "Social sciences:Which best describes your research field?",
    "humanities": "Humanities:Which best describes your research field?",
}

FREQ_SCORE = {
    "Never": 0.0,
    "I've used them only a few times": 0.25,
    "I use them occasionally": 0.50,
    "I use them more than once a week": 0.75,
    "I use them every day": 1.00,
}

FIELD_TO_ARXIV_MACROS = {
    "computing": ["cs"],
    "mathematics": ["math", "math-ph", "stat"],
    "physical": [
        "physics",
        "cond-mat",
        "astro-ph",
        "gr-qc",
        "hep-ex",
        "hep-lat",
        "hep-ph",
        "hep-th",
        "nucl-ex",
        "nucl-th",
        "nlin",
        "quant-ph",
    ],
    "engineering": ["eess"],
    "biological": ["q-bio"],
    "biomedical": ["q-bio"],
    "economics": ["econ", "q-fin"],
}


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    survey = pd.read_csv(DATA / "nature_survey.csv")
    survey["genai_frequency_score"] = survey[FREQ_COL].map(FREQ_SCORE)
    survey["genai_any_use"] = survey[FREQ_COL].notna() & (survey[FREQ_COL] != "Never")
    survey["genai_weekly_or_daily"] = survey[FREQ_COL].isin(
        ["I use them more than once a week", "I use them every day"]
    )
    for name, col in PURPOSE_COLS.items():
        survey[name] = survey[col].notna()

    rows = []
    for field_name, col in FIELD_COLS.items():
        mask = survey[col].notna()
        frame = survey.loc[mask].copy()
        if frame.empty:
            continue
        row = {
            "nature_field": field_name,
            "n_respondents": int(len(frame)),
            "mean_frequency_score": float(frame["genai_frequency_score"].mean()),
            "share_any_use": float(frame["genai_any_use"].mean()),
            "share_weekly_or_daily": float(frame["genai_weekly_or_daily"].mean()),
        }
        for name in PURPOSE_COLS:
            row[f"share_{name}"] = float(frame[name].mean())
        rows.append(row)

    by_field = pd.DataFrame(rows).sort_values("mean_frequency_score", ascending=False)
    by_field.to_csv(RESULTS / "q1_nature_genai_adoption_by_field.csv", index=False)

    mapped_rows = []
    for _, row in by_field.iterrows():
        macros = FIELD_TO_ARXIV_MACROS.get(row["nature_field"], [])
        for macro in macros:
            mapped = row.to_dict()
            mapped["arxiv_macro_field"] = macro
            mapped["mapping_status"] = "broad_validation_only"
            mapped_rows.append(mapped)
    mapped_df = pd.DataFrame(mapped_rows)
    mapped_df.to_csv(RESULTS / "q1_nature_arxiv_macro_exposure_validation.csv", index=False)

    manifest = {
        "script": "q1_build_genai_adoption_validation.py",
        "date": "2026-04-24",
        "source": "data/nature_survey.csv",
        "rows": int(len(survey)),
        "frequency_column": FREQ_COL,
        "purpose_columns": PURPOSE_COLS,
        "field_columns": FIELD_COLS,
        "outputs": [
            "results/q1_nature_genai_adoption_by_field.csv",
            "results/q1_nature_arxiv_macro_exposure_validation.csv",
        ],
        "claim_boundary": "validation-only broad-field adoption evidence; not paper-level GenAI adoption",
    }
    (RESULTS / "q1_genai_adoption_validation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

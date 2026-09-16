"""Create a reproducible title-plus-abstract embedding for the locked cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    cohort = pd.read_csv(DATA / "q1_arxiv_analysis_cohort.csv")
    source = pd.read_csv(DATA / "arxiv_meta_sampled.csv", usecols=["title", "abstract"])
    selected = source.iloc[cohort.row_id.to_numpy(dtype=np.int64)].reset_index(drop=True)
    texts = (selected.title.fillna("") + ". " + selected.abstract.fillna("")).tolist()

    model = SentenceTransformer(MODEL, revision=MODEL_REVISION, device="cpu")
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32, copy=False)

    output = DATA / "q1_locked_title_abstract_embeddings.npy"
    np.save(output, embeddings)
    row_manifest = RESULTS / "q1_locked_embedding_rows.csv"
    cohort[["id", "row_id", "analysis_date"]].to_csv(row_manifest, index=False)

    manifest = {
        "script": Path(__file__).name,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "text_input": "title + '. ' + abstract",
        "missing_text": "empty string",
        "tokenizer": "model default; truncation at the model maximum sequence length",
        "pooling": "model default mean pooling",
        "normalize_embeddings": True,
        "dtype": str(embeddings.dtype),
        "rows": int(embeddings.shape[0]),
        "dimension": int(embeddings.shape[1]),
        "batch_size": args.batch_size,
        "output": str(output.relative_to(ROOT)).replace("\\", "/"),
        "output_sha256": sha256(output),
        "row_manifest": str(row_manifest.relative_to(ROOT)).replace("\\", "/"),
    }
    (RESULTS / "q1_locked_embedding_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

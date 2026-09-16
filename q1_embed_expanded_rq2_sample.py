from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed the guarded expanded RQ2 arXiv STEM sample.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None, help="Optional pilot limit; omit for full sample.")
    parser.add_argument("--resume", action="store_true", help="Resume writing into existing memmap output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from sentence_transformers import SentenceTransformer

    sample_path = DATA / "q1_rq2_expanded_stem_sample.csv"
    sample = pd.read_csv(sample_path, dtype={"id": str})
    if args.limit is not None:
        sample = sample.head(args.limit).copy()

    model = SentenceTransformer(MODEL, revision=MODEL_REVISION)
    dim = int(model.get_sentence_embedding_dimension())
    output_name = "q1_rq2_expanded_embeddings.npy" if args.limit is None else f"q1_rq2_expanded_embeddings_pilot_{args.limit}.npy"
    output_path = DATA / output_name
    done_path = RESULTS / output_name.replace(".npy", "_done_rows.txt")

    total = len(sample)
    start = 0
    if args.resume and output_path.exists() and done_path.exists():
        start = int(done_path.read_text(encoding="utf-8").strip() or 0)

    mode = "r+" if output_path.exists() and args.resume else "w+"
    embeddings = np.lib.format.open_memmap(output_path, mode=mode, dtype="float32", shape=(total, dim))

    texts = (sample["title"].fillna("") + ". " + sample["abstract"].fillna("")).tolist()
    for begin in range(start, total, args.batch_size):
        end = min(begin + args.batch_size, total)
        batch = texts[begin:end]
        encoded = model.encode(
            batch,
            batch_size=args.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        embeddings[begin:end, :] = np.asarray(encoded, dtype="float32")
        embeddings.flush()
        done_path.write_text(str(end), encoding="utf-8")
        print(f"embedded {end}/{total}")
    del embeddings

    manifest = {
        "script": "q1_embed_expanded_rq2_sample.py",
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "text_input": "title + '. ' + abstract",
        "tokenizer_max_length": int(model.max_seq_length),
        "pooling": "model default mean pooling",
        "normalize_embeddings": True,
        "output_sha256": sha256(output_path),
        "sample": str(sample_path.relative_to(ROOT)),
        "output": str(output_path.relative_to(ROOT)),
        "rows": int(total),
        "dimension": dim,
        "batch_size": args.batch_size,
        "pilot_limit": args.limit,
        "status": "complete",
    }
    manifest_name = "q1_rq2_expanded_embedding_manifest.json" if args.limit is None else f"q1_rq2_expanded_embedding_pilot_{args.limit}_manifest.json"
    (RESULTS / manifest_name).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

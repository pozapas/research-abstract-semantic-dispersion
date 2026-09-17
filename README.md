# Has research become less diverse after ChatGPT?

This repository contains the code, derived aggregate results, and figures for
the study:

> Has research become less diverse after ChatGPT? A finite-corpus analysis of
> semantic dispersion, composition, and identification

The paper reports a descriptive change. It does not claim that generative AI
caused the change.

## Headline results

- The locked cohort has 31,802 papers.
- Exact mean pairwise distance falls from 1.353381 to 1.331600.
- The exact relative change is -1.609%.
- The main category-month panel has 9,331 common analysis cells.
- The full panel change is -0.007699.
- The marker-negative change is -0.005351, or 69.5% of the full change.
- The marker-composition-gap change is -0.002348, or 30.5%.
- The cell identity closes to a maximum residual of 4.44e-16.
- No field-level exposure design passes all identification checks.

The topic-clustering result is not stable. One reasonable setting changes its
sign. The repository keeps this result as a sensitivity check.

## Fast verification

Create the recorded environment:

    conda env create -f environment.yml
    conda activate semantic-dispersion

Check the tracked release:

    python run_pipeline.py verify

This command checks the required files, headline numbers, source manifest, and
common secret patterns. It does not download or rebuild the large source data.

## Source data

The arXiv source is the public Cornell University dataset:

https://www.kaggle.com/datasets/Cornell-University/arxiv

The local analysis snapshot was retrieved on 23 April 2026. It has:

- File name: `arxiv-metadata-oai-snapshot.json`
- Size: 5,223,246,694 bytes
- SHA-256: `fb088d1d7ae7cf2772889dc2f467692bd3bed6a4c77484982eacab223e2bf5c5`

The upstream dataset changes over time. A current download can have a different
hash and can produce a different cohort. The tracked row manifest and aggregate
results preserve the reported analysis state.

Place the exact source file at:

    data/archive/arxiv-metadata-oai-snapshot.json

This repository does not redistribute:

- the 5.22 GB arXiv JSON file;
- the 2.09 GB extracted text CSV;
- the 521 MB expanded text sample;
- the 48.8 MB and 653.9 MB embedding arrays;
- Nature survey microdata; or
- credentials and API keys.

The arXiv source is CC0. Other source files keep their original terms.

## Full rebuild

The complete run needs about 12 GB of free work space. The expanded embedding
step is much faster on a CUDA GPU.

Build the extracted metadata and the locked cohort:

    python run_pipeline.py prepare

Build the locked embeddings and RQ1 outputs:

    python run_pipeline.py locked --batch-size 128

Build the expanded sample, embeddings, panel, and exposure diagnostics:

    python run_pipeline.py expanded --batch-size 256

Build all figures:

    python run_pipeline.py figures

Run all stages:

    python run_pipeline.py all

The tracked Nature field aggregates are sufficient for the reported expanded
analysis. To rebuild those aggregates from microdata, obtain the survey under
its source terms, save it as `data/nature_survey.csv`, and run:

    python q1_build_genai_adoption_validation.py

## Cohort rules

The eligible arXiv category prefixes are:

- `cs.`
- `math.`
- `stat.`
- `physics.`
- `q-bio.`
- `eess.`

The locked cohort starts with a 50,000-row sample. It uses seed 42. It then uses
the first-submission date and keeps papers from 2018 through 2025.

The expanded cohort uses a deterministic BLAKE2 ordering key with seed 20260424.
It keeps at most 50 papers in each primary-category-by-month cell.

## Text representation

Both cohorts use:

- model: `sentence-transformers/all-MiniLM-L6-v2`;
- revision: `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`;
- input: title, a full stop, and abstract;
- maximum length: 256 tokens;
- pooling: model-default mean pooling;
- output: 384 dimensions, L2 normalized, `float32`; and
- distance: Euclidean distance.

## Repository map

- `00_prepare_data.py`: Extract source metadata and rebuild the locked cohort.
- `q1_embed_locked_cohort.py`: Build the locked embedding matrix.
- `q1_exact_semantic_distance.py`: Compute the exact all-pairs result.
- `q1_locked_diagnostics.py`: Build repeated-sample, topology, time-series,
  and projection outputs.
- `q1_topic_stability.py`: Test topic-cluster settings and noise rules.
- `q1_prepare_expanded_rq2_cohort.py`: Build the capped expanded cohort.
- `q1_embed_expanded_rq2_sample.py`: Build the expanded embeddings.
- `q1_build_expanded_rq2_event_study.py`: Build the main monthly panel.
- `q1_rq2_phase*.py`: Run field-level exposure diagnostics.
- `q1_rq2_decomposition_verify.py`: Verify the exact pair identity.
- `q1_make_figures.py`: Build the publication figures.
- `results/`: Curated aggregate outputs and manifests.
- `figures/`: PNG and vector PDF figures.

## License

Original code is available under the MIT License. Third-party data, pretrained
models, and other third-party files keep their own licenses and terms.

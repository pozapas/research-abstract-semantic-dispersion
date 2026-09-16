# Data files

This repository does not store the 5.22 GB arXiv JSON snapshot, the 2.09 GB
extracted CSV, the 521 MB expanded text sample, the embedding arrays, the
Nature survey microdata, or any API key.

For an exact rebuild, place the arXiv file at:

    data/archive/arxiv-metadata-oai-snapshot.json

The expected SHA-256 value is in `source_manifest.json`.

The tracked file `q1_arxiv_analysis_cohort.csv` contains the locked row
identifiers, categories, titles, dates, and source row indices. It does not
contain abstracts. The tracked `google_trends.csv` file is the monthly public
search series used in the reported task-fit analysis.

The field-level Nature aggregates are in `../results/`. The restricted
microdata are not redistributed. To rebuild those aggregates, obtain the source
survey under its original terms and place it at `data/nature_survey.csv`.

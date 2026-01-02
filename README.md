# ECB Communication: Replication & Extensions (Amaya & Filbien, 2015)

A reproducible project to replicate and extend empirical results on ECB press-conference communication, from text-derived measures to event-study and regression outputs. The pipeline builds text-based measures (similarity and sentiment), merges them with market reactions (CAR), constructs macro controls, and produces key regression tables and summary statistics. All data artifacts live at the project root and are reused across replication and extension workflows.

## Research Context

The replication component follows the methodology in Amaya & Filbien (2015), focusing on:
- Consecutive-text similarity (Jaccard similarity on stemmed bigrams)
- Dictionary-based pessimism using the Loughran–McDonald lexicon
- Event-study CAR around ECB communication events using a constant-mean return model
- Table-style regressions and descriptive statistics over an observation window

The extension component adds alternative similarity measures (TF-IDF cosine) and corresponding regression variants.

The replication pipeline is parameterized to run on the original paper window (1999–2013) or on an extended sample up to 2025.

## Quickstart

1. Create the expected data folders at the project root (if they don’t exist yet):
   - `data_raw/`, `data_clean/`, `data_features/`, `outputs/`

2. Place required external inputs in `data_raw/`:
   - `^SX5E data.xlsx`
   - `Loughran-McDonald_MasterDictionary_1993-2024.csv`
   - `AMECO-AVGDGP-EA12.csv`
   - `HICP_data_base100_2005.csv`
   - `MRO.csv`

3. Install dependencies (Python 3.10+ recommended):
   - Core packages used across the pipeline: pandas, numpy, matplotlib, requests, beautifulsoup4, nltk, statsmodels
   - For extensions: scikit-learn
   - For scraping (optional): selenium + a compatible browser driver (e.g., GeckoDriver for Firefox)

4. Run the replication pipeline:
   - Open `replication/main.py` and run it.
   - Recommended default for reproducible runs: set `RUN_SCRAPING = False` unless you explicitly want to scrape the ECB website.
   - If `RUN_SCRAPING = False`, ensure `data_raw/ecb_statements_raw.csv` already exists (from a prior run or shared artifact).

5. Run the extension pipeline:
   - Open `extension/main.py` and run it.
   - Extensions reuse replication artifacts from `data_clean/` and write additional features to `data_features/`.
   
Optional: configure the analysis window (used by plots, regressions, and any windowed diagnostics):
- Set `START_DATE` / `END_DATE` in `replication/main.py` and `extension/main.py`
- These are passed to downstream scripts as `ECB_START_DATE` / `ECB_END_DATE`
- Typical usage:
  - Paper window: `1999-01-01` → `2013-12-31`
  - Extended window: up to `2025-12-31` (or any end date supported by your available inputs)

## Research Context

The replication component follows the methodology in Amaya & Filbien (2015), focusing on:
- Consecutive-text similarity (Jaccard similarity on stemmed bigrams)
- Dictionary-based pessimism using the Loughran–McDonald lexicon
- Event-study CAR around ECB communication events using a constant-mean return model
- Table-style regressions and descriptive statistics over an observation window

The extension component adds alternative similarity measures (TF-IDF cosine) and corresponding regression variants.

The replication pipeline is parameterized to run on the original paper window (1999–2013) or on an extended sample up to 2025 by adjusting `START_DATE` / `END_DATE` (propagated as `ECB_START_DATE` / `ECB_END_DATE`).

## Quickstart

1. Create the expected data folders at the project root (if they don’t exist yet):
   - `data_raw/`, `data_clean/`, `data_features/`, `outputs/`

2. Place required external inputs in `data_raw/`:
   - `^SX5E data.xlsx`
   - `Loughran-McDonald_MasterDictionary_1993-2024.csv`
   - `AMECO-AVGDGP-EA12.csv`
   - `HICP_data_base100_2005.csv`
   - `MRO.csv`

3. Install dependencies (Python 3.10+ recommended):
   - Core packages used across the pipeline: pandas, numpy, matplotlib, requests, beautifulsoup4, nltk, statsmodels
   - For extensions: scikit-learn
   - For scraping (optional): selenium + a compatible browser driver (e.g., GeckoDriver for Firefox)

4. Run the replication pipeline:
   - Open `replication/main.py` and run it.
   - Recommended default for reproducible runs: set `RUN_SCRAPING = False` unless you explicitly want to scrape the ECB website.
   - If `RUN_SCRAPING = False`, ensure `data_raw/ecb_statements_raw.csv` already exists (from a prior run or shared artifact).

5. Run the extension pipeline:
   - Open `extension/main.py` and run it.
   - Extensions reuse replication artifacts from `data_clean/` and write additional features to `data_features/`.

Optional: set the analysis window (used by plots/regressions) in the orchestrators:
- `START_DATE` / `END_DATE` in `replication/main.py` and `extension/main.py`
- These are propagated to scripts via `ECB_START_DATE` / `ECB_END_DATE`.


## Repository Layout


All datasets and outputs are intentionally stored at the project root to ensure both replication and extensions can share the same artifacts.

```bash
project_root/
├─ replication/                         # Paper replication pipeline
│  ├─ main.py                           # Orchestrates replication steps
│  ├─ 1_scraping_ecb.py
│  ├─ 2_scraping_statements.py
│  ├─ 3_filter_raw_before_preprocess.py
│  ├─ 4_pre-process.py
│  ├─ 5_similarity_jaccard_bigrams.py
│  ├─ 6_sentiment_pessimism_lm.py
│  ├─ 7_event_study_car.py
│  ├─ 7b_prepare_controls_month_end.py
│  ├─ 8_regressions-table3.py
│  ├─ 8_regressions-table4.py
│  └─ 9_summary_table2.py
│
├─ extension/                           # Extensions (alternative similarity + tables)
│  ├─ main.py
│  ├─ E5_tfidf_cosine.py
│  ├─ E8_merge_car_pessimism_similarity.py
│  └─ E9_tfidf_tables_concise.py
│
├─ extra_code/                          # Non-core tools and fallbacks
│  ├─ 01_collect_urls_parallel.py
│  └─ 6_cosine_compare.py
│
├─ data_raw/                            # Raw inputs + scraped artifacts (shared)
├─ data_clean/                          # Clean datasets (shared)
├─ data_features/                       # Additional engineered features (extensions)
└─ outputs/                             # Final tables/figures (shared)
```


## Data Contracts

The pipeline assumes a clear separation of responsibilities:
- `data_raw/` contains scraped and externally sourced raw files
- `data_clean/` contains preprocessed and analysis-ready datasets
- `data_features/` contains additional feature sets (e.g., TF-IDF similarity)
- `outputs/` contains final tables and figures

### Required External Inputs (place in `data_raw/`)

These are not produced by the code and must be provided:
- `^SX5E data.xlsx`  
  EuroStoxx 50 price data used for the event study.
- `Loughran-McDonald_MasterDictionary_1993-2024.csv`  
  Loughran–McDonald dictionary used for pessimism measurement.
- `AMECO-AVGDGP-EA12.csv`  
  Annual output gap (AMECO-style export).
- `HICP_data_base100_2005.csv`  
  Monthly HICP index (base 100).
- `MRO.csv`  
  Main Refinancing Operations rate series.

### Scraped Outputs (produced by replication scraping steps)

- `data_raw/ecb_speech_urls.csv`  
  Columns: `date`, `url`
- `data_raw/ecb_statements_raw.csv`  
  Columns: `date`, `title`, `subtitle`, `url`, `method`, `statement_text`, `qa_text`, `error`

## Replication Pipeline

### Orchestrator: `replication/main.py`

`replication/main.py` runs scripts in dependency order and propagates a shared analysis window via environment variables:
- `ECB_START_DATE`
- `ECB_END_DATE`

Key operational toggles are centralized in `CONFIG`, including disabling scraping for offline reproducibility.

### Step Outputs (high-level)

1. URL collection (optional scraping)  
   `1_scraping_ecb.py` → `data_raw/ecb_speech_urls.csv`

2. Transcript scraping (optional scraping)  
   `2_scraping_statements.py` → `data_raw/ecb_statements_raw.csv`

3. Filtering non-target documents  
   `3_filter_raw_before_preprocess.py` →  
   `data_raw/ecb_statements_raw_filtered.csv`  
   `data_raw/ecb_statements_raw_removed.csv`

4. Text preprocessing (tokens + Porter stems)  
   `4_pre-process.py` → `data_clean/ecb_statements_preprocessed.csv`

5. Similarity (paper measure)  
   `5_similarity_jaccard_bigrams.py` →  
   `data_clean/ecb_similarity_jaccard_bigrams.csv` (+ optional plot in `outputs/`)

6. Pessimism (Loughran–McDonald)  
   `6_sentiment_pessimism_lm.py` →  
   `data_clean/ecb_pessimism_lm.csv`  
   `data_clean/ecb_statements_with_pessimism.csv` (+ optional plot in `outputs/`)

7b. Macro controls (EOM alignment)  
   `7b_prepare_controls_month_end.py` → `data_clean/controls_month_end.csv`

7. Event study (CAR) and merge with pessimism  
   `7_event_study_car.py` →  
   `data_clean/ecb_event_study_car.csv`  
   `data_clean/ecb_pessimism_with_car.csv`

8. Table-style regressions  
   `8_regressions-table3.py` → `outputs/table3_*.csv` + regression dataset  
   `8_regressions-table4.py` → `outputs/table4_*.csv` + regression dataset

9. Summary statistics  
   `9_summary_table2.py` → `outputs/table2_summary_stats.csv`

## Extension Pipeline

Extensions reuse replication artifacts from `data_clean/` and write new features to `data_features/`.

### Orchestrator: `extension/main.py`

Runs extension steps in dependency order with the same shared window variables:
- `ECB_START_DATE`
- `ECB_END_DATE`

### Extension Steps

1. TF-IDF cosine similarity (consecutive)  
   `5_tfidf_cosine.py` → `data_features/ecb_similarity_tfidf.csv`

2. Merge analysis dataset (CAR/pessimism + TF-IDF similarity)  
   `8_merge_car_pessimism_similarity.py` → `data_clean/ecb_analysis_dataset.csv`

3. TF-IDF variants of Table 3 / Table 4 regressions (log + z specs)  
   `8_tfidf_tables_concise.py` → `outputs/` (+ regression datasets)

## Extra Code (Fallbacks and Diagnostics)

The `extra_code/` directory contains non-core tooling intended for robustness and exploratory checks:
- `01_collect_urls_parallel.py`  
  A fallback URL discovery strategy (parallel pattern probing) if Selenium-based scraping is not feasible.
- `6_cosine_compare.py`  
  A compact diagnostic comparing CountVectorizer cosine vs TF-IDF cosine, producing a CSV and figures under `data_features/`.

These scripts follow the same “project-root data layout” convention.

## Environment Setup

Recommended: Python 3.10+.

Core dependencies typically include:
- pandas, numpy, matplotlib
- requests, beautifulsoup4
- nltk (stopwords), scikit-learn (extensions), statsmodels (regressions)

Scraping dependencies (only if enabled):
- selenium
- a compatible browser driver (e.g., GeckoDriver for Firefox)

Tip: for reproducible replication in shared settings (CI, offline runs), disable scraping and ship the scraped CSVs in `data_raw/`.

## Running the Pipelines

Replication (end-to-end):
- Run `replication/main.py` from your IDE or CLI.
- Configure:
  - `START_DATE`, `END_DATE`
  - `RUN_SCRAPING` (True/False)
  - `REQUIRE_RAW_IF_NO_SCRAPE`

Extensions:
- Run `extension/main.py`.
- Configure:
  - `START_DATE`, `END_DATE`
  - step toggles for TF-IDF similarity, merge, and TF-IDF tables

## Design Principles

- Single source of truth for data artifacts: root-level `data_raw/`, `data_clean/`, `data_features/`, `outputs/`
- Pipelines are explicit and ordered via orchestrator mains
- Scripts are runnable independently but remain consistent via root-relative data contracts
- Extensions never duplicate replication outputs; they consume them

## Citation

Amaya, D., & Filbien, J.-Y. (2015). *The similarity of ECB’s communication*. Finance Research Letters, 13, 234–242.

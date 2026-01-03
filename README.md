# The similarity of ECB’s communication: Replication & Extensions (Amaya & Filbien, 2015)

A reproducible project to replicate and extend empirical results on the similarity of ECB's press-conference communication, from text-derived measures to event-study and regression outputs. The pipeline builds text-based measures (similarity and sentiment), merges them with market reactions (CAR), constructs macro controls, and produces key regression tables and summary statistics.

## Research Context

The replication component follows Amaya & Filbien (2015), focusing on:
- Consecutive-text similarity (Jaccard similarity on stemmed bigrams)
- Dictionary-based pessimism using the Loughran–McDonald lexicon
- Event-study CAR around ECB communication events using a constant-mean return model
- Table-style regressions and descriptive statistics over an observation window

The extension component adds alternative similarity measures (TF-IDF cosine) and corresponding regression variants.

The pipeline is parameterized to run on the original paper window (1999–2013) or on an extended sample (up to 2025), using `ECB_START_DATE` / `ECB_END_DATE`.

## Quickstart

1. Create the expected data folders at the project root (if they don’t exist yet):
   - `data_raw/`, `data_clean/`, `data_features/`, `outputs/plots/`, `outputs/tables/`

2. Place required external inputs in `data_raw/`:
   - `^SX5E data.xlsx`
   - `Loughran-McDonald_MasterDictionary_1993-2024.csv`
   - `AMECO-AVGDGP-EA12.csv`
   - `HICP_data_base100_2005.csv`
   - `MRO.csv`

3. Install dependencies (Python 3.10+ recommended):
   - Core: pandas, numpy, matplotlib, requests, beautifulsoup4, nltk, statsmodels
   - Extensions: scikit-learn
   - Scraping (optional): selenium + a compatible driver (e.g., GeckoDriver for Firefox)

4. Run the replication pipeline:
   - Run `replication/main.py`.
   - Recommended: set `RUN_SCRAPING = False` for offline reproducibility.
   - If `RUN_SCRAPING = False`, ensure `data_raw/ecb_statements_raw.csv` exists.

5. Run the extension pipeline:
   - Run `extension/main.py`.
   - Extensions reuse replication artifacts from `data_clean/` and write additional features to `data_features/`.

### Analysis window (optional)

Set the analysis window in the orchestrators (propagated to scripts as env vars):
- `ECB_START_DATE`
- `ECB_END_DATE`

Typical usage:
- Paper window: `1999-01-01` → `2013-12-31`
- Extended window: up to `2025-12-31`


## Repository Layout


All datasets and outputs are intentionally stored at the project root to ensure both replication and extensions can share the same artifacts.

```bash
project_root/
├─ replication/
│  ├─ main.py
│  ├─ 1_scraping_ecb.py
│  ├─ 2_scraping_statements.py
│  ├─ 3_filter_raw_before_preprocess.py
│  ├─ 4_pre-process.py
│  ├─ 05_similarity_jaccard_bigrams.py
│  ├─ 6_sentiment_pessimism_lm.py
│  ├─ 7_event_study_car.py
│  ├─ 7b_prepare_controls_month_end.py
│  ├─ 8_regressions-table3.py
│  ├─ 8_regressions-table4.py
│  └─ 9_summary_table2.py
│
├─ extension/
│  ├─ main.py
│  ├─ E5_tfidf_cosine.py
│  ├─ E8_merge_car_pessimism_similarity.py
│  └─ E9_tfidf_tables_concise.py
│
├─ extra_code/
│  ├─ 01_collect_urls_parallel.py
│  └─ 6_cosine_compare.py
│
├─ data_raw/
├─ data_clean/
├─ data_features/
└─ outputs/
   ├─ plots/
   └─ tables/
```
## Data Contracts

- `data_raw/`: scraped + external raw inputs  
- `data_clean/`: cleaned / analysis-ready datasets  
- `data_features/`: engineered features (extensions/diagnostics)  
- `outputs/plots/`: figures  
- `outputs/tables/`: final tables (CSV)  

### Required External Inputs (`data_raw/`)

- `^SX5E data.xlsx` (EuroStoxx 50 prices)  
- `Loughran-McDonald_MasterDictionary_1993-2024.csv` (LM dictionary)  
- `AMECO-AVGDGP-EA12.csv` (annual output gap)  
- `HICP_data_base100_2005.csv` (monthly HICP)  
- `MRO.csv` (MRO rate series)  

### Scraped Outputs (`data_raw/`)

- `data_raw/ecb_speech_urls.csv` (columns: `date`, `url`)  
- `data_raw/ecb_statements_raw.csv` (columns: `date`, `title`, `subtitle`, `url`, `method`, `statement_text`, `qa_text`, `error`)  

## Replication Pipeline

### Orchestrator: `replication/main.py`

Runs scripts in dependency order and propagates the window via:
- `ECB_START_DATE`
- `ECB_END_DATE`

### Step outputs (high-level)

1. URL collection (optional)  
   `1_scraping_ecb.py` → `data_raw/ecb_speech_urls.csv`

2. Transcript scraping (optional)  
   `2_scraping_statements.py` → `data_raw/ecb_statements_raw.csv`

3. Filtering non-target documents  
   `3_filter_raw_before_preprocess.py` →  
   - `data_raw/ecb_statements_raw_filtered.csv`  
   - `data_raw/ecb_statements_raw_removed.csv`

4. Text preprocessing (tokens + Porter stems)  
   `4_pre-process.py` → `data_clean/ecb_statements_preprocessed.csv`

5. Similarity (paper measure: Jaccard bigrams)  
   `05_similarity_jaccard_bigrams.py` →  
   - `data_clean/ecb_similarity_jaccard_bigrams.csv`  
   - (optional plot) `outputs/plots/similarity_jaccard_bigrams_<START>_<END>.png`

6. Pessimism (Loughran–McDonald)  
   `6_sentiment_pessimism_lm.py` →  
   - `data_clean/ecb_pessimism_lm.csv`  
   - `data_clean/ecb_statements_with_pessimism.csv`  
   - (optional plot) `outputs/plots/pessimism_lm_<START>_<END>.png`

7b. Macro controls (EOM alignment)  
   `7b_prepare_controls_month_end.py` → `data_clean/controls_month_end.csv`

7. Event study (CAR) + merge with pessimism  
   `7_event_study_car.py` →  
   - `data_clean/ecb_event_study_car.csv`  
   - `data_clean/ecb_pessimism_with_car.csv`

8. Regression tables (paper-style)  
   - `8_regressions-table3.py` → `outputs/tables/table3_similarity_regressions.csv`  
   - `8_regressions-table4.py` → `outputs/tables/table4_absCAR_regressions.csv`

9. Summary statistics  
   `9_summary_table2.py` → `outputs/tables/table2_summary_stats.csv`

## Extension Pipeline

Extensions reuse replication artifacts from `data_clean/` and write features to `data_features/`.

### Orchestrator: `extension/main.py`

Runs extension steps in dependency order with the same window variables:
- `ECB_START_DATE`
- `ECB_END_DATE`

### Extension steps

1. TF-IDF cosine similarity (consecutive)  
   `E5_tfidf_cosine.py` →  
   - `data_features/ecb_similarity_tfidf.csv`  
   - `outputs/plots/ts_sim_tfidf.png`

2. Merge analysis dataset (CAR/pessimism + TF-IDF similarity)  
   `E8_merge_car_pessimism_similarity.py` → `data_clean/ecb_analysis_dataset.csv`

3. TF-IDF regression tables (Table 3 + Table 4, log + z)  
   `E9_tfidf_tables_concise.py` →  
   - `outputs/tables/table3_similarity_regressions_tfidf.csv`  
   - `outputs/tables/table4_absCAR_regressions_tfidf.csv`  
   - `outputs/tables/table4_absCAR_regressions_tfidf_z.csv`

## Extra Code (Fallbacks and Diagnostics)

- `extra_code/01_collect_urls_parallel.py`: fallback URL discovery (parallel probing) if Selenium scraping fails.  
- `extra_code/6_cosine_compare.py`: compares CountVectorizer vs TF-IDF consecutive cosine similarity:  
  - CSV: `data_features/ecb_similarity_cosines.csv`  
  - Figures: `outputs/plots/cosine_series.png`, `outputs/plots/cosine_scatter.png`  

## Environment Setup

Recommended: Python 3.10+.

Core dependencies:
- pandas, numpy, matplotlib
- requests, beautifulsoup4
- nltk (stopwords), statsmodels (regressions), scikit-learn (extensions)

Scraping dependencies (only if enabled):
- selenium + driver (e.g., GeckoDriver)

## Citation

Amaya, D., & Filbien, J.-Y. (2015). *The similarity of ECB’s communication*. Finance Research Letters, 13, 234–242.



# The similarity of ECB’s communication: Replication & Extensions (Amaya & Filbien, 2015)

A reproducible project to replicate and extend empirical results on the similarity of ECB's press-conference communication. The pipeline builds text-based measures (similarity and sentiment), merges them with market reactions (CAR), constructs macro controls, and produces key regression tables and summary statistics.

This repo includes multiple extension modules (TF-IDF similarity, uncertainty LM, learning speed, and a “good vs bad news” asymmetry extension), with a clearer split between:

- `data_clean/`: cleaned / baseline replication artifacts (incl. the **raw Jaccard similarity file** and **month-end controls**)
- `data_features/`: engineered features and extension outputs (uncertainty LM features, pessimism LM features, merged uncertainty datasets, etc.)

---

## Research Context

The replication component follows Amaya & Filbien (2015), focusing on:

- Consecutive-text similarity (Jaccard similarity on stemmed bigrams)
- Dictionary-based pessimism using the Loughran–McDonald lexicon
- Event-study CAR around ECB communication events using a constant-mean return model
- Table-style regressions and descriptive statistics over an observation window

The extensions add:

- TF-IDF cosine similarity (alternative similarity measure)
- **Uncertainty LM** indicator + event-study merge + regression tables
- Learning speed (Jaccard-based) + plots + regressions
- Pessimism asymmetry split (bad vs good news) + Table-4 style regressions with interactions

The pipeline is parameterized to run on the original paper window (1999–2013) or on an extended sample (up to 2025), using `ECB_START_DATE` / `ECB_END_DATE`.

---

## Quickstart

1. Create folders at the project root (if needed):

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
   - Scraping (optional): selenium + driver (e.g., GeckoDriver)

4. Run the **replication pipeline**:

   - `python replication/main.py`
   - Recommended: set `RUN_SCRAPING = False` for offline reproducibility.
   - If scraping is off, ensure `data_raw/ecb_statements_raw.csv` exists.

5. Run **extensions** (pick what you need):

   - TF-IDF similarity extension: `python extension/main.py`
   - Uncertainty LM extension: `python extension/uncertainty/main.py`
   - Learning speed extension: `python extension/learning_speed/main.py`
   - Good vs bad news asymmetry scripts: run scripts under `extension/separate_goodnews_badnews/`

### Analysis window (optional)

Set the analysis window in orchestrators (propagated as env vars):

- `ECB_START_DATE`
- `ECB_END_DATE`

Typical usage:

- Paper window: `1999-01-01` → `2013-12-31`
- Extended window: up to `2025-12-31`

---

## Repository Layout

All datasets and outputs are stored at the project root so replication + extensions share artifacts.

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
│
│  ├─ tfidf_cosine/
│  │  ├─ main.py                      # TF-IDF cosine extension orchestrator
│  │  ├─ E5_tfidf_cosine.py
│  │  ├─ E8_merge_car_pessimism_similarity.py
│  │  └─ E9_tfidf_tables_concise.py
│  │
│  ├─ uncertainty/
│  │  ├─ main.py                      # uncertainty-only pipeline orchestrator
│  │  ├─ E6_uncertainty_lm.py
│  │  ├─ E7_event_study_car_uncertainty.py
│  │  └─ E9_regression_uncertainty.py
│  │
│  ├─ learning_speed/
│  │  ├─ main.py                      # learning-speed orchestrator
│  │  ├─ E5_jaccard_learning_speed.py
│  │  ├─ E5c_plot_learning_speed.py
│  │  └─ E9b_regression_jaccard_learning_speed.py
│  │
│  └─ separate_goodnews_badnews/
│     ├─ E6_sentiment_pessimism_lm.py  # pessimism split outputs to data_features/
│     └─ E8_regressions-table4_*.py    # asymmetry / interactions regressions
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
- `data_clean/`: cleaned / baseline replication artifacts (incl. raw Jaccard similarity + month-end controls)  
- `data_features/`: engineered features & extension outputs  
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

---

## Replication Pipeline

### Orchestrator: `replication/main.py`

Runs scripts in dependency order and propagates the window via:
- `ECB_START_DATE`
- `ECB_END_DATE`

### Key outputs (replication)

- **Text preprocessing**  
  `4_pre-process.py` → `data_clean/ecb_statements_preprocessed.csv`

- **Similarity (paper measure: raw Jaccard bigrams)**  
  `05_similarity_jaccard_bigrams.py` →  
  - `data_clean/ecb_similarity_jaccard_bigrams.csv`  
  - (optional) `outputs/plots/similarity_jaccard_bigrams_<START>_<END>.png`

- **Pessimism (LM)**  
  `6_sentiment_pessimism_lm.py` →  
  - `data_clean/ecb_pessimism_lm.csv`  
  - `data_clean/ecb_statements_with_pessimism.csv`

- **7b. Macro controls (month-end)**  
  `7b_prepare_controls_month_end.py` → `data_clean/controls_month_end.csv`

- **Event study (CAR) + merge with pessimism**  
  `7_event_study_car.py` →  
  - `data_clean/ecb_event_study_car.csv`  
  - `data_clean/ecb_pessimism_with_car.csv`

- **8–9. Tables**  
  - `outputs/tables/table3_similarity_regressions.csv`  
  - `outputs/tables/table4_absCAR_regressions.csv`  
  - `outputs/tables/table2_summary_stats.csv`

---

## Extension Modules

### 1) TF-IDF cosine similarity (extension baseline)

Orchestrator: `extension/tfidf_cosine/main.py`

Outputs (examples):
- `data_features/ecb_similarity_tfidf.csv`
- plots in `outputs/plots/`
- TF-IDF regression tables in `outputs/tables/`

### 2) Uncertainty LM extension (uncertainty-only)

Orchestrator: `extension/uncertainty/main.py`

- **E6_uncertainty_lm.py**  
  Input: `data_clean/ecb_statements_preprocessed.csv`  
  Output:  
  - `data_features/ecb_uncertainty_lm.csv`  
  - `data_features/ecb_statements_with_uncertainty.csv`

- **E7_event_study_car_uncertainty.py**  
  Inputs:  
  - `data_raw/^SX5E data.xlsx`  
  - `data_features/ecb_uncertainty_lm.csv`  
  Outputs:  
  - `data_features/ecb_event_study_car_uncertainty.csv`  
  - `data_features/ecb_uncertainty_with_car.csv`

- **E9_regression_uncertainty.py (uncertainty-only tables)**  
  Inputs:  
  - `data_clean/ecb_similarity_jaccard_bigrams.csv` (raw Jaccard file)  
  - `data_clean/controls_month_end.csv`  
  - `data_features/ecb_uncertainty_with_car.csv`  
  Outputs:  
  - `outputs/tables/table4_absCAR_regressions_uncertainty.csv`  
  - `outputs/tables/table2_summary_stats_uncertainty.csv`

### 3) Learning speed extension (Jaccard-based)

Orchestrator: `extension/learning_speed/main.py`

Uses replication artifacts and generates learning-speed outputs + plots + regressions (script-specific outputs).

### 4) Good vs Bad news asymmetry extension

Folder: `extension/separate_goodnews_badnews/`

- **E6_sentiment_pessimism_lm.py (asymmetry split)**  
  Inputs:  
  - `data_clean/ecb_statements_preprocessed.csv`  
  Output features:  
  - `data_features/ecb_pessimism_lm.csv`  
  - `data_features/ecb_statements_with_pessimism.csv`

- **E8_regressions-table4_*.py (asymmetry regressions)**  
  Inputs (current structure):  
  - `data_clean/ecb_similarity_jaccard_bigrams.csv`  
  - `data_clean/controls_month_end.csv`  
  - `data_clean/ecb_pessimism_with_car.csv` (from replication event study)  
  Output:  
  - `outputs/tables/Etable4_absCAR_regressions_asymmetry.csv` (name may vary by script)

---

## Extra Code (Fallbacks and Diagnostics)

- `extra_code/01_collect_urls_parallel.py`: fallback URL discovery  
- `extra_code/6_cosine_compare.py`: cosine similarity diagnostics  
  - CSV: `data_features/ecb_similarity_cosines.csv`  
  - Plots: `outputs/plots/cosine_series.png`, `outputs/plots/cosine_scatter.png`

---

## Environment Setup

Recommended: Python 3.10+.

Core dependencies:
- pandas, numpy, matplotlib  
- requests, beautifulsoup4  
- nltk (stopwords), statsmodels (regressions), scikit-learn (extensions)

Scraping dependencies (only if enabled):
- selenium + driver (e.g., GeckoDriver)

---

## Citation

Amaya, D., & Filbien, J.-Y. (2015). *The similarity of ECB’s communication*. Finance Research Letters, 13, 234–242.

# Choosing a Classification Method for Measuring Government Criticism in Armenian Civil Society Texts

**Albert Ananyan**
Machine Learning in Political Science (POLI_SCI 490) — Winter 2026, Northwestern University

## Summary

This project compares five supervised classification methods across two text representations to identify the best approach for coding approximately 4,100 publications from five Armenian watchdog civil society organizations (CSOs) as "critical" or "non-critical" of the Armenian government. The corpus spans 2014-2024 and covers the period before and after Armenia's 2018 Velvet Revolution.

The central finding is that the choice of text representation (bag-of-words vs. contextual BERT embeddings) matters substantially more than the choice of classifier. All BERT-based classifiers produce substantively similar criticality trajectories when applied to the full corpus.

## Folder Structure

```
cso_classification_project/
├── README.md
├── analysis.qmd               # Main analysis (Quarto document)
├── finetune_hyebert.py         # Fine-tuning script for HyeBERT
├── final_paper_ananyan.pdf     # Final paper
├── references.bib
├── fig1-fig9 PNG files         # Figures used in the paper
└── data/
    ├── coded_corpus_combined.csv     # Hand-coded training corpus (642 docs)
    ├── uic_publications.csv
    ├── hcav_publications.csv
    ├── hahr_publications.csv
    ├── csi_publications.csv
    ├── ti_armenia_publications.csv
    └── stopwords-hy-clean.json       # Custom Armenian stopword list
```

## Datasets

| File | Description |
|------|-------------|
| `coded_corpus_combined.csv` | Hand-coded training data from all 5 CSOs. Each document coded for `crit_armenian_human` (critical of Armenian government) and `crit_foreign_human` (critical of foreign actors). |
| `*_publications.csv` | Full publication archives scraped from each CSO's website. Contains title, date, link, and content. |
| `stopwords-hy-clean.json` | Custom Armenian stopword list (no standardized list available for Armenian). |

## Software Requirements

```r
# R packages
install.packages(c("tidyverse", "quanteda", "quanteda.textmodels",
                    "quanteda.textstats", "reticulate", "knitr",
                    "kableExtra", "jsonlite", "lubridate"))

# Python packages (via reticulate or pip)
# pip install torch transformers scikit-learn pandas numpy statsmodels scipy
```

## Reproducing the Analysis

1. Clone this repository and ensure R, Python, and the required packages are installed.
2. Open `analysis.qmd` in RStudio or any Quarto-compatible editor.
3. Run all chunks sequentially. The document will:
   - Load and preprocess the coded training corpus
   - Train Naive Bayes on bag-of-words features
   - Generate BERT embeddings using HyeBERT (Armenian BERT)
   - Train and cross-validate Logistic Regression, Random Forest, Gradient Boosting, and Stacking Ensemble
   - Compare fine-tuned vs frozen BERT embeddings
   - Classify the full corpus and generate CSO-year criticality trajectories
   - Run sensitivity analysis across classifiers
   - Estimate document-level logistic regression with organization fixed effects

**Note:** The BERT embedding step requires downloading the HyeBERT model (~440MB) on first run.

## Key Results

| Method | Representation | Accuracy | F1 (Critical) |
|--------|---------------|----------|---------------|
| Naive Bayes | Bag-of-words | 72.5% (SD 3.7) | 0.857 |
| Logistic Regression | BERT embeddings | 76.4% (SD 3.4) | 0.799 |
| Random Forest | BERT embeddings | 77.6% (SD 3.4) | 0.811 |
| Gradient Boosting | BERT embeddings | 76.7% (SD 3.0) | 0.805 |
| Stacking Ensemble | BERT embeddings | 77.6% (SD 2.5) | 0.811 |

Fine-tuned HyeBERT (59.2%) underperforms frozen embeddings (72.1%) with only 642 training examples.

## Data Sources

- Union of Informed Citizens (UIC): https://uic.am
- Helsinki Citizens' Assembly Vanadzor (HCAV): https://hcav.am
- Transparency International Armenia: https://transparency.am
- Civic Society Institute (CSI): https://csi.am
- Helsinki Association for Human Rights (HAHR): https://hahr.am
- HyeBERT model: https://huggingface.co/aking11/hyebert

## Contact

Albert Ananyan — Northwestern University

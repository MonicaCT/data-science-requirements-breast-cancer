# Validation Report - Data Science Requirements Website

Date: 2026-07-13

## Scope

Reusable portfolio website applied to the existing data science requirements project for the Breast Cancer Wisconsin Diagnostic dataset.

## Reused products

- Final report: `report/final_report.html`, `report/final_report.pdf`
- Report preview: `report/final_report_preview.png`
- Figures: `figures/01_project_flow.png`, `figures/02_dataset_comparison_matrix.png`, `figures/04_missingness_map.png`, `figures/05_target_distribution.png`, `figures/06_effect_size_lollipop.png`, `figures/07_correlation_heatmap.png`, `figures/12_executive_dashboard.png`
- Tables: `tables/functional_requirements.csv`, `tables/nonfunctional_requirements.csv`, `tables/data_dictionary.csv`, `tables/dataset_comparison.csv`, `tables/data_quality_summary.csv`, `tables/variable_effect_sizes.csv`
- Citation metadata: `CITATION.cff`
- License: `LICENSE`

## Validation checklist

| Check | Status | Notes |
|---|---|---|
| Reusable template | PASS | `docs/index.html` uses the approved shared CSS and JS template. |
| GitHub Pages structure | PASS | Repository is prepared for `main` / `/docs`. |
| HTML sections | PASS | Hero, summary, dataset snapshot, findings, quality, figures, tables, requirements, methodology, report, reproducibility, limitations, citation and portfolio navigation are present. |
| Figures | PASS | Eight existing figures/report preview assets are reused by URL; no figure was regenerated. |
| Tables | PASS | Six existing CSV tables are linked; no table was regenerated. |
| Final report | PASS | Existing HTML/PDF report is linked and was not modified. |
| README | PASS | Top buttons expose Website, Final Report, Main Figures, Executive Tables, Methodology, Repository and Back to Portfolio. |
| Local paths | PASS | No Windows local paths were introduced. |
| Sensitive strings | PASS | No credential-like strings were introduced. |
| Analytical outputs | PASS | `data/`, `figures/`, `report/`, `src/` and `tables/` were not modified. |
| Pipeline execution | PASS | No scripts, models, statistics, figures or tables were executed or regenerated. |
| GitHub Pages publication | WARNING | Public availability depends on GitHub Pages being configured to deploy from `main` and `/docs`. |

## GitHub Pages expected configuration

```text
Source: Deploy from a branch
Branch: main
Folder: /docs
```

If the public URL returns 404, the expected status is:

```text
Pending human configuration
```

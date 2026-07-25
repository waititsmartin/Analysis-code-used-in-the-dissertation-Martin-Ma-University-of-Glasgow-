# Analysis Code Repository

This repository contains the analysis code used in the dissertation
"Search Attention and Gain–Loss-Day Turnover Asymmetry in Global
Semiconductor Equities" (Martin Ma, University of Glasgow).

## Structure

- `python/` — Python scripts used for (a) independent verification of
  Stata regression output via manual two-way fixed-effects estimation
  and wild cluster bootstrap, and (b) cleaning/reshaping Stata `esttab`
  CSV output for inclusion in tables.
- `stata/` — Stata do-files used to construct the panel, run the
  baseline and robustness regressions, and conduct wild cluster
  bootstrap inference via `boottest`.

## Workflow

The empirical workflow used throughout this dissertation was:
1. Build the panel and run the primary regressions in Stata
   (`stata/*.do`).
2. Independently replicate the key coefficients in Python
   (`python/wild_bootstrap.py`) as a cross-check, since Stata and
   Python implement the two-way fixed-effects demeaning and
   cluster-robust standard errors slightly differently in edge cases
   (e.g. unbalanced panels).
3. Where `boottest` could not be used directly because `reghdfe` had
   absorbed more than one fixed effect (a known limitation of
   `boottest`), the panel was first demeaned manually
   (`stata/manual_demeaning_template.do`) and then passed to `regress`
   + `boottest`.

## Requirements

- Stata 17+ with `reghdfe`, `ftools`, and `boottest` (Roodman et al.,
  2019) installed (`ssc install reghdfe`, `ssc install ftools`,
  `ssc install boottest`).
- Python 3.10+ with `pandas` and `numpy`.

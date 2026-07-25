"""
Panel quality-control checks, run before any regression to confirm the
panel is structured as expected. This script reproduces the checks
referenced in Sections 3.1, 4.1, and 5.3 (firm/country coverage, the
China/Post_AI/COVID_Period flags, the Turnover Asymmetry outlier count,
and the China-specific COVID/AI overlap count).
"""

import numpy as np
import pandas as pd


def run_qc(panel_path: str = "master_panel_30.csv"):
    df = pd.read_csv(panel_path, parse_dates=["Week_ID"] if "Week_ID" in
                      pd.read_csv(panel_path, nrows=0).columns else None)

    print("=== Sample coverage ===")
    print("Tickers:", df["Ticker"].nunique())
    print(df.groupby("Country")["Ticker"].nunique())
    print()

    print("=== China flag cross-check (should be 1 only for CN rows) ===")
    print(pd.crosstab(df["Country"], df["China"]))
    print()

    print("=== Post_AI / COVID_Period coverage by country ===")
    print(df.groupby("Country")[["Post_AI", "COVID_Period"]].sum())
    print()

    print("=== COVID/AI overlap (should be non-zero for China only) ===")
    print(df.groupby("Country")["COVID_AI_Overlap"].sum() if "COVID_AI_Overlap" in df.columns
          else "COVID_AI_Overlap column not found — check panel version")
    print()

    print("=== China baseline period check ===")
    print("(weeks where COVID_Period=0 AND Post_AI=0 — should be non-empty for")
    print(" China too, confirming the COVID_Period start date fix; see Section 3.2)")
    cn = df[df["Country"] == "CN"]
    baseline_cn = cn[(cn["COVID_Period"] == 0) & (cn["Post_AI"] == 0)]
    print(f"China baseline weeks: {len(baseline_cn)} "
          f"({baseline_cn['Week_ID'].min() if len(baseline_cn) else 'n/a'} to "
          f"{baseline_cn['Week_ID'].max() if len(baseline_cn) else 'n/a'})")
    print()

    print("=== Turnover Asymmetry outliers (>4 SD; addressed via winsorization, M5) ===")
    x = df["Turnover_Asymmetry"].dropna()
    n_outliers = (np.abs(x - x.mean()) > 4 * x.std()).sum()
    print(f"mean={x.mean():.5f}  sd={x.std():.5f}  N beyond 4 SD: {n_outliers} "
          f"({100 * n_outliers / len(x):.2f}% of {len(x)})")
    print()

    print("=== Turnover Asymmetry descriptive stats by country (Table 4.1 preview) ===")
    print(df.groupby("Country")["Turnover_Asymmetry"].describe().round(5))


if __name__ == "__main__":
    run_qc()

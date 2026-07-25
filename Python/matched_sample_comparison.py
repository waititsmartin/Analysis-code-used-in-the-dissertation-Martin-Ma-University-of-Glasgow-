"""
Matched-sample comparison of the full-sample vs. rolling-window
Z-score specifications (Section 4.4, "Methodological Sensitivity").

M1 (rolling Z-score) and M2 (full-sample Z-score) naturally have
different sample sizes, because the rolling window's 52-week burn-in
period drops each firm's earliest observations. This script
additionally re-estimates both specifications on the SAME
(rolling-Z-score-compatible) sub-sample, to isolate how much of the
difference between M1 and M2 is attributable to the standardization
method itself, as opposed to the difference in sample composition —
a distinction raised in external review and addressed via the caveat
added to Section 4.4.
"""

import numpy as np
import pandas as pd

from wild_bootstrap import fast_within, cluster_se


def matched_sample_comparison(panel_path: str = "master_panel_30.csv"):
    df = pd.read_csv(panel_path)
    df["log_weekly_volume"] = np.log(df["weekly_volume"].replace(0, np.nan))
    df["z_rolling_x_china"] = df["svi_zscore_rolling52w"] * df["china"]
    df["z_full_x_china"] = df["svi_zscore_fullsample"] * df["china"]

    xcols_rolling = ["svi_zscore_rolling52w", "z_rolling_x_china",
                      "weekly_volatility", "log_weekly_volume", "market_return"]
    xcols_full = ["svi_zscore_fullsample", "z_full_x_china",
                  "weekly_volatility", "log_weekly_volume", "market_return"]

    # Restrict both specifications to the sub-sample where the ROLLING
    # Z-score is available (i.e. drop each firm's 52-week burn-in period
    # from both), so the two models are estimated on identical samples.
    est = df.dropna(subset=xcols_rolling + ["turnover_asymmetry"]).copy()

    def _fit(xcols, target_col):
        firm_codes, firm_uniques = pd.factorize(est["ticker"])
        week_codes, week_uniques = pd.factorize(est["week_id"])
        n_firm, n_week = len(firm_uniques), len(week_uniques)

        Xraw = est[xcols].values.astype(float)
        Yraw = est[["turnover_asymmetry"]].values.astype(float)
        Xf = fast_within(Xraw, firm_codes, week_codes, n_firm, n_week)
        Yf = fast_within(Yraw, firm_codes, week_codes, n_firm, n_week)[:, 0]

        beta, *_ = np.linalg.lstsq(Xf, Yf, rcond=None)
        resid = Yf - Xf @ beta
        se = cluster_se(Xf, resid, firm_codes, n_firm)

        idx = xcols.index(target_col)
        return beta[idx], se[idx], beta[idx] / se[idx], len(est)

    b_roll, se_roll, t_roll, n_roll = _fit(xcols_rolling, "z_rolling_x_china")
    b_full, se_full, t_full, n_full = _fit(xcols_full, "z_full_x_china")

    print("Matched-sample comparison (same N for both specifications):")
    print(f"  Rolling Z-score:     beta={b_roll:.6f}  se={se_roll:.6f}  t={t_roll:.3f}  N={n_roll}")
    print(f"  Full-sample Z-score: beta={b_full:.6f}  se={se_full:.6f}  t={t_full:.3f}  N={n_full}")
    print()
    print("Compare against the unmatched (full) samples reported in Table 4.3,")
    print("where M1 (rolling) uses N=9,182 and M2 (full-sample) uses N=10,553.")


if __name__ == "__main__":
    matched_sample_comparison()

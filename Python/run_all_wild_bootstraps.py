"""
Batch runner that reproduces every wild-cluster-bootstrap p-value
reported in Section 4.2-4.3 of the dissertation (Tables 4.3 and 4.4),
in a single pass. This is the script that actually produced the
"wild bootstrap p = ..." figures quoted throughout the Results chapter,
and was used to independently cross-check the corresponding Stata
`boottest` output.

Requires wild_bootstrap.py in the same directory (or on the path).
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from wild_bootstrap import wild_bootstrap


def main(panel_path: str = "master_panel_30.csv", B: int = 4999):
    df = pd.read_csv(panel_path)

    df["log_weekly_volume"] = np.log(df["weekly_volume"].replace(0, np.nan))
    df["z_rolling_x_china"] = df["svi_zscore_rolling52w"] * df["china"]
    df["z_full_x_china"] = df["svi_zscore_fullsample"] * df["china"]
    df["z_x_postai"] = df["svi_zscore_rolling52w"] * df["post_ai"]
    df["z_x_china_x_postai"] = df["svi_zscore_rolling52w"] * df["china"] * df["post_ai"]
    df["z_x_covid"] = df["svi_zscore_rolling52w"] * df["covid_period"]
    df["z_x_china_x_covid"] = df["svi_zscore_rolling52w"] * df["china"] * df["covid_period"]
    df["z_x_china_x_covid_x_postai"] = (
        df["svi_zscore_rolling52w"] * df["china"] * df["covid_period"] * df["post_ai"]
    )

    results = {}

    # M1: baseline, rolling Z-score
    xcols_m1 = ["svi_zscore_rolling52w", "z_rolling_x_china",
                "weekly_volatility", "log_weekly_volume", "market_return"]
    results["M1: z_rolling_x_china"] = wild_bootstrap(
        df, xcols_m1, "z_rolling_x_china",
        ycol="turnover_asymmetry", fcol="ticker", tcol="week_id", B=B
    )

    # M2: baseline, full-sample Z-score
    xcols_m2 = ["svi_zscore_fullsample", "z_full_x_china",
                "weekly_volatility", "log_weekly_volume", "market_return"]
    results["M2: z_full_x_china"] = wild_bootstrap(
        df, xcols_m2, "z_full_x_china",
        ycol="turnover_asymmetry", fcol="ticker", tcol="week_id", B=B
    )

    # M3: triple interaction, full sample (incl. COVID/AI overlap)
    xcols_m3 = ["svi_zscore_rolling52w", "z_rolling_x_china", "z_x_postai",
                "z_x_china_x_postai", "z_x_covid", "z_x_china_x_covid",
                "weekly_volatility", "log_weekly_volume", "market_return"]
    results["M3: z_x_china_x_postai (incl. overlap)"] = wild_bootstrap(
        df, xcols_m3, "z_x_china_x_postai",
        ycol="turnover_asymmetry", fcol="ticker", tcol="week_id", B=B
    )

    # M3b: same, excluding the China-specific COVID/AI overlap weeks
    df_b = df[df["covid_ai_overlap"] == 0].copy()
    results["M3b: z_x_china_x_postai (excl. overlap)"] = wild_bootstrap(
        df_b, xcols_m3, "z_x_china_x_postai",
        ycol="turnover_asymmetry", fcol="ticker", tcol="week_id", B=B
    )

    # M8: quadruple interaction (full sample, overlap modelled explicitly)
    xcols_m8 = ["svi_zscore_rolling52w", "z_x_postai", "z_x_china_x_postai",
                "z_x_covid", "z_x_china_x_covid", "z_x_china_x_covid_x_postai",
                "weekly_volatility", "log_weekly_volume", "market_return"]
    results["M8: z_x_china_x_covid_x_postai"] = wild_bootstrap(
        df, xcols_m8, "z_x_china_x_covid_x_postai",
        ycol="turnover_asymmetry", fcol="ticker", tcol="week_id", B=B
    )
    results["M8: z_x_china_x_postai"] = wild_bootstrap(
        df, xcols_m8, "z_x_china_x_postai",
        ycol="turnover_asymmetry", fcol="ticker", tcol="week_id", B=B
    )

    print(f"{'Model / Coefficient':45s}{'beta':>12s}{'SE(cl)':>12s}{'t':>8s}"
          f"{'p(normal)':>12s}{'p(wild boot)':>14s}{'N':>8s}")
    for k, (b, se, t, p_wild, n) in results.items():
        p_norm = 2 * (1 - norm.cdf(abs(t)))
        print(f"{k:45s}{b:12.6f}{se:12.6f}{t:8.3f}{p_norm:12.4f}{p_wild:14.4f}{n:8d}")


if __name__ == "__main__":
    main()

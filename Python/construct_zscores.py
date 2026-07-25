"""
Construct the two attention-standardization variables used throughout
the dissertation (Section 3.2):

  - SVI_Zscore_FullSample: standardizes each firm's weekly raw search
    index using the ENTIRE 2019-2026 sample's mean and standard
    deviation. Retained only as a methodological-sensitivity
    comparison (Section 4.4), since it uses future information to
    standardize past observations (a look-ahead bias).

  - SVI_Zscore_Rolling52w: standardizes each week using only the
    trailing 52 weeks of that firm's own history (no future
    information). This is the main specification, following the
    convention in Bijl, Kringhaug, Molnár & Sandvik (2016) and
    Dimpfl & Jank (2016).

This is the single most consequential piece of methodology in the
dissertation: the divergence between these two constructions is what
originally motivated the correction described in Section 3.2 and the
methodological-sensitivity check in Section 4.4.
"""

import numpy as np
import pandas as pd


def add_zscore_columns(df: pd.DataFrame, raw_col: str = "Raw_SVI",
                        ticker_col: str = "Ticker", date_col: str = "Week_End_Date",
                        rolling_window: int = 52, min_periods: int = 26) -> pd.DataFrame:
    """
    Adds two new columns to df: `{raw_col}_Zscore_FullSample` and
    `{raw_col}_Zscore_Rolling52w`, computed per-ticker.

    min_periods=26 (half the window) is used so that a firm's very
    earliest weeks are not entirely dropped, at the cost of somewhat
    noisier early-window estimates; the resulting missing values
    (roughly the first `rolling_window - min_periods` weeks per firm)
    are handled as missing data in the main panel construction, not
    imputed.
    """
    df = df.sort_values([ticker_col, date_col]).copy()

    def _full_sample_z(g):
        return (g[raw_col] - g[raw_col].mean()) / g[raw_col].std()

    def _rolling_z(g):
        r = g[raw_col].rolling(window=rolling_window, min_periods=min_periods)
        return (g[raw_col] - r.mean()) / r.std()

    df["SVI_Zscore_FullSample"] = df.groupby(ticker_col, group_keys=False).apply(_full_sample_z)
    df["SVI_Zscore_Rolling52w"] = df.groupby(ticker_col, group_keys=False).apply(_rolling_z)

    return df


def diagnose_lookahead_bias(df: pd.DataFrame, ticker_col: str = "Ticker",
                             date_col: str = "Week_End_Date") -> pd.DataFrame:
    """
    Quick diagnostic used when first identifying the look-ahead bias:
    for each firm, reports the correlation between calendar time and
    the full-sample Z-score. A strong positive correlation indicates
    the attention series has a structural upward trend that full-sample
    standardization will relabel as a sequence of "abnormal" shocks
    concentrated in the post-2022 period.
    """
    out = []
    for ticker, g in df.groupby(ticker_col):
        g = g.sort_values(date_col)
        if g["SVI_Zscore_FullSample"].notna().sum() < 10:
            continue
        wk_num = pd.to_datetime(g[date_col]).rank()
        corr = np.corrcoef(wk_num, g["SVI_Zscore_FullSample"].fillna(0))[0, 1]
        out.append({"Ticker": ticker, "corr_time_vs_zscore": round(corr, 3)})
    return pd.DataFrame(out).sort_values("corr_time_vs_zscore", ascending=False)


if __name__ == "__main__":
    raw = pd.read_csv("all_search_attention_raw_30.csv", parse_dates=["Week_End_Date"])
    raw = add_zscore_columns(raw)

    print("Look-ahead bias diagnostic (correlation between calendar time and full-sample Z-score):")
    print(diagnose_lookahead_bias(raw).to_string(index=False))

    raw.to_csv("all_search_attention_with_zscores.csv", index=False)

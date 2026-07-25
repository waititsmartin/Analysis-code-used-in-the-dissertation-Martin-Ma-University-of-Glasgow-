"""
Independent Python cross-check of Stata reghdfe / boottest results.

This script implements, from first principles (no linearmodels dependency),
a two-way fixed-effects panel regression with firm-clustered standard errors,
and a restricted wild cluster bootstrap (Cameron, Gelbach & Miller, 2008)
for a chosen coefficient of interest. It was used throughout the dissertation
to independently verify Stata's reghdfe/boottest output, since the two tools
occasionally diverge on small-sample degrees-of-freedom adjustments.

Usage pattern:
    import pandas as pd
    from wild_bootstrap import within_transform, cluster_se, wild_bootstrap

    df = pd.read_csv("master_panel_30.csv")
    xcols = ["svi_zscore_rolling52w", "z_rolling_x_china", "weekly_volatility",
             "log_weekly_volume", "market_return"]
    beta, se, t_obs, p_wild, n = wild_bootstrap(
        df, xcols, target_col="z_rolling_x_china",
        ycol="turnover_asymmetry", fcol="ticker", tcol="week_id", B=999
    )
"""

import numpy as np
import pandas as pd


def fast_within(mat: np.ndarray, firm_idx: np.ndarray, week_idx: np.ndarray,
                n_firm: int, n_week: int, tol: float = 1e-10, maxit: int = 100) -> np.ndarray:
    """
    Iteratively demean an (N, K) matrix on two categorical dimensions
    (firm and week), equivalent to the within-transformation implied by
    two-way fixed effects. Converges via alternating projection, which is
    exact for balanced panels and a very close numerical approximation
    for the mildly unbalanced panels used here.
    """
    mat = mat.copy()
    for _ in range(maxit):
        before = mat.copy()

        sums = np.zeros((n_firm, mat.shape[1]))
        cnts = np.bincount(firm_idx, minlength=n_firm).astype(float)
        for k in range(mat.shape[1]):
            sums[:, k] = np.bincount(firm_idx, weights=mat[:, k], minlength=n_firm)
        fmeans = sums / cnts[:, None]
        mat = mat - fmeans[firm_idx]

        sums = np.zeros((n_week, mat.shape[1]))
        cnts2 = np.bincount(week_idx, minlength=n_week).astype(float)
        for k in range(mat.shape[1]):
            sums[:, k] = np.bincount(week_idx, weights=mat[:, k], minlength=n_week)
        wmeans = sums / cnts2[:, None]
        mat = mat - wmeans[week_idx]

        if np.abs(mat - before).max() < tol:
            break
    return mat


def cluster_se(X: np.ndarray, resid: np.ndarray, firm_idx: np.ndarray, n_firm: int) -> np.ndarray:
    """Firm-clustered standard errors (Petersen, 2009), following the
    standard Stata vce(cluster) small-sample adjustment G/(G-1) * (N-1)/(N-K)."""
    XtX_inv = np.linalg.inv(X.T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for f in range(n_firm):
        idx = firm_idx == f
        if idx.sum() == 0:
            continue
        u = X[idx].T @ resid[idx]
        meat += np.outer(u, u)
    G, N, K = n_firm, len(resid), X.shape[1]
    c = G / (G - 1) * (N - 1) / (N - K)
    V = c * XtX_inv @ meat @ XtX_inv
    return np.sqrt(np.diag(V))


def wild_bootstrap(df: pd.DataFrame, xcols: list, target_col: str,
                    ycol: str = "turnover_asymmetry", fcol: str = "ticker",
                    tcol: str = "week_id", B: int = 999, seed: int = 42):
    """
    Restricted wild cluster bootstrap (Cameron, Gelbach & Miller, 2008)
    for the null H0: beta[target_col] = 0, using Rademacher weights.
    Equivalent in spirit to Stata's `boottest target_col, cluster(firm) reps(B)`
    after the panel has been demeaned on two fixed effects (since boottest
    itself only supports a single absorbed fixed effect).

    Returns: (beta, cluster_se, t_obs, p_wild, N)
    """
    np.random.seed(seed)
    est = df.dropna(subset=xcols + [ycol]).copy()
    firm_codes, firm_uniques = pd.factorize(est[fcol])
    week_codes, week_uniques = pd.factorize(est[tcol])
    n_firm, n_week = len(firm_uniques), len(week_uniques)

    target_idx = xcols.index(target_col)
    xcols_restricted = [c for c in xcols if c != target_col]

    Xraw = est[xcols].values.astype(float)
    Yraw = est[[ycol]].values.astype(float)
    Xf = fast_within(Xraw, firm_codes, week_codes, n_firm, n_week)
    Yf = fast_within(Yraw, firm_codes, week_codes, n_firm, n_week)[:, 0]

    beta, *_ = np.linalg.lstsq(Xf, Yf, rcond=None)
    resid = Yf - Xf @ beta
    se = cluster_se(Xf, resid, firm_codes, n_firm)
    t_obs = beta[target_idx] / se[target_idx]

    Xr_raw = est[xcols_restricted].values.astype(float)
    Xr = fast_within(Xr_raw, firm_codes, week_codes, n_firm, n_week)
    beta_r, *_ = np.linalg.lstsq(Xr, Yf, rcond=None)
    resid_r = Yf - Xr @ beta_r
    fitted_r = Yf - resid_r

    boot_t = np.zeros(B)
    for b in range(B):
        w = np.random.choice([-1.0, 1.0], size=n_firm)
        wvec = w[firm_codes]
        y_star = fitted_r + resid_r * wvec
        beta_b, *_ = np.linalg.lstsq(Xf, y_star, rcond=None)
        resid_b = y_star - Xf @ beta_b
        se_b = cluster_se(Xf, resid_b, firm_codes, n_firm)
        boot_t[b] = beta_b[target_idx] / se_b[target_idx]

    p_wild = (np.abs(boot_t) >= np.abs(t_obs)).mean()
    return beta[target_idx], se[target_idx], t_obs, p_wild, len(est)


if __name__ == "__main__":
    # Example usage (adjust paths/columns to match your actual panel)
    df = pd.read_csv("master_panel_30.csv")

    df["z_rolling_x_china"] = df["svi_zscore_rolling52w"] * df["china"]
    df["log_weekly_volume"] = np.log(df["weekly_volume"])

    xcols = ["svi_zscore_rolling52w", "z_rolling_x_china",
              "weekly_volatility", "log_weekly_volume", "market_return"]

    beta, se, t_obs, p_wild, n = wild_bootstrap(
        df, xcols, target_col="z_rolling_x_china",
        ycol="turnover_asymmetry", fcol="ticker", tcol="week_id", B=999
    )
    print(f"beta={beta:.6f}  se={se:.6f}  t={t_obs:.3f}  wild-bootstrap p={p_wild:.4f}  N={n}")

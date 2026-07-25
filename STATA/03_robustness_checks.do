* ============================================================
* Robustness checks: M4 (CGO), M5 (winsorized), M6 (lagged
* attention), M7 (market-adjusted returns)
* Input: master_panel_30.csv
* ============================================================

clear all
import delimited "master_panel_30.csv", clear varnames(1)

gen log_weekly_volume = ln(weekly_volume)
gen z_rolling_x_china = svi_zscore_rolling52w * china

encode ticker, gen(firm_id)
encode week_id, gen(week_num)

* ============================================================
* M4: Alternative disposition-effect proxy — lagged capital
* gains overhang (CGO). Requires a trailing reference-price
* window, hence the smaller N (5,344 vs 9,182).
* ============================================================
gen z_x_cgolag1 = svi_zscore_rolling52w * cgo_lag1
gen z_x_china_x_cgolag1 = svi_zscore_rolling52w * china * cgo_lag1

reghdfe turnover_asymmetry svi_zscore_rolling52w z_rolling_x_china cgo_lag1 z_x_cgolag1 z_x_china_x_cgolag1 weekly_volatility log_weekly_volume market_return, absorb(firm_id week_num) vce(cluster firm_id)
eststo M4_cgo

* ============================================================
* M5: Winsorized Turnover Asymmetry (1st/99th percentiles)
* ============================================================
quietly summarize turnover_asymmetry, detail
gen turnover_asymmetry_wins = turnover_asymmetry
replace turnover_asymmetry_wins = r(p1) if turnover_asymmetry < r(p1)
replace turnover_asymmetry_wins = r(p99) if turnover_asymmetry > r(p99)

reghdfe turnover_asymmetry_wins svi_zscore_rolling52w z_rolling_x_china weekly_volatility log_weekly_volume market_return, absorb(firm_id week_num) vce(cluster firm_id)
eststo M5_winsorized

* ============================================================
* M6: Lagged attention (one-week lag), addressing reverse
* causality
* ============================================================
gen z_rolling_lag1_x_china = svi_zscore_rolling52w_lag1 * china

reghdfe turnover_asymmetry svi_zscore_rolling52w_lag1 z_rolling_lag1_x_china weekly_volatility log_weekly_volume market_return, absorb(firm_id week_num) vce(cluster firm_id)
eststo M6_lagged

* ============================================================
* M7: Market-adjusted abnormal return in place of the raw
* market return control
* ============================================================
gen abnormal_return = weekly_log_return - market_return

reghdfe turnover_asymmetry svi_zscore_rolling52w z_rolling_x_china weekly_volatility log_weekly_volume abnormal_return, absorb(firm_id week_num) vce(cluster firm_id)
eststo M7_market_adjusted

* ============================================================
* Export
* ============================================================
esttab M4_cgo M5_winsorized M6_lagged M7_market_adjusted using "regression_robustness.csv", replace se star(* 0.10 ** 0.05 *** 0.01) stats(N N_clust r2_within) csv

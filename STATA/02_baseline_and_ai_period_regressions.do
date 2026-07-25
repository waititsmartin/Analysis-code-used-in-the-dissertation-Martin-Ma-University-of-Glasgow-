* ============================================================
* Baseline regressions (M1, M2) and AI-period / COVID-confound
* regressions (M3, M3b, M8)
* Input: master_panel_30.csv
* ============================================================

clear all
import delimited "master_panel_30.csv", clear varnames(1)

gen log_weekly_volume = ln(weekly_volume)
gen z_rolling_x_china = svi_zscore_rolling52w * china
gen z_full_x_china    = svi_zscore_fullsample * china
gen z_x_postai              = svi_zscore_rolling52w * post_ai
gen z_x_china_x_postai       = svi_zscore_rolling52w * china * post_ai
gen z_x_covid                = svi_zscore_rolling52w * covid_period
gen z_x_china_x_covid        = svi_zscore_rolling52w * china * covid_period
gen z_x_china_x_covid_x_postai = svi_zscore_rolling52w * china * covid_period * post_ai

encode ticker, gen(firm_id)
encode week_id, gen(week_num)

tab covid_ai_overlap country

* ============================================================
* M1: Baseline, rolling-window Z-score (main specification)
* ============================================================
reghdfe turnover_asymmetry svi_zscore_rolling52w z_rolling_x_china weekly_volatility log_weekly_volume market_return, absorb(firm_id week_num) vce(cluster firm_id)
eststo M1_main

* ============================================================
* M2: Baseline, full-sample Z-score (methodological comparison)
* ============================================================
reghdfe turnover_asymmetry svi_zscore_fullsample z_full_x_china weekly_volatility log_weekly_volume market_return, absorb(firm_id week_num) vce(cluster firm_id)
eststo M2_fullsample

* ============================================================
* M3: Triple/COVID interaction, full sample (includes overlap weeks)
* ============================================================
reghdfe turnover_asymmetry svi_zscore_rolling52w z_rolling_x_china z_x_postai z_x_china_x_postai z_x_covid z_x_china_x_covid weekly_volatility log_weekly_volume market_return, absorb(firm_id week_num) vce(cluster firm_id)
eststo M3_covid_ai
boottest z_x_china_x_postai, cluster(firm_id) reps(999) nograph

* ============================================================
* M3b: Same as M3, excluding the 50 China-specific COVID/AI
* overlap firm-weeks
* ============================================================
preserve
keep if covid_ai_overlap == 0
reghdfe turnover_asymmetry svi_zscore_rolling52w z_rolling_x_china z_x_postai z_x_china_x_postai z_x_covid z_x_china_x_covid weekly_volatility log_weekly_volume market_return, absorb(firm_id week_num) vce(cluster firm_id)
eststo M3b_no_overlap
boottest z_x_china_x_postai, cluster(firm_id) reps(999) nograph
restore

* ============================================================
* M8: Quadruple interaction, full sample (overlap modelled
* explicitly instead of excluded). Drops z_rolling_x_china due
* to collinearity with the quadruple term.
*
* NOTE: boottest does not support models where reghdfe has
* absorbed more than one fixed effect. See
* 03_manual_demeaning_for_boottest.do for the workaround used
* to bootstrap M8's coefficients.
* ============================================================
reghdfe turnover_asymmetry svi_zscore_rolling52w z_x_postai z_x_china_x_postai z_x_covid z_x_china_x_covid z_x_china_x_covid_x_postai weekly_volatility log_weekly_volume market_return, absorb(firm_id week_num) vce(cluster firm_id)
eststo M8_covid_amplify_ai

* ============================================================
* Export
* ============================================================
esttab M1_main M2_fullsample M3_covid_ai M3b_no_overlap M8_covid_amplify_ai using "regression_summary_full.csv", replace se star(* 0.10 ** 0.05 *** 0.01) stats(N N_clust r2_within) csv

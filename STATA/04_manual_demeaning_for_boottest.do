* ============================================================
* Template: manual two-way demeaning + boottest
*
* boottest (Roodman et al., 2019) does not support models where
* reghdfe has absorbed more than one fixed effect. This template
* works around that limitation by:
*   1. Running reghdfe once as a benchmark (to sanity-check the
*      manual demeaning against).
*   2. Manually, iteratively demeaning the dependent variable and
*      every regressor on both firm and week, using only base Stata
*      commands (bysort/egen), so no additional dependency is
*      required beyond reghdfe itself.
*   3. Running a plain `regress` on the demeaned variables (which
*      boottest fully supports).
*   4. Running boottest on the demeaned-variable regression.
*
* This exact template was used for (a) the M8 quadruple-interaction
* model in 02_baseline_and_ai_period_regressions.do, where boottest
* needed to be run on the quadruple interaction term, and (b) the
* within-country price-limit-hit test in
* 05_price_limit_within_country_test.do.
*
* Adjust the `local varlist` and the two reghdfe/regress calls to
* match the specification you are testing.
* ============================================================

* ---- Example: M8's quadruple interaction term ----

* Step 1: benchmark reghdfe (for comparison only)
reghdfe turnover_asymmetry svi_zscore_rolling52w z_x_postai z_x_china_x_postai z_x_covid z_x_china_x_covid z_x_china_x_covid_x_postai weekly_volatility log_weekly_volume market_return, absorb(firm_id week_num) vce(cluster firm_id)
eststo M8_benchmark

* Step 2: manual iterative two-way demeaning
local varlist turnover_asymmetry svi_zscore_rolling52w z_x_postai z_x_china_x_postai z_x_covid z_x_china_x_covid z_x_china_x_covid_x_postai weekly_volatility log_weekly_volume market_return

foreach v of local varlist {
    gen r_`v' = `v'
}

foreach v of local varlist {
    quietly drop if missing(`v')
}

forvalues iter = 1/30 {
    foreach v of local varlist {
        quietly bysort firm_id: egen double tmp_m = mean(r_`v')
        quietly replace r_`v' = r_`v' - tmp_m
        quietly drop tmp_m

        quietly bysort week_num: egen double tmp_m = mean(r_`v')
        quietly replace r_`v' = r_`v' - tmp_m
        quietly drop tmp_m
    }
}

* Step 3: plain regress on demeaned variables
* (coefficients here should match Step 1's benchmark almost exactly;
*  standard errors may differ very slightly since reghdfe applies a
*  small-sample correction for absorbed-FE degrees of freedom that
*  plain regress does not)
regress r_turnover_asymmetry r_svi_zscore_rolling52w r_z_x_postai r_z_x_china_x_postai r_z_x_covid r_z_x_china_x_covid r_z_x_china_x_covid_x_postai r_weekly_volatility r_log_weekly_volume r_market_return, vce(cluster firm_id)

* Step 4: wild cluster bootstrap on the coefficient of interest
boottest r_z_x_china_x_covid_x_postai, cluster(firm_id) reps(999) nograph
boottest r_z_x_china_x_postai, cluster(firm_id) reps(999) nograph

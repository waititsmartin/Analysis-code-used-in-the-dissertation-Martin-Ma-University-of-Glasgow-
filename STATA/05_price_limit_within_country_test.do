* ============================================================
* Within-country identification: price-limit-hit weeks
* (Section 5.3)
*
* Tests whether the China-specific attention interaction is
* driven specifically by the daily price-limit mechanism, by
* comparing attention effects in Chinese firm-weeks where the
* limit was triggered against those where it was not.
*
* Step 1 constructs the weekly price-limit-hit flag from daily
* price data. Step 2 merges this into the main panel (restricted
* to the Chinese sub-sample) and estimates the model, using the
* manual-demeaning + boottest workaround from
* 04_manual_demeaning_for_boottest.do since this specification
* also uses two absorbed fixed effects.
* ============================================================

* ============================================================
* STEP 1: Construct the weekly price-limit-hit flag
* Input: CN_Stock_Daily_10.csv (daily OHLCV for the 10 China firms)
* ============================================================
clear all
import delimited "CN_Stock_Daily_10.csv", clear varnames(1)

gen date_stata = date(date, "YMD")
format date_stata %td
sort ticker date_stata

by ticker (date_stata), sort: gen daily_return_pct = (close - close[_n-1]) / close[_n-1] * 100 if _n > 1

* ChiNext/STAR Market tickers moved to a +/-20% limit from 2020-08-24;
* all other tickers remain at +/-10% throughout.
gen threshold = 10
replace threshold = 20 if strpos(ticker, "300661") > 0 & date_stata >= date("2020-08-24", "YMD")
replace threshold = 20 if strpos(ticker, "300223") > 0 & date_stata >= date("2020-08-24", "YMD")

* Small tolerance band to avoid rounding-related misses at the boundary
gen hit_limit_day = (abs(daily_return_pct) >= (threshold - 0.3)) if !missing(daily_return_pct)

gen iso_year = yofd(date_stata)
gen iso_week = week(date_stata)
gen week_id_iso = string(iso_year) + "-W" + string(iso_week, "%02.0f")

collapse (max) price_limit_hit = hit_limit_day, by(ticker week_id_iso)

rename week_id_iso week_id
save "cn_price_limit_weekly.dta", replace

tab price_limit_hit

* ============================================================
* STEP 2: Merge into the main panel and estimate the model
* ============================================================
clear all
import delimited "master_panel_30.csv", clear varnames(1)

keep if country == "CN"

merge 1:1 ticker week_id using "cn_price_limit_weekly.dta"
tab _merge
keep if _merge == 3
drop _merge

replace price_limit_hit = 0 if missing(price_limit_hit)

gen log_weekly_volume = ln(weekly_volume)
gen z_x_pricelimit = svi_zscore_rolling52w * price_limit_hit

encode ticker, gen(firm_id)
encode week_id, gen(week_num)

* Benchmark reghdfe
reghdfe turnover_asymmetry svi_zscore_rolling52w z_x_pricelimit weekly_volatility log_weekly_volume market_return, absorb(firm_id week_num) vce(cluster firm_id)
eststo M9_pricelimit_china

* Manual demeaning (see 04_manual_demeaning_for_boottest.do for details
* on why this step is necessary)
local varlist turnover_asymmetry svi_zscore_rolling52w z_x_pricelimit weekly_volatility log_weekly_volume market_return

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

regress r_turnover_asymmetry r_svi_zscore_rolling52w r_z_x_pricelimit r_weekly_volatility r_log_weekly_volume r_market_return, vce(cluster firm_id)

* Note: China subsample has only 10 clusters (fewer than the full
* 30-firm sample), making wild cluster bootstrap especially important
boottest r_z_x_pricelimit, cluster(firm_id) reps(999) nograph

esttab M9_pricelimit_china using "table_pricelimit_china.csv", replace se star(* 0.10 ** 0.05 *** 0.01) stats(N N_clust r2_within) csv

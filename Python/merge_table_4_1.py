"""
Merge the four Stata-exported descriptive-statistics panels
(table_4_1_panelA_fullsample.csv, table_4_1_panel_US.csv,
table_4_1_panel_JP.csv, table_4_1_panel_CN.csv — produced by
stata/01_descriptive_stats.do) into a single combined table with a
Panel-label row inserted before each block, ready to be cleaned by
clean_esttab_output.parse_descriptive_stats_panel() and pasted into
Table 4.1 of the dissertation.
"""

import pandas as pd

FILES = {
    "Panel A: Full Sample": "table_4_1_panelA_fullsample.csv",
    "Panel B: United States": "table_4_1_panel_US.csv",
    "Panel C: Japan": "table_4_1_panel_JP.csv",
    "Panel D: China": "table_4_1_panel_CN.csv",
}

OUTPUT_PATH = "table_4_1_combined.csv"


def load_panel(path):
    df = pd.read_csv(path)
    first_col = df.columns[0]
    return df.rename(columns={first_col: "Variable"})


def main():
    combined_rows = []
    for panel_label, path in FILES.items():
        try:
            df = load_panel(path)
        except FileNotFoundError:
            print(f"[warning] file not found, skipped: {path}")
            continue

        panel_header = {col: "" for col in df.columns}
        panel_header["Variable"] = panel_label
        combined_rows.append(pd.DataFrame([panel_header]))
        combined_rows.append(df)

    if not combined_rows:
        print("No files were successfully read; check paths.")
        return

    combined = pd.concat(combined_rows, ignore_index=True)
    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"Written: {OUTPUT_PATH}")
    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()

"""
Script 10: Multivariate Cointegration & VAR Analysis

Objective:
- Reinforce course content by conducting cointegration tests and VAR modeling
  between Hang Seng Index (HSI) and key external drivers (HSCEI, USDHKD, S&P500).
- Produce interpretable diagnostics (Johansen test, Granger causality, impulse
  responses, FEVD) for inclusion in the main report.

Outputs:
- output/var_adf_results.csv
- output/var_johansen_trace.csv
- output/var_granger_results.csv
- output/var_summary.txt
- output/figures/26_var_irf.png
- output/figures/27_var_fevd.png

"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen

DATA_PATH = Path("data/hsi_features.csv")
OUTPUT_DIR = Path("output")
FIG_DIR = OUTPUT_DIR / "figures"


def ensure_dirs():
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)


def load_series():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run scripts/07_feature_engineering.py first."
        )
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"]).sort_values("Date")
    series = df[
        [
            "Date",
            "Close",
            "HSCEI_close",
            "USDHKD_close",
            "SP500_close",
        ]
    ].dropna()
    series = series.set_index("Date")
    series.columns = ["HSI", "HSCEI", "USDHKD", "SP500"]
    return series


def adf_test(series: pd.DataFrame):
    rows = []
    for col in series.columns:
        result = adfuller(series[col])
        rows.append(
            {
                "Series": col,
                "ADF": result[0],
                "p-value": result[1],
                "Lags": result[2],
                "Obs": result[3],
            }
        )
        diff_result = adfuller(series[col].diff().dropna())
        rows.append(
            {
                "Series": f"d({col})",
                "ADF": diff_result[0],
                "p-value": diff_result[1],
                "Lags": diff_result[2],
                "Obs": diff_result[3],
            }
        )
    adf_df = pd.DataFrame(rows)
    adf_df.to_csv(OUTPUT_DIR / "var_adf_results.csv", index=False)
    return adf_df


def johansen_test(series: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1):
    jres = coint_johansen(series, det_order, k_ar_diff)
    trace_df = pd.DataFrame(
        {
            "Rank": range(len(jres.lr1)),
            "Trace Statistic": jres.lr1,
            "5% Critical": jres.cvt[:, 1],
            "1% Critical": jres.cvt[:, 0],
        }
    )
    trace_df.to_csv(OUTPUT_DIR / "var_johansen_trace.csv", index=False)
    return jres, trace_df


def fit_var(series: pd.DataFrame):
    diff_data = series.diff().dropna()
    model = VAR(diff_data)
    lag_results = model.select_order(maxlags=10)
    selected_lag = lag_results.selected_orders["aic"]
    if selected_lag is None:
        selected_lag = 2
    var_result = model.fit(selected_lag)

    with open(OUTPUT_DIR / "var_summary.txt", "w", encoding="utf-8") as f:
        f.write(str(var_result.summary()))
        f.write("\n\nLag Order Selection (AIC): {}\n".format(selected_lag))
        f.write(str(lag_results.summary()))

    return var_result


def granger_tests(var_result, series_cols):
    rows = []
    for caused in series_cols:
        for causing in series_cols:
            if caused == causing:
                continue
            test_res = var_result.test_causality(caused, causing, kind="f")
            rows.append(
                {
                    "Null": f"{causing} does NOT Granger-cause {caused}",
                    "F-stat": getattr(test_res, "test_statistic", np.nan),
                    "p-value": test_res.pvalue,
                }
            )
    granger_df = pd.DataFrame(rows)
    granger_df.to_csv(OUTPUT_DIR / "var_granger_results.csv", index=False)
    return granger_df


def plot_irf(var_result):
    irf = var_result.irf(20)
    fig = irf.plot(orth=False)
    fig.suptitle("VAR Impulse Responses (20 steps)", fontsize=14, fontweight="bold")
    fig.savefig(FIG_DIR / "26_var_irf.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fevd(var_result):
    fevd = var_result.fevd(20)
    fig = fevd.plot()
    fig.suptitle("VAR Forecast Error Variance Decomposition", fontsize=14, fontweight="bold")
    fig.savefig(FIG_DIR / "27_var_fevd.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ensure_dirs()
    series = load_series()

    print("Running ADF tests...")
    adf_df = adf_test(series)
    print(adf_df.head())

    print("\nJohansen cointegration test...")
    jres, trace_df = johansen_test(series)
    print(trace_df)

    print("\nFitting VAR on differenced data...")
    var_result = fit_var(series)

    print("\nGranger causality tests...")
    granger_df = granger_tests(var_result, list(series.columns))
    print(granger_df.head())

    print("\nGenerating impulse response and FEVD plots...")
    plot_irf(var_result)
    plot_fevd(var_result)

    print("\n[OK] VAR analysis completed. Outputs saved under output/")


if __name__ == "__main__":
    main()



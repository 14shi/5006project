"""
Script 13: Technical Indicator Predictability Tests

Goal:
- Evaluate whether常见技术指标与外部因子在课程框架下对下一期收益具有统计显著性。
- 使用 OLS / Ljung-Box / Granger 等检验方法，量化指标的解释力。

Outputs:
- output/indicator_predictability.csv
- output/figures/31_indicator_tstats.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import grangercausalitytests

DATA_PATH = Path("data/hsi_features.csv")
OUTPUT_DIR = Path("output")
FIG_DIR = OUTPUT_DIR / "figures"


INDICATORS: Dict[str, str] = {
    "SMA_ratio_20": "价格相对20日均线偏离",
    "RSI_14": "相对强弱指标",
    "MACD_hist": "MACD 柱状图",
    "Bollinger_percent": "布林通道百分位",
    "SP500_return": "S&P500 收益率",
    "VIX_return": "VIX 变动",
}


def ensure_dirs():
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)


def load_features() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError("data/hsi_features.csv 不存在，请先运行 scripts/07_feature_engineering.py")
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"]).sort_values("Date")
    required = ["Target_Return_1d"] + list(INDICATORS.keys())
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"缺少列: {missing}")
    df = df[["Date", "Target_Return_1d", "Returns"] + list(INDICATORS.keys())].dropna()
    return df


def compute_ljungbox(series: pd.Series, lags: int = 10) -> float:
    """Return Ljung-Box p-value for indicator autocorrelation (是否可视为随机游走)"""
    lb = acorr_ljungbox(series, lags=[lags], return_df=True)
    return float(lb["lb_pvalue"].iloc[0])


def granger_pvalue(df: pd.DataFrame, indicator: str, maxlag: int = 2) -> float:
    """Granger causality test of indicator -> returns."""
    subset = df[[indicator, "Target_Return_1d"]].dropna()
    test_df = subset.rename(columns={"Target_Return_1d": "target"})
    try:
        res = grangercausalitytests(test_df[["target", indicator]], maxlag=maxlag, verbose=False)
        # take smallest p-value across lags
        pvals = [res[lag][0]["ssr_ftest"][1] for lag in range(1, maxlag + 1)]
        return float(min(pvals))
    except ValueError:
        return np.nan


def evaluate_indicator(df: pd.DataFrame, indicator: str) -> Dict[str, float]:
    y = df["Target_Return_1d"].values
    x = df[indicator].values
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    signal = np.where(df[indicator] > df[indicator].median(), 1, 0)
    direction = (df["Target_Return_1d"] > 0).astype(int).values
    accuracy = (signal == direction).mean()

    return {
        "Indicator": indicator,
        "Description": INDICATORS[indicator],
        "Coef": model.params[1],
        "tStat": model.tvalues[1],
        "PValue": model.pvalues[1],
        "R2": model.rsquared,
        "DirectionAcc": accuracy,
        "SampleSize": len(df),
        "LjungBox_p": compute_ljungbox(df[indicator]),
        "Granger_p": granger_pvalue(df, indicator),
    }


def plot_tstats(result_df: pd.DataFrame):
    ordered = result_df.sort_values("tStat", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(ordered["Indicator"], ordered["tStat"], color="#2a6f97")
    ax.invert_yaxis()
    ax.set_xlabel("t-Statistic")
    ax.set_title("技术指标对下一期收益的t检验")
    for idx, tval in enumerate(ordered["tStat"]):
        ax.text(tval, idx, f"{tval:.2f}", ha="left", va="center")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "31_indicator_tstats.png", dpi=300)
    plt.close(fig)


def main():
    ensure_dirs()
    df = load_features()
    results = [evaluate_indicator(df, ind) for ind in INDICATORS.keys()]
    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_DIR / "indicator_predictability.csv", index=False)
    print(result_df[["Indicator", "tStat", "PValue", "DirectionAcc", "Granger_p"]])
    plot_tstats(result_df)
    print("\n[OK] Indicator tests completed.")


if __name__ == "__main__":
    main()


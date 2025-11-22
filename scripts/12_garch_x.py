"""
Script 12: GARCH-X with External Drivers

Goal:
- Extend the volatility analysis by adding exogenous drivers (technical indicators
  and global factors) to the GARCH mean equation, as discussed in class (GARCH-X).
- Quantify whether VIX / S&P500 变动及技术指标对恒指收益存在显著影响。

Outputs:
- output/garch_x_results.csv
- output/garch_x_summary.txt
- output/figures/30_garchx_volatility.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("data/hsi_features.csv")
OUTPUT_DIR = Path("output")
FIG_DIR = OUTPUT_DIR / "figures"


def ensure_dirs():
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError("data/hsi_features.csv 不存在，请先运行 scripts/07_feature_engineering.py")

    df = pd.read_csv(DATA_PATH, parse_dates=["Date"]).sort_values("Date")
    required_cols = [
        "Returns",
        "SMA_ratio_20",
        "RSI_14",
        "SP500_return",
        "VIX_return",
        "USDHKD_return",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}")

    df = df[["Date"] + required_cols].dropna().copy()
    df["Returns_pct"] = df["Returns"] * 100  # scale for stability

    return df


def prepare_exog(df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    scaler = StandardScaler()
    exog = scaler.fit_transform(df[feature_cols].values)
    return exog


def fit_garch(returns: np.ndarray, mean: str = "Zero", x: np.ndarray | None = None):
    model = arch_model(
        returns,
        mean=mean,
        vol="GARCH",
        p=1,
        o=0,
        q=1,
        dist="t",
        x=x,
        lags=1 if mean == "ARX" else None,
    )
    result = model.fit(disp="off")
    return result


def persistence(params: Dict[str, float]) -> float:
    alpha = params.get("alpha[1]", 0.0)
    beta = params.get("beta[1]", 0.0)
    return alpha + beta


def export_results(base_res, garchx_res, feature_names: List[str]):
    rows = []
    for name, res in [("GARCH(1,1)", base_res), ("GARCH-X", garchx_res)]:
        rows.append(
            {
                "Model": name,
                "LogLik": res.loglikelihood,
                "AIC": res.aic,
                "BIC": res.bic,
                "Persistence": persistence(res.params),
            }
        )

    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(OUTPUT_DIR / "garch_x_results.csv", index=False)

    summary_obj = garchx_res.summary()
    coef_table = summary_obj.tables[1]
    coef_df = pd.DataFrame(coef_table.data[1:], columns=coef_table.data[0])
    coef_df = coef_df.rename(
        columns={
            "": "Parameter",
            "Unnamed: 0": "Parameter",
            "coef": "Coefficient",
            "std err": "StdErr",
            "t": "t-Stat",
            "P>|t|": "P-Value",
        }
    )
    coef_df.to_csv(OUTPUT_DIR / "garch_x_coefficients.csv", index=False)

    with open(OUTPUT_DIR / "garch_x_summary.txt", "w", encoding="utf-8") as f:
        f.write(str(garchx_res.summary()))
        f.write("\n\nExogenous features:\n")
        for idx, feat in enumerate(feature_names):
            f.write(f"- x{idx + 1}: {feat}\n")

    return comp_df, coef_df


def plot_volatility(result, dates: pd.Series, title: str, filename: str):
    cond_vol = result.conditional_volatility / 100
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates.iloc[-len(cond_vol):], cond_vol, label="Conditional Volatility")
    ax.set_title(title)
    ax.set_ylabel("Volatility")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=300)
    plt.close(fig)


def main():
    ensure_dirs()
    df = load_dataset()
    feature_cols = ["SMA_ratio_20", "RSI_14", "SP500_return", "VIX_return", "USDHKD_return"]
    exog = prepare_exog(df, feature_cols)
    returns = df["Returns_pct"].values

    print("Fitting baseline GARCH(1,1)...")
    base_res = fit_garch(returns)

    print("Fitting GARCH-X (ARX mean + exogenous drivers)...")
    garchx_res = fit_garch(returns, mean="ARX", x=exog)

    comparison_df, coef_df = export_results(base_res, garchx_res, feature_cols)
    print("\nModel comparison:")
    print(comparison_df)
    print("\nKey exogenous coefficients:")
    print(coef_df.head(len(feature_cols) + 2))  # include mean/lags

    plot_volatility(garchx_res, df["Date"], "GARCH-X Conditional Volatility", "30_garchx_volatility.png")
    print("\n[OK] Saved comparison, coefficients, and volatility plot.")


if __name__ == "__main__":
    main()


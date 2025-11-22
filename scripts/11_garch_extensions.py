"""
Script 11: Advanced GARCH Extensions (EGARCH & GJR)

Goal:
- Extend the volatility modeling section by comparing standard GARCH with
  asymmetry-aware specifications (EGARCH, GJR-GARCH) as discussed in class.
- Provide diagnostics and persistence metrics for inclusion in the final report.

Outputs:
- output/garch_extension_comparison.csv
- output/figures/28_egarch_volatility.png
- output/figures/29_egarch_residuals.png

"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model

DATA_PATH = Path("data/hsi_with_returns.csv")
OUTPUT_DIR = Path("output")
FIG_DIR = OUTPUT_DIR / "figures"


def ensure_dirs():
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)


def load_returns():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run scripts/01_eda.py first."
        )
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"]).sort_values("Date")
    returns = df["Returns"].dropna() * 100  # scale for stability
    returns.index = df.loc[returns.index, "Date"]
    return returns


def fit_model(returns, vol="GARCH", p=1, o=0, q=1, dist="t", mean="Zero"):
    model = arch_model(returns, mean=mean, vol=vol, p=p, o=o, q=q, dist=dist)
    fitted = model.fit(disp="off")
    return fitted


def persistence(params):
    alpha = params.get("alpha[1]", 0)
    beta = params.get("beta[1]", 0)
    return alpha + beta


def plot_volatility(fitted, name, filename):
    cond_vol = fitted.conditional_volatility / 100
    returns = fitted.resid / 100

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].plot(returns.index, returns, color="lightgray", linewidth=0.6, label="Returns")
    axes[0].plot(cond_vol.index, cond_vol, linewidth=1.5, color="#D62828", label="Cond. Volatility")
    axes[0].plot(cond_vol.index, -cond_vol, linewidth=1.0, color="#D62828")
    axes[0].set_title(f"{name}: Returns vs Conditional Volatility")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(cond_vol.index, cond_vol, linewidth=1.2, color="#2E86AB")
    axes[1].set_title(f"{name}: Conditional Volatility")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=300)
    plt.close(fig)


def plot_std_resid(fitted, name, filename):
    std_resid = fitted.std_resid
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(std_resid, linewidth=0.7, color="#2E86AB")
    axes[0, 0].set_title(f"{name}: Standardized Residuals")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].hist(std_resid, bins=40, density=True, alpha=0.7, color="#A23B72")
    axes[0, 1].set_title("Distribution")
    axes[0, 1].grid(alpha=0.3)

    from statsmodels.graphics.tsaplots import plot_acf

    plot_acf(std_resid, lags=40, ax=axes[1, 0])
    axes[1, 0].set_title("ACF of Std Residuals")

    plot_acf(std_resid ** 2, lags=40, ax=axes[1, 1])
    axes[1, 1].set_title("ACF of Sq Std Residuals")

    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=300)
    plt.close(fig)


def main():
    ensure_dirs()
    returns = load_returns()

    models = {
        "GARCH(1,1)": {"vol": "GARCH", "p": 1, "o": 0, "q": 1},
        "EGARCH(1,1)": {"vol": "EGARCH", "p": 1, "o": 0, "q": 1},
        "GJR-GARCH(1,1)": {"vol": "GARCH", "p": 1, "o": 1, "q": 1},
    }

    rows = []
    fitted_models = {}

    for name, params in models.items():
        print(f"Fitting {name}...")
        fitted = fit_model(returns, **params)
        fitted_models[name] = fitted
        rows.append(
            {
                "Model": name,
                "LogLik": fitted.loglikelihood,
                "AIC": fitted.aic,
                "BIC": fitted.bic,
                "Persistence": persistence(fitted.params),
            }
        )

    comparison_df = pd.DataFrame(rows).sort_values("AIC")
    comparison_df.to_csv(OUTPUT_DIR / "garch_extension_comparison.csv", index=False)
    print(comparison_df)

    best_name = comparison_df.iloc[0]["Model"]
    best_model = fitted_models[best_name]

    plot_volatility(best_model, best_name, "28_egarch_volatility.png")
    plot_std_resid(best_model, best_name, "29_egarch_residuals.png")

    print(f"\n[OK] Advanced GARCH analysis completed. Best model: {best_name}")


if __name__ == "__main__":
    main()



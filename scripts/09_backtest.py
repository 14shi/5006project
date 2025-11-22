"""

Script 09: Backtesting & Practical Utility Evaluation

目的：
- 对机器学习模型 (脚本 08) 的预测进行简单策略回测，展示其现实效用
- 输出收益、波动率、最大回撤等指标，满足项目要求中“说明模型价值”的部分

策略设定：
1. Regression_RF：若随机森林预测收益 > 0 → 持有多头，否则空头
2. Regression_RF_Targeted：仅当预测绝对收益 > 0.15% 时才开仓，否则空仓
3. Logistic_Prob：若逻辑回归概率 > 0.55 → 多头；< 0.45 → 空头；其余空仓

输出：
- output/backtest_summary.csv
- output/figures/25_strategy_equity.png

"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = Path("data/hsi_features.csv")
PRED_PATH = Path("output/ml_predictions.csv")
OUTPUT_DIR = Path("output")
FIG_DIR = OUTPUT_DIR / "figures"


def ensure_directories():
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)


def load_data():
    if not DATA_PATH.exists() or not PRED_PATH.exists():
        raise FileNotFoundError(
            "缺少特征数据或预测结果，请先运行 scripts/07_feature_engineering.py 和 scripts/08_ml_models.py"
        )
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    pred = pd.read_csv(PRED_PATH, parse_dates=["Date"])
    df = df.sort_values("Date")
    pred = pred.sort_values("Date")
    return df, pred


def compute_strategy_metrics(returns: pd.Series, strategy_returns: pd.Series, name: str) -> Dict[str, float]:
    cum_returns = (1 + strategy_returns).cumprod()
    total_return = cum_returns.iloc[-1] - 1
    annualized_return = strategy_returns.mean() * 252
    annualized_vol = strategy_returns.std(ddof=0) * np.sqrt(252)
    sharpe = annualized_return / (annualized_vol + 1e-9)

    running_max = cum_returns.cummax()
    drawdown = cum_returns / running_max - 1
    max_drawdown = drawdown.min()

    hit_rate = (np.sign(strategy_returns) == np.sign(returns)).mean()

    return {
        "Strategy": name,
        "Total_Return": total_return,
        "Annualized_Return": annualized_return,
        "Annualized_Vol": annualized_vol,
        "Sharpe": sharpe,
        "Max_Drawdown": max_drawdown,
        "Hit_Rate": hit_rate,
        "Trades": (strategy_returns != 0).sum(),
        "Long_Ratio": (strategy_returns > 0).sum() / len(strategy_returns),
        "Short_Ratio": (strategy_returns < 0).sum() / len(strategy_returns),
    }


def build_signals(test_df: pd.DataFrame, pred_df: pd.DataFrame) -> Dict[str, pd.Series]:
    signals = {}

    # 取随机森林回归预测
    rf_pred = pred_df[(pred_df["Model"] == "RandomForest") & (pred_df["Type"] == "Regression")]
    rf_pred = rf_pred.set_index("Date")["Prediction"].reindex(test_df["Date"])
    signals["Regression_RF"] = np.sign(rf_pred).fillna(0)
    signals["Regression_RF_Targeted"] = np.where(rf_pred.abs() > 0.0015, np.sign(rf_pred), 0)

    # 逻辑回归概率策略
    lr_pred = pred_df[(pred_df["Model"] == "LogisticRegression") & (pred_df["Type"] == "Classification")]
    lr_pred = lr_pred.set_index("Date")
    prob = lr_pred["Probability"].reindex(test_df["Date"])
    signal = np.zeros(len(prob))
    signal[prob > 0.55] = 1
    signal[prob < 0.45] = -1
    signals["Logistic_Prob"] = pd.Series(signal, index=test_df["Date"])

    return {name: pd.Series(sig, index=test_df["Date"]) for name, sig in signals.items()}


def main():
    ensure_directories()
    df, pred = load_data()

    test_window = 150
    test_df = df.iloc[-test_window:].copy()
    actual_returns = test_df["Target_Return_1d"].values
    actual_series = pd.Series(actual_returns, index=test_df["Date"])

    signals = build_signals(test_df, pred)

    equity = pd.DataFrame({"Date": test_df["Date"], "Buy_and_Hold": actual_series.cumsum()})
    metrics = []

    for name, sig in signals.items():
        strat_returns = sig.values * actual_returns
        metrics.append(compute_strategy_metrics(actual_series, pd.Series(strat_returns, index=test_df["Date"]), name))
        equity[name] = (1 + pd.Series(strat_returns, index=test_df["Date"])).cumprod()

    # Buy & Hold基准（满仓，参考实际收益）
    bh_returns = actual_series
    metrics.append(compute_strategy_metrics(bh_returns, bh_returns, "Buy_and_Hold"))

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(OUTPUT_DIR / "backtest_summary.csv", index=False)

    # 绘制权益曲线
    plt.figure(figsize=(12, 6))
    for col in equity.columns[1:]:
        plt.plot(equity["Date"], equity[col], label=col)
    plt.title("策略权益曲线（测试期 150 天）", fontsize=14, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Equity Curve (normalized)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "25_strategy_equity.png", dpi=300)
    plt.close()

    print("\n[OK] 回测完成，结果已保存：")
    print(" - output/backtest_summary.csv")
    print(" - output/figures/25_strategy_equity.png")


if __name__ == "__main__":
    main()



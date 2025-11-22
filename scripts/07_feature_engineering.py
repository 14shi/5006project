"""
Script 07: Feature Engineering with Technical Indicators & External Drivers

This script enriches the HSI dataset with:
- Technical indicators commonly discussed in the course (MA, RSI, MACD, Bollinger, etc.)
- Volatility/volume derived metrics
- External market drivers (FX, regional/global indices, volatility indices)
- Prediction targets for next-day return direction and volatility

Outputs:
- data/hsi_features.csv : feature-rich dataset
- output/feature_metadata.json : summary of generated features

"""

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
FIG_DIR = OUTPUT_DIR / "figures"

EXTERNAL_TICKERS = {
    "USDHKD": "USDHKD=X",          # 港币汇率 (反映资金流动/货币政策)
    "HSCEI": "^HSCE",              # 国企指数，衡量内地企业表现
    "SP500": "^GSPC",              # 全球风险情绪
    "VIX": "^VIX",                 # 全球波动率指标
    "US10Y": "^TNX"                # 美债收益率（资金成本）
}


def ensure_directories():
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)


def load_base_data(filepath: Path = DATA_DIR / "hsi_with_returns.csv") -> pd.DataFrame:
    """Load preprocessed HSI data (must contain Close, High, Low, Open, Volume, Returns)."""
    print(f"加载基础数据: {filepath}")
    df = pd.read_csv(filepath, parse_dates=["Date"], index_col="Date")
    print(f"数据条数: {len(df)}, 时间范围: {df.index.min().date()} ~ {df.index.max().date()}")
    return df


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # Moving averages
    for window in [5, 10, 20, 50, 100, 200]:
        df[f"SMA_{window}"] = close.rolling(window).mean()
        df[f"SMA_ratio_{window}"] = close / df[f"SMA_{window}"] - 1
    for span in [5, 12, 26, 50]:
        df[f"EMA_{span}"] = close.ewm(span=span, adjust=False).mean()

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # RSI / Stochastic Oscillator
    df["RSI_14"] = compute_rsi(close, window=14)
    lowest_low = low.rolling(window=14).min()
    highest_high = high.rolling(window=14).max()
    df["Stoch_%K"] = (close - lowest_low) / (highest_high - lowest_low + 1e-9)
    df["Stoch_%D"] = df["Stoch_%K"].rolling(window=3).mean()

    # Bollinger Bands
    rolling_mean = close.rolling(window=20).mean()
    rolling_std = close.rolling(window=20).std()
    df["Bollinger_upper"] = rolling_mean + 2 * rolling_std
    df["Bollinger_lower"] = rolling_mean - 2 * rolling_std
    df["Bollinger_bandwidth"] = (df["Bollinger_upper"] - df["Bollinger_lower"]) / (rolling_mean + 1e-9)
    df["Bollinger_percent"] = (close - df["Bollinger_lower"]) / ((df["Bollinger_upper"] - df["Bollinger_lower"]) + 1e-9)

    # ATR / True Range
    prev_close = close.shift(1)
    tr_components = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1)
    true_range = tr_components.max(axis=1)
    df["ATR_14"] = true_range.rolling(window=14).mean()

    # Returns & volatility windows
    df["Return_1d"] = df["Returns"]
    for window in [2, 3, 5, 10, 20]:
        df[f"Return_{window}d"] = close.pct_change(window)
        df[f"Volatility_{window}d"] = df["Returns"].rolling(window).std()

    # Momentum
    for window in [5, 10, 20]:
        df[f"Momentum_{window}d"] = close - close.shift(window)

    # Volume-based indicators
    df["Volume_change"] = volume.pct_change()
    for window in [5, 20]:
        df[f"Volume_SMA_{window}"] = volume.rolling(window).mean()
        df[f"Volume_ratio_{window}"] = volume / (df[f"Volume_SMA_{window}"] + 1e-9)

    # On-Balance Volume
    direction = np.sign(close.diff()).fillna(0)
    df["OBV"] = (direction * volume).cumsum()
    df["OBV_ratio_20"] = df["OBV"] / (df["OBV"].rolling(20).mean() + 1e-9)

    # Price range features
    df["High_Low_spread"] = (high - low) / close
    df["Close_Open_diff"] = (close - df["Open"]) / close
    df["HLCO_range"] = (high - low) / (df["Open"] + 1e-9)

    return df


def fetch_external_series(start: str, end: str) -> Dict[str, pd.Series]:
    """Download external market data and return daily percentage change series."""
    external_data = {}
    print("\n下载外部市场数据：")
    for name, ticker in EXTERNAL_TICKERS.items():
        try:
            print(f"- {name} ({ticker})")
            data = yf.download(ticker, start=start, end=end, progress=False)
            if data.empty:
                print(f"  [警告] {name} 数据为空，跳过")
                continue

            # 处理多层列索引 (Price/Ticker)
            if isinstance(data.columns, pd.MultiIndex):
                closes = data.xs("Close", axis=1, level=0)
            else:
                closes = data["Close"]

            # 单个ticker -> Series
            if isinstance(closes, pd.DataFrame):
                closes = closes.squeeze()

            closes = closes.rename(f"{name}_close")
            returns = closes.pct_change().rename(f"{name}_return")

            external_data[f"{name}_close"] = closes
            external_data[f"{name}_return"] = returns
        except Exception as exc:
            import traceback

            print(f"  [错误] 无法获取 {name} ({ticker}): {exc}")
            traceback.print_exc()
    return external_data


def build_feature_dataset(df: pd.DataFrame) -> pd.DataFrame:
    start = df.index.min().strftime("%Y-%m-%d")
    end = (df.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    df_features = add_technical_indicators(df)

    # External features
    external_series = fetch_external_series(start, end)
    if external_series:
        external_df = pd.concat(external_series.values(), axis=1)
        df_features = df_features.join(external_df, how="left")
        df_features[external_df.columns] = df_features[external_df.columns].fillna(method="ffill")

    # Targets (predict next-day return, direction, volatility)
    df_features["Target_Return_1d"] = df_features["Returns"].shift(-1)
    df_features["Target_Direction"] = (df_features["Target_Return_1d"] > 0).astype(int)
    df_features["Target_Volatility"] = df_features["Returns"].shift(-1).abs()

    # Drop rows with insufficient history or missing targets
    df_features = df_features.dropna().copy()

    # Add metadata columns
    df_features["Date"] = df_features.index

    return df_features


def save_outputs(df_features: pd.DataFrame):
    feature_path = DATA_DIR / "hsi_features.csv"
    df_features.to_csv(feature_path, index=False)
    print(f"\n[OK] 已保存特征数据集: {feature_path} (行数: {len(df_features)})")

    metadata = {
        "total_rows": len(df_features),
        "start_date": str(df_features["Date"].min().date()),
        "end_date": str(df_features["Date"].max().date()),
        "feature_count": len(df_features.columns) - 1,  # exclude Date
        "targets": ["Target_Return_1d", "Target_Direction", "Target_Volatility"],
        "external_features": list(EXTERNAL_TICKERS.keys()),
    }

    metadata_path = OUTPUT_DIR / "feature_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"[OK] 已保存特征摘要: {metadata_path}")


def main():
    ensure_directories()
    base_df = load_base_data()
    feature_df = build_feature_dataset(base_df)
    save_outputs(feature_df)

    print("\n特征工程完成！建议下一步：")
    print("1) 使用新特征训练机器学习模型 (scripts/08_ml_models.py)")
    print("2) 进行回测与实用性评估")


if __name__ == "__main__":
    main()



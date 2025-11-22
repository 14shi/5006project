"""

Script 08: Machine Learning Models for Enhanced Forecasting

目标：
- 利用 scripts/07_feature_engineering.py 生成的特征，构建更具“现实效用”的预测模型
- 同时评估收益率预测（回归）与方向预测（分类）的表现
- 输出可复现的指标、交叉验证结果以及特征重要性，方便在报告中解释“为何模型有效/无效”

输出文件：
- output/ml_regression_metrics.csv
- output/ml_classification_metrics.csv
- output/ml_cv_scores.csv
- output/ml_predictions.csv
- output/ml_feature_importance.csv

"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("data/hsi_features.csv")
OUTPUT_DIR = Path("output")


def ensure_directories():
    OUTPUT_DIR.mkdir(exist_ok=True)


def load_feature_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 不存在，请先运行 scripts/07_feature_engineering.py 生成特征数据。"
        )
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    print(f"加载特征数据: {path}，行数 {len(df)}, 列数 {len(df.columns)}")
    return df


def train_test_split(df: pd.DataFrame, test_days: int = 150) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if test_days >= len(df):
        raise ValueError("test_days 太大，无法划分训练/测试集")
    train_df = df.iloc[:-test_days].copy()
    test_df = df.iloc[-test_days:].copy()
    print(
        f"训练集: {len(train_df)} ({train_df['Date'].min().date()} ~ {train_df['Date'].max().date()})\n"
        f"测试集: {len(test_df)} ({test_df['Date'].min().date()} ~ {test_df['Date'].max().date()})"
    )
    return train_df, test_df


def prepare_features(
    df: pd.DataFrame,
    drop_cols: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(method="bfill")
    return X, feature_cols


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    r2 = r2_score(y_true, y_pred)
    me = np.mean(y_true - y_pred)
    direction_true = (y_true > 0).astype(int)
    direction_pred = (y_pred > 0).astype(int)
    dir_acc = accuracy_score(direction_true, direction_pred)
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
        "ME": me,
        "Direction_Accuracy": dir_acc,
    }


def evaluate_classification(
    y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray | None = None
) -> Dict[str, float]:
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }
    if proba is not None:
        try:
            metrics["ROC_AUC"] = roc_auc_score(y_true, proba)
        except ValueError:
            metrics["ROC_AUC"] = np.nan
    else:
        metrics["ROC_AUC"] = np.nan
    return metrics


def cross_validate_model(model, X: pd.DataFrame, y: np.ndarray, task: str, splits: int = 5):
    tscv = TimeSeriesSplit(n_splits=splits)
    scores = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        if task == "regression":
            score = evaluate_regression(y_val, preds)
        else:
            proba = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_val)[:, 1]
            score = evaluate_classification(y_val, (preds > 0.5).astype(int) if task == "classification_proba" else preds, proba)
        score["Fold"] = fold
        score["Task"] = task
        score["Model"] = model.named_steps if isinstance(model, Pipeline) else str(model)
        scores.append(score)
    return scores


def main():
    ensure_directories()
    df = load_feature_dataset()

    target_cols = ["Target_Return_1d", "Target_Direction", "Target_Volatility"]
    drop_cols = target_cols + ["Date"]

    X_all, feature_cols = prepare_features(df, drop_cols=drop_cols)
    y_return = df["Target_Return_1d"].values
    y_direction = df["Target_Direction"].values

    # 划分训练/测试（末150日为测试）
    train_df, test_df = train_test_split(df)
    X_train = X_all.loc[train_df.index]
    X_test = X_all.loc[test_df.index]
    y_train_reg = train_df["Target_Return_1d"].values
    y_test_reg = test_df["Target_Return_1d"].values
    y_train_cls = train_df["Target_Direction"].values
    y_test_cls = test_df["Target_Direction"].values

    # 定义模型
    reg_models = {
        "Ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_leaf=10, random_state=42
        ),
    }
    cls_models = {
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    }

    # 训练 + 评估
    reg_results = []
    cls_results = []
    predictions = []

    for name, model in reg_models.items():
        print(f"\n训练回归模型：{name}")
        model.fit(X_train, y_train_reg)
        preds = model.predict(X_test)
        metrics = evaluate_regression(y_test_reg, preds)
        metrics["Model"] = name
        reg_results.append(metrics)
        predictions.append(
            pd.DataFrame(
                {
                    "Date": test_df["Date"],
                    "Model": name,
                    "Type": "Regression",
                    "Actual": y_test_reg,
                    "Prediction": preds,
                }
            )
        )

    for name, model in cls_models.items():
        print(f"\n训练分类模型：{name}")
        model.fit(X_train, y_train_cls)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        metrics = evaluate_classification(y_test_cls, preds, proba)
        metrics["Model"] = name
        cls_results.append(metrics)
        predictions.append(
            pd.DataFrame(
                {
                    "Date": test_df["Date"],
                    "Model": name,
                    "Type": "Classification",
                    "Actual": y_test_cls,
                    "Prediction": preds,
                    "Probability": proba,
                }
            )
        )

    # 特征重要性（随机森林）
    rf_model = reg_models["RandomForest"]
    if hasattr(rf_model, "feature_importances_"):
        importances = rf_model.feature_importances_
    else:
        rf_model.fit(X_train, y_train_reg)
        importances = rf_model.feature_importances_
    feature_importance = pd.DataFrame(
        {"Feature": feature_cols, "Importance": importances}
    ).sort_values("Importance", ascending=False)

    # 保存结果
    reg_df = pd.DataFrame(reg_results)
    cls_df = pd.DataFrame(cls_results)
    pred_df = pd.concat(predictions, ignore_index=True)

    reg_df.to_csv(OUTPUT_DIR / "ml_regression_metrics.csv", index=False)
    cls_df.to_csv(OUTPUT_DIR / "ml_classification_metrics.csv", index=False)
    pred_df.to_csv(OUTPUT_DIR / "ml_predictions.csv", index=False)
    feature_importance.to_csv(OUTPUT_DIR / "ml_feature_importance.csv", index=False)

    print("\n[OK] 已保存机器学习模型评估结果：")
    print(" - output/ml_regression_metrics.csv")
    print(" - output/ml_classification_metrics.csv")
    print(" - output/ml_predictions.csv")
    print(" - output/ml_feature_importance.csv")


if __name__ == "__main__":
    main()



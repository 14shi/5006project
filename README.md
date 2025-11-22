# MSBD5006 恒生指数时间序列分析项目

## 1. 项目概述

本项目使用 2015 年 1 月至 2025 年 11 月的恒生指数（HSI）日度数据，完成以下工作：
1. 数据获取与清洗；
2. 探索性分析及平稳性、相关性检验；
3. ARIMA 与 GARCH（含 GJR、EGARCH）建模；
4. 基于技术指标与外部市场因子的特征工程；
5. 机器学习预测（Ridge、RandomForest、Logistic、GradientBoosting）及策略回测；
6. Johansen 协整检验与 VAR（含 Granger 因果、冲击响应、方差分解）分析。

所有脚本位于 `scripts/` 目录，数据与图表输出放置在 `data/` 与 `output/` 目录，综合分析见 `notebooks/main_analysis.ipynb`。

## 2. 目录结构

```
5006-pro/
├── data/
│   ├── hsi_raw.csv               # 恒指原始数据
│   ├── hsi_with_returns.csv      # 含收益率的处理数据
│   └── hsi_features.csv          # 技术指标与外部因子特征
│
├── scripts/
│   ├── 00_data_collection.py     # 数据下载与验证
│   ├── 01_eda.py                 # 探索性分析
│   ├── 02_stationarity.py        # 平稳性检验
│   ├── 03_correlation.py         # 自相关与 ARCH 检验
│   ├── 04_arima.py               # ARIMA 建模
│   ├── 05_garch.py               # 基准 GARCH 建模
│   ├── 06_forecasting.py         # 滚动预测与评估
│   ├── 07_feature_engineering.py # 技术指标与外部因子构建
│   ├── 08_ml_models.py           # 机器学习预测
│   ├── 09_backtest.py            # 策略回测
│   ├── 10_var_analysis.py        # 协整与 VAR 诊断
│   ├── 11_garch_extensions.py    # GJR/EGARCH 比较
│   ├── 12_garch_x.py             # 加入外生因子的 GARCH-X
│   └── 13_indicator_tests.py     # 技术指标显著性检验
│
├── notebooks/
│   └── main_analysis.ipynb       # 综合分析 Notebook
│
├── output/
│   ├── figures/01–31.png         # 全部图表
│   ├── forecast_comparison.csv   # ARIMA 及基准比较
│   ├── forecast_results.csv      # ARIMA 预测结果
│   ├── ml_*                      # 机器学习评估与特征重要性
│   ├── backtest_summary.csv      # 策略回测指标
│   ├── var_*                     # VAR 相关检验
│   ├── garch_extension_comparison.csv
│   ├── garch_x_results.csv       # GARCH vs. GARCH-X 对比
│   ├── garch_x_coefficients.csv  # 外生因子系数显著性
│   └── indicator_predictability.csv
│
├── requirements.txt              # 依赖说明
└── PROJECT_SUMMARY.md            # 中文项目总结
```

## 3. 运行步骤

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
2. 按顺序执行脚本（若已下载数据可跳过 00）：
   ```bash
   python scripts/00_data_collection.py
   python scripts/01_eda.py
   python scripts/02_stationarity.py
   python scripts/03_correlation.py
   python scripts/04_arima.py
   python scripts/05_garch.py
   python scripts/06_forecasting.py
   python scripts/07_feature_engineering.py
   python scripts/08_ml_models.py
   python scripts/09_backtest.py
   python scripts/10_var_analysis.py
   python scripts/11_garch_extensions.py
   python scripts/12_garch_x.py
   python scripts/13_indicator_tests.py
   ```
3. 运行 `notebooks/main_analysis.ipynb`，即可查看全部图表与解释。

## 4. 方法概述

1. **探索性分析**：绘制价格、收益率、分布、滚动统计与成交量，识别数据特征。  
2. **平稳性与相关性**：ADF/KPSS 检验表明收益率平稳；ACF/PACF 与 Ljung-Box 检验揭示 ARCH 效应。  
3. **ARIMA**：通过 AIC/BIC 筛选得到 ARIMA(2,0,3)，在 252 个测试交易日上的方向准确率约 52%。  
4. **GARCH 及扩展**：基准 GARCH(1,1)-t 描述波动聚集，GJR-GARCH 在 AIC/BIC 上更优，能够刻画负向冲击导致的波动放大。  
5. **GARCH-X**：在均值方程引入 S&P500、VIX、USDHKD 变动及 SMA/RSI 指标，显著降低 AIC（8181 → 7488），说明外部因子对波动率与收益具有解释力。  
6. **特征工程与机器学习**：构建 70 余个技术指标及 USDHKD、HSCEI、S&P500、VIX、US10Y 等外部因子；随机森林与逻辑回归在方向预测上略优于 ARIMA。  
7. **策略回测**：基于机器学习输出构建多/空/观望策略，测试期内 Regression_RF_Targeted 的年化收益 49%、Sharpe 3.88。  
8. **VAR/协整**：Johansen 检验显示样本期内无显著协整；差分 VAR 表明 S&P500 对 HSI 具有显著 Granger 因果，冲击影响约 5–7 个交易日。  
9. **技术指标显著性检验**：使用 OLS/Ljung-Box/Granger 检验 5 个指标，结果显示 S&P500、VIX 对下一期收益显著（p<0.01），多数纯技术指标未表现出显著预测力。

## 5. 结果摘要

- ARIMA 基准：R² 0.018，方向准确率 52.38%，仍符合弱式有效市场假设。  
- GJR-GARCH：在 GJR/EGARCH 扩展中 AIC/BIC 最低，条件波动率能识别高风险区段。  
- GARCH-X：引入 S&P500、VIX、USDHKD 与 SMA/RSI 后，AIC 降至 7488，S&P500 与 VIX 系数在 1% 水平显著，表明外部冲击对收益具有即时影响。  
- VAR：S&P500 → HSI 的 Granger 因果显著，冲击响应显示全球风险情绪对恒指有短期影响。  
- 机器学习：RandomForest 测试期 MAE 0.0089、R² 0.046、方向准确率 57%；Logistic 准确率 58%、召回率 79%。  
- 策略：在忽略交易成本的假设下，基于概率阈值的策略在风险调整收益上优于 Buy & Hold；报告已说明现实执行仍需考虑成本与市场效率。  
- 指标检验：`indicator_predictability.csv` 显示纯价格指标的 t 统计值不显著，而 S&P500 / VIX 变动的 t 统计绝对值大于 13，验证外部因子在课程方法中的作用。

## 6. 数据与图表说明

- `data/`：包括原始数据、处理数据、特征数据，可复现所有分析。  
- `output/figures/`：共 31 张图，涵盖 EDA、平稳性、ACF/PACF、ARIMA/GARCH 诊断、预测结果、策略曲线、VAR IRF/FEVD、GARCH-X 与指标检验等。  
- `output/*.csv`：共 24 个，记录各阶段的统计指标、预测结果、机器学习评估、回测指标及 VAR/GARCH 扩展与指标检验比较。  
- `output/best_arima_model.pkl` 与 `output/best_garch_model.pkl`：用于保存训练好的模型。  

## 7. 报告与展示

- `notebooks/main_analysis.ipynb`：整合所有图表与文字说明，可导出为 PDF 作为课程报告。  
- `PROJECT_SUMMARY.md`：中文总结报告，便于快速审阅项目背景、方法与结果。  
- 推荐展示图表：01（价格）、11（ACF/PACF）、17（ARIMA 诊断）、20（GARCH 波动率）、22（ARIMA 预测）、24（方法比较）、25（策略曲线）、26–27（VAR IRF/FEVD）、28–29（GARCH 扩展）、30（GARCH-X 波动率）、31（指标检验 t 统计）。





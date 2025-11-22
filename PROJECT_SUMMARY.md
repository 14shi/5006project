
## 1. 项目概述

本项目以恒生指数（HSI）2015 年至 2025 年的日度数据为研究对象，完成了从数据获取、探索性分析、平稳性检验、时间序列建模、波动率建模、特征扩展、机器学习预测、策略回测到协整/VAR 分析的完整流程。所有脚本均存放于 `scripts/` 目录，数据和图表输出位于 `data/` 与 `output/` 目录，综合分析见 `notebooks/main_analysis.ipynb`。  

## 2. 数据与预处理

1. `scripts/00_data_collection.py` 使用 `yfinance` 下载恒指原始数据（共 2,679 交易日），保存为 `data/hsi_raw.csv`。  
2. `scripts/01_eda.py` 完成缺失值检查、收益率与对数收益率计算、分布分析、滚动统计、成交量分析，生成 `data/hsi_with_returns.csv` 与 01–06 号图表。  
3. 所有脚本默认依赖 `requirements.txt` 中的环境，图表输出统一保存在 `output/figures/`。  

## 3. 分析方法

### 3.1 平稳性与相关性

- `scripts/02_stationarity.py`：对价格、收益率、对数收益率进行 ADF、KPSS 检验及滚动均值/方差分析（图 07–10），结论为价格非平稳而收益率平稳。  
- `scripts/03_correlation.py`：输出 ACF/PACF、平方收益率 ACF/PACF、Ljung-Box 检验及相关性热图（图 11–16），确认收益率存在弱自相关及明显 ARCH 效应。  

### 3.2 ARIMA 与预测

- `scripts/04_arima.py`：以收益率为输入，通过网格搜索选择 ARIMA(2,0,3)，输出诊断图与比较表（图 17–18、`output/arima_*`）。  
- `scripts/06_forecasting.py`：固定训练集 2529 天、测试集 252 天（约 12 个月），进行滚动预测，并与 Naive / Historical Mean 对比（图 22、`output/forecast_*`）。ARIMA 的测试期 R² 为 0.018，方向准确率 52.38%。  

### 3.3 波动率建模

- `scripts/05_garch.py`：对收益率拟合 GARCH(1,1)-t，并输出条件波动率、残差诊断、模型比较（图 19–21、`output/garch_*`）。  
- `scripts/11_garch_extensions.py`：进一步比较 GARCH、EGARCH、GJR-GARCH，`garch_extension_comparison.csv` 显示 GJR-GARCH(1,1) 的 AIC 最低（8820.46），且持久性 0.95，优于标准 GARCH 的 0.99。相关图表为 28–29 号。  
- `scripts/12_garch_x.py`：在均值方程中加入 `SMA_ratio_20`、`RSI_14`、`SP500_return`、`VIX_return`、`USDHKD_return`，构建 GARCH-X。结果显示 AIC 从 8181 降至 7488，S&P500 与 VIX 系数在 1% 水平显著（图 30、`output/garch_x_*`）。  

### 3.4 特征工程与机器学习

- `scripts/07_feature_engineering.py`：构建 70 余个技术指标（SMA、EMA、MACD、RSI、Bollinger、ATR、成交量、OBV 等）并引入外部市场因子（USDHKD、HSCEI、S&P500、VIX、US10Y），生成 `data/hsi_features.csv` 及 `output/feature_metadata.json`。  
- `scripts/08_ml_models.py`：基于新特征训练 Ridge、RandomForest、Logistic、GradientBoosting。RandomForest 的测试期 MAE 为 0.0089、R² 为 0.046，方向准确率 57%；Logistic 的准确率 58%、召回率 79%（`output/ml_*`）。  
- `scripts/09_backtest.py`：将机器学习预测转化为多/空/观望信号，回测 150 天。Regression_RF_Targeted 策略在测试期取得年化收益 49%、Sharpe 3.88、最大回撤 -3.3%；Logistic_Prob 策略年化收益 37.9%、Sharpe 2.73（`output/backtest_summary.csv`、图 25）。  

### 3.5 协整与 VAR
### 3.6 技术指标显著性检验

- `scripts/13_indicator_tests.py`：基于 `hsi_features.csv`，使用 OLS、Ljung-Box、Granger 检验 SMA、RSI、MACD、Bollinger 以及 S&P500、VIX 收益对下一期收益的解释力。输出 `indicator_predictability.csv` 及 t 统计图（图 31），结果表明纯价格指标的 t 统计不显著，而 S&P500/VIX 的 t 统计绝对值大于 13。

- `scripts/10_var_analysis.py`：对 HSI、HSCEI、USDHKD、S&P500 的对数价格执行 ADF 与 Johansen 检验（`var_adf_results.csv`、`var_johansen_trace.csv`），结果显示样本期内不存在显著协整。  
- 对差分后数据拟合 VAR 模型并进行 Granger 因果、冲击响应、方差分解（图 26–27、`var_granger_results.csv`）。S&P500 对 HSI 的 Granger 因果在 1% 水平显著，冲击影响约 5–7 个交易日。  

## 4. 结果总结

1. **ARIMA**：在 252 个测试交易日上，R² 0.018，方向准确率 52.38%，结果仍表明恒指收益接近随机游走。  
2. **GARCH 家族**：GJR-GARCH 捕捉负向冲击导致的波动放大效应；GARCH-X 进一步证明 S&P500 与 VIX 可显著改善均值方程拟合，AIC 显著下降。  
3. **VAR**：未发现显著协整关系，但 S&P500 对 HSI 的短期冲击可通过 Granger/IRF 量化，支持“全球风险偏好影响恒指”的判断。  
4. **机器学习与策略**：RandomForest 和 Logistic 在方向预测上比 ARIMA 更优，通过阈值策略可在无交易成本假设下得到较高的 Sharpe 值；但报告中已强调真实交易仍受成本和市场效率限制。  
5. **指标检验**：OLS/Ljung-Box/Granger 结果表明，大部分单一技术指标缺乏稳定预测力，而外部因子（S&P500、VIX）在 1% 水平显著；因此外生变量是改善波动率与收益预测的重要来源。  
6. **模块影响说明**：滚动预测阶段使用 252 个测试交易日，其结果仅影响 `scripts/06_forecasting.py` 及相关图表/指标；机器学习与策略回测仍保持 150 日窗口，以便与既有分析对比；VAR、GARCH 扩展等模块与该改动无直接耦合。  

## 5. 文件与运行说明

- **脚本**：`scripts/00_data_collection.py` 至 `scripts/13_indicator_tests.py`。运行顺序可参考 README 中的 “Quick Start”。  
- **数据**：`data/hsi_raw.csv`、`data/hsi_with_returns.csv`、`data/hsi_features.csv`。  
- **图表**：`output/figures/01_price_series.png` 至 `output/figures/31_indicator_tstats.png`。  
- **CSV 输出**：共 24 个，覆盖 ARIMA/GARCH/ML/BACKTEST/VAR/GARCH-X/指标检验等所有阶段。  
- **模型文件**：`output/best_arima_model.pkl`、`output/best_garch_model.pkl`。  
- **Notebook**：`notebooks/main_analysis.ipynb`，可直接运行或导出为 PDF/HTML 作为最终报告。  
- **README.md**：说明项目结构、依赖安装、脚本作用以及输出文件列表。  

 **推荐图表**：01（价格）、11（ACF/PACF）、17（ARIMA 诊断）、20（GARCH 波动率）、22（ARIMA 预测）、24（方法对比）、25（策略权益）、26–27（VAR IRF/FEVD）、28–29（GARCH 扩展）、30（GARCH-X 波动率）、31（指标检验）。  


## 7. 结论

项目完成了从传统时间序列方法到多变量分析和机器学习扩展的完整流程。ARIMA 与 GARCH 部分验证了弱式有效市场假说；VAR 分析展示了全球市场冲击对恒指的传导路径；机器学习和策略回测说明在引入技术指标与外部因子后，可获得小幅度的方向改进。所有代码和结果已经过整理，读者可根据 README 提供的步骤复现整个分析，并在 `notebooks/main_analysis.ipynb` 中查看完整报告。  

***
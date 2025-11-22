"""
Script 02: Stationarity Analysis

This script performs stationarity tests on HSI data including ADF and KPSS tests,
and analyzes rolling statistics to assess time series properties.


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_data(filepath='data/hsi_with_returns.csv'):
    """Load processed HSI data"""
    print("Loading processed HSI data...")
    df = pd.read_csv(filepath, index_col=0, parse_dates=True, header=[0, 1])
    
    # Flatten column names if multi-level
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    print(f"Data loaded: {len(df)} observations")
    return df

def adf_test(series, name="Series"):
    """
    Perform Augmented Dickey-Fuller test for stationarity
    
    Null Hypothesis: Series has a unit root (non-stationary)
    Alternative: Series is stationary
    """
    print(f"\n{'='*60}")
    print(f"AUGMENTED DICKEY-FULLER TEST: {name}")
    print(f"{'='*60}")
    
    # Drop NaN values
    series_clean = series.dropna()
    
    # Perform ADF test
    result = adfuller(series_clean, autolag='AIC')
    
    print(f"ADF Statistic: {result[0]:.6f}")
    print(f"p-value: {result[1]:.6f}")
    print(f"Used lag: {result[2]}")
    print(f"Number of observations: {result[3]}")
    print(f"\nCritical Values:")
    for key, value in result[4].items():
        print(f"  {key}: {value:.4f}")
    
    # Interpretation
    if result[1] < 0.05:
        print(f"\n[RESULT] Reject null hypothesis (p < 0.05)")
        print(f"[CONCLUSION] {name} is STATIONARY")
        is_stationary = True
    else:
        print(f"\n[RESULT] Fail to reject null hypothesis (p >= 0.05)")
        print(f"[CONCLUSION] {name} is NON-STATIONARY")
        is_stationary = False
    
    return {
        'test': 'ADF',
        'statistic': result[0],
        'p_value': result[1],
        'lags': result[2],
        'nobs': result[3],
        'critical_values': result[4],
        'is_stationary': is_stationary
    }

def kpss_test(series, name="Series", regression='c'):
    """
    Perform KPSS test for stationarity
    
    Null Hypothesis: Series is stationary
    Alternative: Series has a unit root (non-stationary)
    
    regression: 'c' for constant (level stationary)
                'ct' for constant and trend (trend stationary)
    """
    print(f"\n{'='*60}")
    print(f"KPSS TEST: {name}")
    print(f"{'='*60}")
    
    # Drop NaN values
    series_clean = series.dropna()
    
    # Perform KPSS test
    result = kpss(series_clean, regression=regression, nlags='auto')
    
    print(f"KPSS Statistic: {result[0]:.6f}")
    print(f"p-value: {result[1]:.6f}")
    print(f"Used lag: {result[2]}")
    print(f"\nCritical Values:")
    for key, value in result[3].items():
        print(f"  {key}: {value:.4f}")
    
    # Interpretation
    if result[1] < 0.05:
        print(f"\n[RESULT] Reject null hypothesis (p < 0.05)")
        print(f"[CONCLUSION] {name} is NON-STATIONARY")
        is_stationary = False
    else:
        print(f"\n[RESULT] Fail to reject null hypothesis (p >= 0.05)")
        print(f"[CONCLUSION] {name} is STATIONARY")
        is_stationary = True
    
    return {
        'test': 'KPSS',
        'statistic': result[0],
        'p_value': result[1],
        'lags': result[2],
        'critical_values': result[3],
        'is_stationary': is_stationary
    }

def plot_rolling_stats(df, series_col, title, window=30):
    """Plot series with rolling mean and standard deviation"""
    series = df[series_col].dropna()
    
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot original series
    ax.plot(series.index, series, color='#2E86AB', label='Original', linewidth=1, alpha=0.7)
    ax.plot(rolling_mean.index, rolling_mean, color='#D62828', 
            label=f'Rolling Mean ({window}-day)', linewidth=2)
    ax.plot(rolling_std.index, rolling_std, color='#F18F01', 
            label=f'Rolling Std ({window}-day)', linewidth=2)
    
    ax.set_title(f'{title}: Rolling Statistics', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    return fig

def comprehensive_stationarity_analysis(df):
    """Perform comprehensive stationarity analysis"""
    
    results = []
    
    # Test 1: Price Level (Close)
    print("\n" + "="*60)
    print("TEST 1: PRICE LEVEL (Closing Price)")
    print("="*60)
    adf_price = adf_test(df['Close'], "Closing Price")
    kpss_price = kpss_test(df['Close'], "Closing Price", regression='ct')  # with trend
    results.append({
        'Series': 'Closing Price',
        'ADF_stat': adf_price['statistic'],
        'ADF_pval': adf_price['p_value'],
        'ADF_conclusion': 'Stationary' if adf_price['is_stationary'] else 'Non-Stationary',
        'KPSS_stat': kpss_price['statistic'],
        'KPSS_pval': kpss_price['p_value'],
        'KPSS_conclusion': 'Stationary' if kpss_price['is_stationary'] else 'Non-Stationary'
    })
    
    # Plot rolling stats for price
    fig1 = plot_rolling_stats(df, 'Close', 'Closing Price')
    fig1.savefig('output/figures/07_stationarity_price.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n[OK] Saved: output/figures/07_stationarity_price.png")
    
    # Test 2: Returns
    print("\n" + "="*60)
    print("TEST 2: DAILY RETURNS")
    print("="*60)
    adf_returns = adf_test(df['Returns'], "Daily Returns")
    kpss_returns = kpss_test(df['Returns'], "Daily Returns", regression='c')  # constant only
    results.append({
        'Series': 'Returns',
        'ADF_stat': adf_returns['statistic'],
        'ADF_pval': adf_returns['p_value'],
        'ADF_conclusion': 'Stationary' if adf_returns['is_stationary'] else 'Non-Stationary',
        'KPSS_stat': kpss_returns['statistic'],
        'KPSS_pval': kpss_returns['p_value'],
        'KPSS_conclusion': 'Stationary' if kpss_returns['is_stationary'] else 'Non-Stationary'
    })
    
    # Plot rolling stats for returns
    fig2 = plot_rolling_stats(df, 'Returns', 'Daily Returns')
    fig2.savefig('output/figures/08_stationarity_returns.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n[OK] Saved: output/figures/08_stationarity_returns.png")
    
    # Test 3: Log Returns
    print("\n" + "="*60)
    print("TEST 3: LOG RETURNS")
    print("="*60)
    adf_log_returns = adf_test(df['Log_Returns'], "Log Returns")
    kpss_log_returns = kpss_test(df['Log_Returns'], "Log Returns", regression='c')
    results.append({
        'Series': 'Log Returns',
        'ADF_stat': adf_log_returns['statistic'],
        'ADF_pval': adf_log_returns['p_value'],
        'ADF_conclusion': 'Stationary' if adf_log_returns['is_stationary'] else 'Non-Stationary',
        'KPSS_stat': kpss_log_returns['statistic'],
        'KPSS_pval': kpss_log_returns['p_value'],
        'KPSS_conclusion': 'Stationary' if kpss_log_returns['is_stationary'] else 'Non-Stationary'
    })
    
    # Plot rolling stats for log returns
    fig3 = plot_rolling_stats(df, 'Log_Returns', 'Log Returns')
    fig3.savefig('output/figures/09_stationarity_log_returns.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n[OK] Saved: output/figures/09_stationarity_log_returns.png")
    
    # Test 4: First Difference of Price
    print("\n" + "="*60)
    print("TEST 4: FIRST DIFFERENCE OF PRICE")
    print("="*60)
    df['Price_Diff'] = df['Close'].diff()
    adf_diff = adf_test(df['Price_Diff'], "First Difference")
    kpss_diff = kpss_test(df['Price_Diff'], "First Difference", regression='c')
    results.append({
        'Series': 'First Difference',
        'ADF_stat': adf_diff['statistic'],
        'ADF_pval': adf_diff['p_value'],
        'ADF_conclusion': 'Stationary' if adf_diff['is_stationary'] else 'Non-Stationary',
        'KPSS_stat': kpss_diff['statistic'],
        'KPSS_pval': kpss_diff['p_value'],
        'KPSS_conclusion': 'Stationary' if kpss_diff['is_stationary'] else 'Non-Stationary'
    })
    
    return pd.DataFrame(results)

def create_summary_plot(results_df):
    """Create a summary visualization of stationarity tests"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # ADF Test Statistics
    ax1 = axes[0, 0]
    colors = ['#06A77D' if x == 'Stationary' else '#D62828' 
              for x in results_df['ADF_conclusion']]
    ax1.barh(results_df['Series'], results_df['ADF_stat'], color=colors, alpha=0.7, edgecolor='black')
    ax1.axvline(x=-3.43, color='black', linestyle='--', linewidth=2, label='5% Critical Value')
    ax1.set_xlabel('ADF Statistic', fontsize=11)
    ax1.set_title('ADF Test Statistics\n(More negative = More stationary)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='x')
    
    # ADF p-values
    ax2 = axes[0, 1]
    colors = ['#06A77D' if x < 0.05 else '#D62828' for x in results_df['ADF_pval']]
    bars = ax2.barh(results_df['Series'], results_df['ADF_pval'], color=colors, alpha=0.7, edgecolor='black')
    ax2.axvline(x=0.05, color='black', linestyle='--', linewidth=2, label='Significance Level (0.05)')
    ax2.set_xlabel('p-value', fontsize=11)
    ax2.set_title('ADF Test p-values\n(Lower = More significant)', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='x')
    
    # KPSS Test Statistics
    ax3 = axes[1, 0]
    colors = ['#06A77D' if x == 'Stationary' else '#D62828' 
              for x in results_df['KPSS_conclusion']]
    ax3.barh(results_df['Series'], results_df['KPSS_stat'], color=colors, alpha=0.7, edgecolor='black')
    ax3.axvline(x=0.146, color='black', linestyle='--', linewidth=2, label='5% Critical Value')
    ax3.set_xlabel('KPSS Statistic', fontsize=11)
    ax3.set_title('KPSS Test Statistics\n(Lower = More stationary)', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Summary table
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')
    
    table_data = []
    for _, row in results_df.iterrows():
        table_data.append([
            row['Series'],
            row['ADF_conclusion'],
            row['KPSS_conclusion']
        ])
    
    table = ax4.table(cellText=table_data,
                     colLabels=['Series', 'ADF', 'KPSS'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color code the table cells
    for i in range(1, len(table_data) + 1):
        if table_data[i-1][1] == 'Stationary':
            table[(i, 1)].set_facecolor('#06A77D')
        else:
            table[(i, 1)].set_facecolor('#D62828')
        
        if table_data[i-1][2] == 'Stationary':
            table[(i, 2)].set_facecolor('#06A77D')
        else:
            table[(i, 2)].set_facecolor('#D62828')
    
    ax4.set_title('Stationarity Test Summary', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig

def main():
    """Main execution function"""
    print("="*60)
    print("MSBD5006: Stationarity Analysis")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Perform comprehensive stationarity analysis
    results_df = comprehensive_stationarity_analysis(df)
    
    # Save results
    results_df.to_csv('output/stationarity_test_results.csv', index=False)
    print("\n[OK] Saved: output/stationarity_test_results.csv")
    
    # Create summary plot
    fig = create_summary_plot(results_df)
    fig.savefig('output/figures/10_stationarity_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: output/figures/10_stationarity_summary.png")
    
    # Print summary
    print("\n" + "="*60)
    print("STATIONARITY ANALYSIS SUMMARY")
    print("="*60)
    print(results_df.to_string(index=False))
    print("="*60)
    
    print("\n" + "="*60)
    print("KEY FINDINGS:")
    print("="*60)
    print("1. Price Level: Typically non-stationary (has trend/unit root)")
    print("2. Returns/Log Returns: Typically stationary (suitable for modeling)")
    print("3. First Difference: Makes non-stationary series stationary")
    print("\nFor ARIMA modeling, we will use:")
    print("- Returns/Log Returns (already stationary)")
    print("- Or differenced price series")
    print("="*60)
    
    print("\n" + "="*60)
    print("STATIONARITY ANALYSIS COMPLETED!")
    print("="*60)
    print("\nGenerated Files:")
    print("- 4 stationarity plots in output/figures/")
    print("- stationarity_test_results.csv")
    print("\nNext Step: Run scripts/03_correlation.py")

if __name__ == "__main__":
    main()


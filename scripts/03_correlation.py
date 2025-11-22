"""
Script 03: Serial Correlation Analysis

This script performs serial correlation analysis including ACF, PACF plots,
and Ljung-Box tests to identify correlation patterns in the time series.


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import pearsonr
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

def plot_acf_pacf(series, series_name, lags=40):
    """Plot ACF and PACF for a given series"""
    print(f"\nGenerating ACF and PACF plots for {series_name}...")
    
    # Remove NaN values
    series_clean = series.dropna()
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # ACF plot
    plot_acf(series_clean, lags=lags, ax=axes[0], alpha=0.05)
    axes[0].set_title(f'Autocorrelation Function (ACF): {series_name}', 
                     fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Lag', fontsize=12)
    axes[0].set_ylabel('Autocorrelation', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # PACF plot
    plot_pacf(series_clean, lags=lags, ax=axes[1], alpha=0.05, method='ywm')
    axes[1].set_title(f'Partial Autocorrelation Function (PACF): {series_name}', 
                     fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Lag', fontsize=12)
    axes[1].set_ylabel('Partial Autocorrelation', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def ljung_box_test(series, lags=20, series_name="Series"):
    """Perform Ljung-Box test for serial correlation"""
    print(f"\n{'='*60}")
    print(f"LJUNG-BOX TEST: {series_name}")
    print(f"{'='*60}")
    
    # Remove NaN values
    series_clean = series.dropna()
    
    # Perform Ljung-Box test
    lb_result = acorr_ljungbox(series_clean, lags=lags, return_df=True)
    
    print(f"\nTesting for serial correlation up to lag {lags}:")
    print(f"\nFirst 10 lags:")
    print(lb_result.head(10).to_string())
    
    # Count significant lags (p < 0.05)
    significant_lags = (lb_result['lb_pvalue'] < 0.05).sum()
    print(f"\n[RESULT] Significant lags (p < 0.05): {significant_lags} out of {lags}")
    
    if significant_lags > 0:
        print(f"[CONCLUSION] Evidence of serial correlation in {series_name}")
    else:
        print(f"[CONCLUSION] No strong evidence of serial correlation in {series_name}")
    
    return lb_result

def plot_ljung_box_results(lb_result, series_name):
    """Plot Ljung-Box test results"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot test statistics
    axes[0].bar(lb_result.index, lb_result['lb_stat'], color='#2E86AB', alpha=0.7, edgecolor='black')
    axes[0].set_title(f'Ljung-Box Test Statistics: {series_name}', 
                     fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Lag', fontsize=12)
    axes[0].set_ylabel('Test Statistic', fontsize=12)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Plot p-values
    axes[1].bar(lb_result.index, lb_result['lb_pvalue'], color='#F18F01', alpha=0.7, edgecolor='black')
    axes[1].axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='Significance Level (0.05)')
    axes[1].set_title(f'Ljung-Box Test p-values: {series_name}', 
                     fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Lag', fontsize=12)
    axes[1].set_ylabel('p-value', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    return fig

def analyze_squared_returns(df):
    """Analyze squared returns to test for ARCH effects"""
    print(f"\n{'='*60}")
    print("ANALYZING SQUARED RETURNS (ARCH Effects)")
    print(f"{'='*60}")
    
    # Calculate squared returns
    df['Returns_Squared'] = df['Returns'] ** 2
    
    # Test for serial correlation in squared returns
    lb_squared = ljung_box_test(df['Returns_Squared'], lags=20, 
                                 series_name="Squared Returns")
    
    return df, lb_squared

def calculate_lag_correlations(series, max_lag=10):
    """Calculate correlations at different lags"""
    print(f"\n{'='*60}")
    print("LAG CORRELATIONS")
    print(f"{'='*60}")
    
    series_clean = series.dropna()
    
    correlations = []
    for lag in range(1, max_lag + 1):
        series_lagged = series_clean.shift(lag)
        valid_idx = series_clean.index.intersection(series_lagged.dropna().index)
        
        if len(valid_idx) > 0:
            corr, p_value = pearsonr(series_clean[valid_idx], series_lagged[valid_idx])
            correlations.append({
                'Lag': lag,
                'Correlation': corr,
                'p-value': p_value,
                'Significant': 'Yes' if p_value < 0.05 else 'No'
            })
    
    corr_df = pd.DataFrame(correlations)
    print(corr_df.to_string(index=False))
    
    return corr_df

def create_correlation_heatmap(df, lags=20):
    """Create a correlation heatmap for lagged values"""
    print(f"\nCreating correlation heatmap...")
    
    # Prepare lagged data
    returns = df['Returns'].dropna()
    lagged_data = pd.DataFrame()
    
    for lag in range(0, lags + 1):
        lagged_data[f'Lag_{lag}'] = returns.shift(lag)
    
    # Calculate correlation matrix
    corr_matrix = lagged_data.corr()
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title(f'Correlation Heatmap: Returns at Different Lags', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

def comprehensive_correlation_analysis(df):
    """Perform comprehensive correlation analysis"""
    
    results = {}
    
    # 1. Returns ACF/PACF
    print("\n" + "="*60)
    print("ANALYSIS 1: RETURNS")
    print("="*60)
    
    fig1 = plot_acf_pacf(df['Returns'], 'Daily Returns', lags=40)
    fig1.savefig('output/figures/11_acf_pacf_returns.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: output/figures/11_acf_pacf_returns.png")
    
    lb_returns = ljung_box_test(df['Returns'], lags=20, series_name="Daily Returns")
    results['returns_lb'] = lb_returns
    
    fig2 = plot_ljung_box_results(lb_returns, 'Daily Returns')
    fig2.savefig('output/figures/12_ljungbox_returns.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: output/figures/12_ljungbox_returns.png")
    
    # 2. Log Returns ACF/PACF
    print("\n" + "="*60)
    print("ANALYSIS 2: LOG RETURNS")
    print("="*60)
    
    fig3 = plot_acf_pacf(df['Log_Returns'], 'Log Returns', lags=40)
    fig3.savefig('output/figures/13_acf_pacf_log_returns.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: output/figures/13_acf_pacf_log_returns.png")
    
    lb_log_returns = ljung_box_test(df['Log_Returns'], lags=20, series_name="Log Returns")
    results['log_returns_lb'] = lb_log_returns
    
    # 3. Squared Returns (ARCH effects)
    df, lb_squared = analyze_squared_returns(df)
    results['squared_returns_lb'] = lb_squared
    
    fig4 = plot_acf_pacf(df['Returns_Squared'], 'Squared Returns', lags=40)
    fig4.savefig('output/figures/14_acf_pacf_squared_returns.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: output/figures/14_acf_pacf_squared_returns.png")
    
    fig5 = plot_ljung_box_results(lb_squared, 'Squared Returns')
    fig5.savefig('output/figures/15_ljungbox_squared_returns.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: output/figures/15_ljungbox_squared_returns.png")
    
    # 4. Lag correlations
    corr_df = calculate_lag_correlations(df['Returns'], max_lag=10)
    results['lag_correlations'] = corr_df
    
    # 5. Correlation heatmap
    fig6 = create_correlation_heatmap(df, lags=20)
    fig6.savefig('output/figures/16_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: output/figures/16_correlation_heatmap.png")
    
    return results

def save_results(results):
    """Save all correlation analysis results"""
    
    # Save Ljung-Box results
    results['returns_lb'].to_csv('output/ljungbox_returns.csv')
    print("\n[OK] Saved: output/ljungbox_returns.csv")
    
    results['log_returns_lb'].to_csv('output/ljungbox_log_returns.csv')
    print("[OK] Saved: output/ljungbox_log_returns.csv")
    
    results['squared_returns_lb'].to_csv('output/ljungbox_squared_returns.csv')
    print("[OK] Saved: output/ljungbox_squared_returns.csv")
    
    # Save lag correlations
    results['lag_correlations'].to_csv('output/lag_correlations.csv', index=False)
    print("[OK] Saved: output/lag_correlations.csv")

def main():
    """Main execution function"""
    print("="*60)
    print("MSBD5006: Serial Correlation Analysis")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Perform comprehensive correlation analysis
    results = comprehensive_correlation_analysis(df)
    
    # Save results
    save_results(results)
    
    # Summary
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)
    
    returns_sig = (results['returns_lb']['lb_pvalue'] < 0.05).sum()
    squared_sig = (results['squared_returns_lb']['lb_pvalue'] < 0.05).sum()
    
    print(f"\n1. Returns Serial Correlation:")
    print(f"   - Significant lags: {returns_sig}/20")
    if returns_sig > 0:
        print(f"   - Evidence of serial correlation suggests AR/MA components")
    
    print(f"\n2. Squared Returns (ARCH Effects):")
    print(f"   - Significant lags: {squared_sig}/20")
    if squared_sig > 5:
        print(f"   - Strong evidence of ARCH effects - GARCH modeling recommended")
    elif squared_sig > 0:
        print(f"   - Some evidence of ARCH effects - consider GARCH modeling")
    
    print(f"\n3. Model Implications:")
    print(f"   - ACF/PACF patterns inform ARIMA(p,d,q) order selection")
    print(f"   - Significant squared return correlations suggest volatility clustering")
    
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS COMPLETED!")
    print("="*60)
    print("\nGenerated Files:")
    print("- 6 correlation plots in output/figures/")
    print("- 4 CSV files with correlation test results")
    print("\nNext Step: Run scripts/04_arima.py")

if __name__ == "__main__":
    main()


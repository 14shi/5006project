"""

Script 01: Exploratory Data Analysis (EDA)

This script performs comprehensive exploratory data analysis on the HSI data
including visualization, descriptive statistics, and return analysis.


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directory for figures
os.makedirs('output/figures', exist_ok=True)

def load_data(filepath='data/hsi_raw.csv'):
    """Load HSI data from CSV file"""
    print("Loading HSI data...")
    df = pd.read_csv(filepath, index_col=0, parse_dates=True, header=[0, 1])
    
    # Flatten column names if multi-level
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    print(f"Data loaded: {len(df)} observations")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    return df

def calculate_returns(df):
    """Calculate daily returns and log returns"""
    print("\nCalculating returns...")
    
    # Simple returns
    df['Returns'] = df['Close'].pct_change()
    
    # Log returns
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    print(f"Returns calculated. Missing values: {df['Returns'].isna().sum()}")
    
    return df

def plot_price_series(df):
    """Plot the closing price time series"""
    print("\nGenerating price series plot...")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df.index, df['Close'], linewidth=1.5, color='#2E86AB')
    ax.set_title('Hang Seng Index: Closing Price (2015-2025)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Closing Price (HKD)', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/figures/01_price_series.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: output/figures/01_price_series.png")

def plot_returns_series(df):
    """Plot returns time series"""
    print("\nGenerating returns series plot...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Simple returns
    ax1.plot(df.index, df['Returns'], linewidth=0.8, color='#A23B72', alpha=0.7)
    ax1.set_title('Daily Returns', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Returns', fontsize=12)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax1.grid(True, alpha=0.3)
    
    # Log returns
    ax2.plot(df.index, df['Log_Returns'], linewidth=0.8, color='#F18F01', alpha=0.7)
    ax2.set_title('Log Returns', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Log Returns', fontsize=12)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/figures/02_returns_series.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: output/figures/02_returns_series.png")

def plot_distribution(df):
    """Plot distribution of returns"""
    print("\nGenerating distribution plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Remove NaN values for plotting
    returns = df['Returns'].dropna()
    log_returns = df['Log_Returns'].dropna()
    
    # Histogram with KDE for returns
    axes[0, 0].hist(returns, bins=50, density=True, alpha=0.7, color='#2E86AB', edgecolor='black')
    axes[0, 0].set_title('Distribution of Daily Returns', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Returns', fontsize=10)
    axes[0, 0].set_ylabel('Density', fontsize=10)
    
    # Add normal distribution overlay
    mu, sigma = returns.mean(), returns.std()
    x = np.linspace(returns.min(), returns.max(), 100)
    axes[0, 0].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal Distribution')
    axes[0, 0].legend()
    
    # Q-Q plot for returns
    stats.probplot(returns, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot: Daily Returns', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Histogram with KDE for log returns
    axes[1, 0].hist(log_returns, bins=50, density=True, alpha=0.7, color='#F18F01', edgecolor='black')
    axes[1, 0].set_title('Distribution of Log Returns', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Log Returns', fontsize=10)
    axes[1, 0].set_ylabel('Density', fontsize=10)
    
    # Add normal distribution overlay
    mu_log, sigma_log = log_returns.mean(), log_returns.std()
    x_log = np.linspace(log_returns.min(), log_returns.max(), 100)
    axes[1, 0].plot(x_log, stats.norm.pdf(x_log, mu_log, sigma_log), 'r-', linewidth=2, label='Normal Distribution')
    axes[1, 0].legend()
    
    # Q-Q plot for log returns
    stats.probplot(log_returns, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot: Log Returns', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/figures/03_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: output/figures/03_distribution.png")

def plot_volume_analysis(df):
    """Plot volume analysis"""
    print("\nGenerating volume analysis plot...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Volume over time
    ax1.bar(df.index, df['Volume'], width=1, color='#06A77D', alpha=0.6)
    ax1.set_title('Trading Volume Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Volume', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Volume vs Returns scatter plot
    returns_clean = df['Returns'].dropna()
    volume_clean = df.loc[returns_clean.index, 'Volume']
    ax2.scatter(returns_clean, volume_clean, alpha=0.3, color='#2E86AB', s=10)
    ax2.set_title('Volume vs Returns', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Returns', fontsize=12)
    ax2.set_ylabel('Volume', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    corr = returns_clean.corr(volume_clean)
    ax2.text(0.05, 0.95, f'Correlation: {corr:.4f}', 
             transform=ax2.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('output/figures/04_volume_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: output/figures/04_volume_analysis.png")

def descriptive_statistics(df):
    """Calculate and save descriptive statistics"""
    print("\nCalculating descriptive statistics...")
    
    returns = df['Returns'].dropna()
    log_returns = df['Log_Returns'].dropna()
    
    stats_dict = {
        'Metric': [
            'Count', 'Mean', 'Std Dev', 'Min', '25%', 'Median', '75%', 'Max',
            'Skewness', 'Kurtosis', 'Jarque-Bera Stat', 'Jarque-Bera p-value'
        ],
        'Returns': [
            len(returns),
            returns.mean(),
            returns.std(),
            returns.min(),
            returns.quantile(0.25),
            returns.median(),
            returns.quantile(0.75),
            returns.max(),
            returns.skew(),
            returns.kurtosis(),
            stats.jarque_bera(returns)[0],
            stats.jarque_bera(returns)[1]
        ],
        'Log Returns': [
            len(log_returns),
            log_returns.mean(),
            log_returns.std(),
            log_returns.min(),
            log_returns.quantile(0.25),
            log_returns.median(),
            log_returns.quantile(0.75),
            log_returns.max(),
            log_returns.skew(),
            log_returns.kurtosis(),
            stats.jarque_bera(log_returns)[0],
            stats.jarque_bera(log_returns)[1]
        ]
    }
    
    stats_df = pd.DataFrame(stats_dict)
    
    # Save to CSV
    stats_df.to_csv('output/descriptive_statistics.csv', index=False)
    print("[OK] Saved: output/descriptive_statistics.csv")
    
    # Print to console
    print("\n" + "="*60)
    print("DESCRIPTIVE STATISTICS")
    print("="*60)
    print(stats_df.to_string(index=False))
    print("="*60)
    
    # Interpretation
    print("\nKEY FINDINGS:")
    if abs(returns.skew()) > 0.5:
        print(f"- Returns show {'negative' if returns.skew() < 0 else 'positive'} skewness ({returns.skew():.4f})")
    if returns.kurtosis() > 3:
        print(f"- Returns exhibit excess kurtosis ({returns.kurtosis():.4f}), indicating fat tails")
    if stats.jarque_bera(returns)[1] < 0.05:
        print("- Jarque-Bera test rejects normality (p < 0.05)")
    
    return stats_df

def plot_rolling_statistics(df, window=30):
    """Plot rolling mean and standard deviation"""
    print(f"\nGenerating rolling statistics plot (window={window})...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Calculate rolling statistics for returns
    returns = df['Returns'].dropna()
    rolling_mean = returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std()
    
    # Rolling mean
    ax1.plot(returns.index, returns, linewidth=0.5, alpha=0.5, label='Returns', color='lightgray')
    ax1.plot(rolling_mean.index, rolling_mean, linewidth=2, label=f'{window}-day Rolling Mean', color='#2E86AB')
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax1.set_title(f'Rolling Mean of Returns (Window={window} days)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Returns', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Rolling standard deviation
    ax2.plot(rolling_std.index, rolling_std, linewidth=2, color='#A23B72', label=f'{window}-day Rolling Std Dev')
    ax2.set_title(f'Rolling Standard Deviation of Returns (Window={window} days)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Standard Deviation', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/figures/05_rolling_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: output/figures/05_rolling_statistics.png")

def yearly_analysis(df):
    """Analyze returns by year"""
    print("\nPerforming yearly analysis...")
    
    df_year = df.copy()
    df_year['Year'] = df_year.index.year
    
    yearly_stats = df_year.groupby('Year').agg({
        'Returns': ['count', 'mean', 'std', 'min', 'max'],
        'Close': ['first', 'last']
    })
    
    # Calculate annual return
    yearly_stats['Annual_Return'] = (yearly_stats['Close']['last'] / yearly_stats['Close']['first'] - 1) * 100
    
    # Save to CSV
    yearly_stats.to_csv('output/yearly_analysis.csv')
    print("[OK] Saved: output/yearly_analysis.csv")
    
    print("\n" + "="*60)
    print("YEARLY ANALYSIS")
    print("="*60)
    print(yearly_stats)
    print("="*60)
    
    # Plot yearly returns
    fig, ax = plt.subplots(figsize=(12, 6))
    years = yearly_stats.index
    annual_returns = yearly_stats['Annual_Return'].values
    colors = ['#06A77D' if x > 0 else '#D62828' for x in annual_returns]
    
    ax.bar(years, annual_returns, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_title('Annual Returns by Year', fontsize=14, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Annual Return (%)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (year, ret) in enumerate(zip(years, annual_returns)):
        ax.text(year, ret + (1 if ret > 0 else -1), f'{ret:.1f}%', 
                ha='center', va='bottom' if ret > 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('output/figures/06_yearly_returns.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: output/figures/06_yearly_returns.png")

def main():
    """Main execution function"""
    print("="*60)
    print("MSBD5006: Exploratory Data Analysis")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Calculate returns
    df = calculate_returns(df)
    
    # Generate visualizations
    plot_price_series(df)
    plot_returns_series(df)
    plot_distribution(df)
    plot_volume_analysis(df)
    plot_rolling_statistics(df, window=30)
    
    # Calculate statistics
    stats_df = descriptive_statistics(df)
    
    # Yearly analysis
    yearly_analysis(df)
    
    # Save processed data with returns
    df.to_csv('data/hsi_with_returns.csv')
    print("\n[OK] Saved processed data: data/hsi_with_returns.csv")
    
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS COMPLETED!")
    print("="*60)
    print("\nGenerated Files:")
    print("- 6 figures in output/figures/")
    print("- descriptive_statistics.csv")
    print("- yearly_analysis.csv")
    print("- hsi_with_returns.csv")
    print("\nNext Step: Run scripts/02_stationarity.py")

if __name__ == "__main__":
    main()


"""

Script 05: GARCH Volatility Modeling

This script performs GARCH modeling to capture volatility clustering
and time-varying volatility in HSI returns.


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
from scipy import stats
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

def arch_lm_test(returns, lags=10):
    """Perform ARCH LM test for heteroskedasticity"""
    print(f"\n{'='*60}")
    print("ENGLE'S ARCH TEST (LM Test)")
    print(f"{'='*60}")
    
    from statsmodels.stats.diagnostic import het_arch
    
    returns_clean = returns.dropna()
    
    # Perform ARCH LM test
    lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(returns_clean, nlags=lags)
    
    print(f"Testing for ARCH effects up to lag {lags}:")
    print(f"\nLagrange Multiplier (LM) Test:")
    print(f"  LM Statistic: {lm_stat:.4f}")
    print(f"  p-value: {lm_pvalue:.6f}")
    print(f"\nF-Test:")
    print(f"  F-Statistic: {f_stat:.4f}")
    print(f"  p-value: {f_pvalue:.6f}")
    
    if lm_pvalue < 0.05:
        print(f"\n[RESULT] Reject null hypothesis (p < 0.05)")
        print("[CONCLUSION] Strong evidence of ARCH effects - GARCH modeling appropriate")
    else:
        print(f"\n[RESULT] Fail to reject null hypothesis (p >= 0.05)")
        print("[CONCLUSION] No strong evidence of ARCH effects")
    
    return lm_stat, lm_pvalue

def fit_garch_model(returns, p=1, q=1, mean='Constant', vol='GARCH', dist='normal'):
    """
    Fit GARCH model
    
    Parameters:
    -----------
    returns : Series
        Returns data
    p : int
        GARCH order
    q : int
        ARCH order
    mean : str
        Mean model ('Constant', 'Zero', 'AR')
    vol : str
        Volatility model ('GARCH', 'EGARCH', 'TGARCH')
    dist : str
        Distribution ('normal', 't', 'skewt')
    """
    print(f"\n{'='*60}")
    print(f"FITTING {vol}({p},{q}) MODEL")
    print(f"{'='*60}")
    
    returns_clean = returns.dropna() * 100  # Scale returns by 100 for better convergence
    
    # Create model
    if vol == 'GARCH':
        model = arch_model(returns_clean, mean=mean, vol='GARCH', p=p, q=q, dist=dist)
    elif vol == 'EGARCH':
        model = arch_model(returns_clean, mean=mean, vol='EGARCH', p=p, q=q, dist=dist)
    elif vol == 'TGARCH':
        model = arch_model(returns_clean, mean=mean, vol='GARCH', p=p, o=1, q=q, dist=dist)
    else:
        raise ValueError(f"Unknown volatility model: {vol}")
    
    # Fit model
    fitted_model = model.fit(disp='off')
    
    print(fitted_model.summary())
    
    return fitted_model

def plot_conditional_volatility(fitted_model, model_name="GARCH"):
    """Plot conditional volatility from fitted GARCH model"""
    print(f"\nGenerating conditional volatility plot for {model_name}...")
    
    # Get conditional volatility (scaled back)
    cond_vol = fitted_model.conditional_volatility / 100
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot returns with conditional volatility
    returns = fitted_model.resid / 100
    axes[0].plot(returns.index, returns, linewidth=0.5, alpha=0.7, label='Returns', color='lightgray')
    axes[0].plot(cond_vol.index, cond_vol, linewidth=2, label='Conditional Volatility', color='#D62828')
    axes[0].plot(cond_vol.index, -cond_vol, linewidth=2, color='#D62828')
    axes[0].set_title(f'{model_name}: Returns and Conditional Volatility', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Date', fontsize=12)
    axes[0].set_ylabel('Returns / Volatility', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot conditional volatility only
    axes[1].plot(cond_vol.index, cond_vol, linewidth=1.5, color='#2E86AB')
    axes[1].set_title(f'{model_name}: Conditional Volatility Over Time', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].set_ylabel('Conditional Volatility', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # Highlight high volatility periods
    high_vol_threshold = cond_vol.quantile(0.90)
    high_vol_periods = cond_vol[cond_vol > high_vol_threshold]
    axes[1].scatter(high_vol_periods.index, high_vol_periods, color='red', s=10, alpha=0.5, 
                    label=f'High Volatility (>90th percentile)')
    axes[1].legend()
    
    plt.tight_layout()
    return fig

def plot_standardized_residuals(fitted_model, model_name="GARCH"):
    """Plot standardized residuals diagnostics"""
    print(f"\nGenerating standardized residuals diagnostics for {model_name}...")
    
    # Get standardized residuals
    std_resid = fitted_model.std_resid
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Standardized residuals over time
    axes[0, 0].plot(std_resid, linewidth=0.8, color='#2E86AB')
    axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[0, 0].set_title(f'{model_name}: Standardized Residuals', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Time', fontsize=10)
    axes[0, 0].set_ylabel('Standardized Residuals', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Histogram
    axes[0, 1].hist(std_resid, bins=50, density=True, alpha=0.7, color='#2E86AB', edgecolor='black')
    mu, sigma = std_resid.mean(), std_resid.std()
    x = np.linspace(std_resid.min(), std_resid.max(), 100)
    axes[0, 1].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal')
    axes[0, 1].set_title(f'{model_name}: Residuals Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Standardized Residuals', fontsize=10)
    axes[0, 1].set_ylabel('Density', fontsize=10)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Q-Q plot
    stats.probplot(std_resid, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title(f'{model_name}: Q-Q Plot', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. ACF of squared standardized residuals
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(std_resid**2, lags=40, ax=axes[1, 1], alpha=0.05)
    axes[1, 1].set_title(f'{model_name}: ACF of Squared Std. Residuals', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def compare_garch_models(returns):
    """Compare different GARCH specifications"""
    print(f"\n{'='*60}")
    print("COMPARING GARCH MODEL SPECIFICATIONS")
    print(f"{'='*60}")
    
    models = {}
    comparison = []
    
    # GARCH(1,1) - standard
    try:
        print("\n[1/5] Fitting GARCH(1,1)...")
        garch_11 = fit_garch_model(returns, p=1, q=1, vol='GARCH')
        models['GARCH(1,1)'] = garch_11
        comparison.append({
            'Model': 'GARCH(1,1)',
            'Log Likelihood': garch_11.loglikelihood,
            'AIC': garch_11.aic,
            'BIC': garch_11.bic,
            'Parameters': len(garch_11.params)
        })
    except Exception as e:
        print(f"Failed to fit GARCH(1,1): {e}")
    
    # GARCH(1,2)
    try:
        print("\n[2/5] Fitting GARCH(1,2)...")
        garch_12 = fit_garch_model(returns, p=1, q=2, vol='GARCH')
        models['GARCH(1,2)'] = garch_12
        comparison.append({
            'Model': 'GARCH(1,2)',
            'Log Likelihood': garch_12.loglikelihood,
            'AIC': garch_12.aic,
            'BIC': garch_12.bic,
            'Parameters': len(garch_12.params)
        })
    except Exception as e:
        print(f"Failed to fit GARCH(1,2): {e}")
    
    # GARCH(2,1)
    try:
        print("\n[3/5] Fitting GARCH(2,1)...")
        garch_21 = fit_garch_model(returns, p=2, q=1, vol='GARCH')
        models['GARCH(2,1)'] = garch_21
        comparison.append({
            'Model': 'GARCH(2,1)',
            'Log Likelihood': garch_21.loglikelihood,
            'AIC': garch_21.aic,
            'BIC': garch_21.bic,
            'Parameters': len(garch_21.params)
        })
    except Exception as e:
        print(f"Failed to fit GARCH(2,1): {e}")
    
    # EGARCH(1,1) - captures asymmetry
    try:
        print("\n[4/5] Fitting EGARCH(1,1)...")
        egarch_11 = fit_garch_model(returns, p=1, q=1, vol='EGARCH')
        models['EGARCH(1,1)'] = egarch_11
        comparison.append({
            'Model': 'EGARCH(1,1)',
            'Log Likelihood': egarch_11.loglikelihood,
            'AIC': egarch_11.aic,
            'BIC': egarch_11.bic,
            'Parameters': len(egarch_11.params)
        })
    except Exception as e:
        print(f"Failed to fit EGARCH(1,1): {e}")
    
    # GARCH(1,1) with Student's t distribution
    try:
        print("\n[5/5] Fitting GARCH(1,1) with t-distribution...")
        garch_11_t = fit_garch_model(returns, p=1, q=1, vol='GARCH', dist='t')
        models['GARCH(1,1)-t'] = garch_11_t
        comparison.append({
            'Model': 'GARCH(1,1)-t',
            'Log Likelihood': garch_11_t.loglikelihood,
            'AIC': garch_11_t.aic,
            'BIC': garch_11_t.bic,
            'Parameters': len(garch_11_t.params)
        })
    except Exception as e:
        print(f"Failed to fit GARCH(1,1)-t: {e}")
    
    comparison_df = pd.DataFrame(comparison)
    comparison_df = comparison_df.sort_values('AIC')
    
    print(f"\n{'='*60}")
    print("MODEL COMPARISON RESULTS")
    print(f"{'='*60}")
    print(comparison_df.to_string(index=False))
    
    best_model_name = comparison_df.iloc[0]['Model']
    print(f"\n[BEST MODEL by AIC] {best_model_name}")
    
    return models, comparison_df, best_model_name

def plot_model_comparison(comparison_df):
    """Plot GARCH model comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # AIC comparison
    axes[0].barh(comparison_df['Model'], comparison_df['AIC'], 
                 color='#2E86AB', alpha=0.7, edgecolor='black')
    axes[0].set_title('GARCH Model Comparison: AIC', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('AIC (lower is better)', fontsize=12)
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # BIC comparison
    axes[1].barh(comparison_df['Model'], comparison_df['BIC'], 
                 color='#F18F01', alpha=0.7, edgecolor='black')
    axes[1].set_title('GARCH Model Comparison: BIC', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('BIC (lower is better)', fontsize=12)
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig

def analyze_volatility_persistence(fitted_model):
    """Analyze volatility persistence from GARCH model"""
    print(f"\n{'='*60}")
    print("VOLATILITY PERSISTENCE ANALYSIS")
    print(f"{'='*60}")
    
    params = fitted_model.params
    
    # For GARCH(1,1), persistence = alpha + beta
    if 'alpha[1]' in params and 'beta[1]' in params:
        alpha = params['alpha[1]']
        beta = params['beta[1]']
        persistence = alpha + beta
        
        print(f"GARCH(1,1) Parameters:")
        print(f"  Alpha (ARCH): {alpha:.6f}")
        print(f"  Beta (GARCH): {beta:.6f}")
        print(f"  Persistence (alpha + beta): {persistence:.6f}")
        
        if persistence >= 1:
            print("\n[WARNING] Persistence >= 1 suggests integrated GARCH (IGARCH)")
        elif persistence > 0.95:
            print("\n[RESULT] Very high persistence - volatility shocks decay slowly")
        elif persistence > 0.80:
            print("\n[RESULT] High persistence - substantial volatility clustering")
        else:
            print("\n[RESULT] Moderate persistence")
        
        # Half-life of volatility shock
        if persistence < 1:
            half_life = np.log(0.5) / np.log(persistence)
            print(f"\nHalf-life of volatility shock: {half_life:.2f} days")
        
        return persistence
    else:
        print("Cannot calculate persistence for this model specification")
        return None

def main():
    """Main execution function"""
    print("="*60)
    print("MSBD5006: GARCH Volatility Modeling")
    print("="*60)
    
    # Load data
    df = load_data()
    returns = df['Returns'].dropna()
    
    # Test for ARCH effects
    lm_stat, lm_pvalue = arch_lm_test(returns, lags=10)
    
    # Compare different GARCH specifications
    models, comparison_df, best_model_name = compare_garch_models(returns)
    
    # Save comparison results
    comparison_df.to_csv('output/garch_model_comparison.csv', index=False)
    print("\n[OK] Saved: output/garch_model_comparison.csv")
    
    # Plot model comparison
    fig1 = plot_model_comparison(comparison_df)
    fig1.savefig('output/figures/19_garch_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: output/figures/19_garch_model_comparison.png")
    
    # Get best model
    best_model = models[best_model_name]
    
    # Plot conditional volatility for best model
    fig2 = plot_conditional_volatility(best_model, best_model_name)
    fig2.savefig('output/figures/20_conditional_volatility.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: output/figures/20_conditional_volatility.png")
    
    # Plot standardized residuals diagnostics
    fig3 = plot_standardized_residuals(best_model, best_model_name)
    fig3.savefig('output/figures/21_garch_residuals_diagnostics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: output/figures/21_garch_residuals_diagnostics.png")
    
    # Analyze volatility persistence
    persistence = analyze_volatility_persistence(best_model)
    
    # Save best model
    import pickle
    with open('output/best_garch_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print("\n[OK] Saved: output/best_garch_model.pkl")
    
    # Save conditional volatility
    cond_vol_df = pd.DataFrame({
        'Date': best_model.conditional_volatility.index,
        'Conditional_Volatility': best_model.conditional_volatility.values / 100
    })
    cond_vol_df.to_csv('output/conditional_volatility.csv', index=False)
    print("[OK] Saved: output/conditional_volatility.csv")
    
    print("\n" + "="*60)
    print("GARCH MODELING COMPLETED!")
    print("="*60)
    print("\nGenerated Files:")
    print("- GARCH model comparison")
    print("- Conditional volatility plots")
    print("- Residual diagnostics")
    print("- Best model saved as pickle file")
    print(f"\nBest Model: {best_model_name}")
    if persistence:
        print(f"Volatility Persistence: {persistence:.4f}")
    print("\nNext Step: Run scripts/06_forecasting.py")

if __name__ == "__main__":
    main()


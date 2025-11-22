"""

Script 04: ARIMA Modeling

This script performs ARIMA model selection, fitting, and diagnostics
for the HSI returns data.


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
import itertools
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

def grid_search_arima(series, p_range=range(0, 4), d_range=range(0, 2), q_range=range(0, 4)):
    """
    Perform grid search to find optimal ARIMA parameters
    """
    print(f"\n{'='*60}")
    print("ARIMA MODEL SELECTION: Grid Search")
    print(f"{'='*60}")
    print(f"Testing p in {list(p_range)}, d in {list(d_range)}, q in {list(q_range)}")
    
    # Remove NaN values
    series_clean = series.dropna()
    
    results = []
    
    # Generate all combinations of p, d, q
    pdq_combinations = list(itertools.product(p_range, d_range, q_range))
    
    print(f"\nTotal models to test: {len(pdq_combinations)}")
    print("Testing models...")
    
    for i, (p, d, q) in enumerate(pdq_combinations):
        try:
            # Fit ARIMA model
            model = ARIMA(series_clean, order=(p, d, q))
            fitted_model = model.fit()
            
            # Get model information
            aic = fitted_model.aic
            bic = fitted_model.bic
            
            results.append({
                'p': p,
                'd': d,
                'q': q,
                'AIC': aic,
                'BIC': bic,
                'converged': True
            })
            
            if (i + 1) % 10 == 0:
                print(f"  Tested {i + 1}/{len(pdq_combinations)} models...")
                
        except Exception as e:
            # Model failed to converge or fit
            results.append({
                'p': p,
                'd': d,
                'q': q,
                'AIC': np.inf,
                'BIC': np.inf,
                'converged': False
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Filter converged models
    converged_models = results_df[results_df['converged'] == True].copy()
    
    if len(converged_models) == 0:
        print("\n[ERROR] No models converged!")
        return results_df, None, None
    
    # Sort by AIC
    converged_models = converged_models.sort_values('AIC')
    
    print(f"\n[OK] Completed grid search")
    print(f"Converged models: {len(converged_models)}/{len(pdq_combinations)}")
    
    print("\nTop 10 models by AIC:")
    print(converged_models.head(10).to_string(index=False))
    
    # Get best model by AIC
    best_aic_params = converged_models.iloc[0]
    
    # Get best model by BIC
    converged_models_bic = converged_models.sort_values('BIC')
    best_bic_params = converged_models_bic.iloc[0]
    
    print(f"\n[BEST MODEL by AIC] ARIMA({int(best_aic_params['p'])}, {int(best_aic_params['d'])}, {int(best_aic_params['q'])})")
    print(f"  AIC: {best_aic_params['AIC']:.4f}, BIC: {best_aic_params['BIC']:.4f}")
    
    print(f"\n[BEST MODEL by BIC] ARIMA({int(best_bic_params['p'])}, {int(best_bic_params['d'])}, {int(best_bic_params['q'])})")
    print(f"  AIC: {best_bic_params['AIC']:.4f}, BIC: {best_bic_params['BIC']:.4f}")
    
    return results_df, best_aic_params, best_bic_params

def fit_arima_model(series, order):
    """Fit ARIMA model with specified order"""
    print(f"\n{'='*60}")
    print(f"FITTING ARIMA{order} MODEL")
    print(f"{'='*60}")
    
    # Remove NaN values
    series_clean = series.dropna()
    
    # Fit model
    model = ARIMA(series_clean, order=order)
    fitted_model = model.fit()
    
    # Print summary
    print(fitted_model.summary())
    
    return fitted_model

def diagnostic_plots(fitted_model, model_name="ARIMA"):
    """Generate diagnostic plots for fitted ARIMA model"""
    print(f"\nGenerating diagnostic plots for {model_name}...")
    
    # Get residuals
    residuals = fitted_model.resid
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Residuals over time
    axes[0, 0].plot(residuals, linewidth=0.8, color='#2E86AB')
    axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[0, 0].set_title(f'{model_name}: Residuals Over Time', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Time', fontsize=10)
    axes[0, 0].set_ylabel('Residuals', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residuals histogram with normal distribution
    axes[0, 1].hist(residuals, bins=50, density=True, alpha=0.7, color='#2E86AB', edgecolor='black')
    mu, sigma = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    axes[0, 1].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal')
    axes[0, 1].set_title(f'{model_name}: Residuals Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Residuals', fontsize=10)
    axes[0, 1].set_ylabel('Density', fontsize=10)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title(f'{model_name}: Q-Q Plot', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. ACF of residuals
    plot_acf(residuals, lags=40, ax=axes[1, 1], alpha=0.05)
    axes[1, 1].set_title(f'{model_name}: ACF of Residuals', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def residual_diagnostics(fitted_model, model_name="ARIMA"):
    """Perform statistical tests on residuals"""
    print(f"\n{'='*60}")
    print(f"RESIDUAL DIAGNOSTICS: {model_name}")
    print(f"{'='*60}")
    
    residuals = fitted_model.resid
    
    # 1. Ljung-Box test on residuals
    print("\n1. Ljung-Box Test on Residuals:")
    lb_test = acorr_ljungbox(residuals, lags=20, return_df=True)
    print(lb_test.head(10).to_string())
    
    significant_lags = (lb_test['lb_pvalue'] < 0.05).sum()
    print(f"\nSignificant lags (p < 0.05): {significant_lags}/20")
    
    if significant_lags == 0:
        print("[OK] Residuals show no significant autocorrelation")
    else:
        print("[WARNING] Some autocorrelation remains in residuals")
    
    # 2. Normality tests
    print("\n2. Normality Tests:")
    
    # Jarque-Bera test
    jb_stat, jb_pvalue = stats.jarque_bera(residuals)
    print(f"   Jarque-Bera Test: stat={jb_stat:.4f}, p-value={jb_pvalue:.6f}")
    if jb_pvalue < 0.05:
        print("   [RESULT] Residuals are NOT normally distributed")
    else:
        print("   [RESULT] Residuals appear normally distributed")
    
    # Shapiro-Wilk test (for smaller samples)
    if len(residuals) <= 5000:
        sw_stat, sw_pvalue = stats.shapiro(residuals)
        print(f"   Shapiro-Wilk Test: stat={sw_stat:.4f}, p-value={sw_pvalue:.6f}")
    
    # 3. Descriptive statistics of residuals
    print("\n3. Residual Statistics:")
    print(f"   Mean: {residuals.mean():.6f}")
    print(f"   Std Dev: {residuals.std():.6f}")
    print(f"   Skewness: {residuals.skew():.6f}")
    print(f"   Kurtosis: {residuals.kurtosis():.6f}")
    
    return lb_test

def compare_models(models_dict, series):
    """Compare multiple ARIMA models"""
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")
    
    comparison = []
    
    for name, model in models_dict.items():
        comparison.append({
            'Model': name,
            'AIC': model.aic,
            'BIC': model.bic,
            'Log Likelihood': model.llf,
            'Parameters': len(model.params)
        })
    
    comparison_df = pd.DataFrame(comparison)
    print(comparison_df.to_string(index=False))
    
    # Best model by AIC
    best_aic = comparison_df.loc[comparison_df['AIC'].idxmin(), 'Model']
    print(f"\n[BEST MODEL by AIC] {best_aic}")
    
    # Best model by BIC
    best_bic = comparison_df.loc[comparison_df['BIC'].idxmin(), 'Model']
    print(f"[BEST MODEL by BIC] {best_bic}")
    
    return comparison_df

def plot_model_comparison(comparison_df):
    """Plot model comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # AIC comparison
    axes[0].bar(range(len(comparison_df)), comparison_df['AIC'], 
                color='#2E86AB', alpha=0.7, edgecolor='black')
    axes[0].set_xticks(range(len(comparison_df)))
    axes[0].set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
    axes[0].set_title('Model Comparison: AIC', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('AIC', fontsize=12)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # BIC comparison
    axes[1].bar(range(len(comparison_df)), comparison_df['BIC'], 
                color='#F18F01', alpha=0.7, edgecolor='black')
    axes[1].set_xticks(range(len(comparison_df)))
    axes[1].set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
    axes[1].set_title('Model Comparison: BIC', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('BIC', fontsize=12)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

def main():
    """Main execution function"""
    print("="*60)
    print("MSBD5006: ARIMA Modeling")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Use returns for modeling (already stationary)
    returns = df['Returns'].dropna()
    
    # Grid search for best model
    results_df, best_aic_params, best_bic_params = grid_search_arima(
        returns, 
        p_range=range(0, 4), 
        d_range=range(0, 2), 
        q_range=range(0, 4)
    )
    
    # Save grid search results
    results_df.to_csv('output/arima_grid_search_results.csv', index=False)
    print("\n[OK] Saved: output/arima_grid_search_results.csv")
    
    if best_aic_params is None:
        print("\n[ERROR] No suitable ARIMA model found!")
        return
    
    # Fit best model by AIC
    best_order_aic = (int(best_aic_params['p']), int(best_aic_params['d']), int(best_aic_params['q']))
    model_aic = fit_arima_model(returns, best_order_aic)
    
    # Diagnostic plots for best model
    fig1 = diagnostic_plots(model_aic, f"ARIMA{best_order_aic}")
    fig1.savefig('output/figures/17_arima_diagnostics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: output/figures/17_arima_diagnostics.png")
    
    # Residual diagnostics
    lb_results = residual_diagnostics(model_aic, f"ARIMA{best_order_aic}")
    lb_results.to_csv('output/arima_residual_diagnostics.csv')
    print("\n[OK] Saved: output/arima_residual_diagnostics.csv")
    
    # Also fit a few common models for comparison
    models_dict = {
        f'ARIMA{best_order_aic}': model_aic
    }
    
    # Try ARIMA(1,0,0) if not already the best
    if best_order_aic != (1, 0, 0):
        try:
            model_100 = fit_arima_model(returns, (1, 0, 0))
            models_dict['ARIMA(1,0,0)'] = model_100
        except:
            pass
    
    # Try ARIMA(0,0,1) if not already the best
    if best_order_aic != (0, 0, 1):
        try:
            model_001 = fit_arima_model(returns, (0, 0, 1))
            models_dict['ARIMA(0,0,1)'] = model_001
        except:
            pass
    
    # Try ARIMA(1,0,1) if not already the best
    if best_order_aic != (1, 0, 1):
        try:
            model_101 = fit_arima_model(returns, (1, 0, 1))
            models_dict['ARIMA(1,0,1)'] = model_101
        except:
            pass
    
    # Compare models
    if len(models_dict) > 1:
        comparison_df = compare_models(models_dict, returns)
        comparison_df.to_csv('output/arima_model_comparison.csv', index=False)
        print("\n[OK] Saved: output/arima_model_comparison.csv")
        
        fig2 = plot_model_comparison(comparison_df)
        fig2.savefig('output/figures/18_arima_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("[OK] Saved: output/figures/18_arima_model_comparison.png")
    
    # Save best model
    import pickle
    with open('output/best_arima_model.pkl', 'wb') as f:
        pickle.dump(model_aic, f)
    print("\n[OK] Saved: output/best_arima_model.pkl")
    
    print("\n" + "="*60)
    print("ARIMA MODELING COMPLETED!")
    print("="*60)
    print("\nGenerated Files:")
    print("- ARIMA diagnostic plots")
    print("- Grid search results")
    print("- Model comparison")
    print("- Best model saved as pickle file")
    print(f"\nBest Model: ARIMA{best_order_aic}")
    print("\nNext Step: Run scripts/05_garch.py")

if __name__ == "__main__":
    main()


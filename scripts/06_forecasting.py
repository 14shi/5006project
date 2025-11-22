"""
Script 06: Forecasting and Evaluation

This script performs out-of-sample forecasting using fitted ARIMA and GARCH models
and evaluates their performance using various metrics.


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import pickle
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

def split_data(df, test_days=252):
    """Split data into training and testing sets
    
    Args:
        df: Full dataset
        test_days: Number of days to use for testing (default: 252)
    """
    # Use last test_days for testing, rest for training
    train = df.iloc[:-test_days].copy()
    test = df.iloc[-test_days:].copy()
    
    print(f"\nData Split:")
    print(f"  Training: {len(train)} observations ({train.index[0]} to {train.index[-1]})")
    print(f"  Testing:  {len(test)} observations ({test.index[0]} to {test.index[-1]})")
    print(f"  Test period: ~{test_days} days (~{test_days/21:.1f} months)")
    
    return train, test

def calculate_metrics(actual, predicted):
    """Calculate forecast evaluation metrics"""
    # Remove NaN values
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual = actual[mask]
    predicted = predicted[mask]
    
    # Mean Absolute Error
    mae = np.mean(np.abs(actual - predicted))
    
    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((actual - predicted)**2))
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    # Mean Error (Bias)
    me = np.mean(actual - predicted)
    
    # R-squared
    ss_res = np.sum((actual - predicted)**2)
    ss_tot = np.sum((actual - np.mean(actual))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'ME': me,
        'R2': r2,
        'N': len(actual)
    }

def arima_rolling_forecast(train, test, order=(2, 0, 3)):
    """Perform rolling window forecast with ARIMA"""
    print(f"\n{'='*60}")
    print(f"ARIMA{order} ROLLING WINDOW FORECAST")
    print(f"{'='*60}")
    
    returns_train = train['Returns'].dropna()
    returns_test = test['Returns'].dropna()
    
    predictions = []
    history = returns_train.tolist()
    
    print(f"Forecasting {len(returns_test)} steps...")
    print("Progress: ", end='', flush=True)
    
    for i, actual_value in enumerate(returns_test):
        # Fit model on historical data
        model = ARIMA(history, order=order)
        fitted_model = model.fit()
        
        # Forecast one step ahead
        forecast = fitted_model.forecast(steps=1)[0]
        predictions.append(forecast)
        
        # Add actual value to history for next iteration
        history.append(actual_value)
        
        # Progress indicator every 10 steps
        if (i + 1) % 10 == 0:
            print(f"{i + 1}", end='...', flush=True)
        if (i + 1) % 50 == 0:
            print(f" ({(i+1)/len(returns_test)*100:.1f}%)", flush=True)
            print("Progress: ", end='', flush=True)
    
    predictions = np.array(predictions)
    
    # Calculate metrics
    metrics = calculate_metrics(returns_test.values, predictions)
    
    print(f"\n[OK] ARIMA Forecast Completed")
    print(f"Forecast Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")
    
    return predictions, metrics

def garch_forecast(train, test):
    """Perform GARCH volatility forecast"""
    print(f"\n{'='*60}")
    print("GARCH(1,1)-t VOLATILITY FORECAST")
    print(f"{'='*60}")
    
    returns_train = train['Returns'].dropna() * 100
    returns_test = test['Returns'].dropna()
    
    # Fit GARCH model on training data
    print("Fitting GARCH model on training data...")
    model = arch_model(returns_train, mean='Constant', vol='GARCH', p=1, q=1, dist='t')
    fitted_model = model.fit(disp='off')
    
    # Forecast volatility for test period
    print(f"Forecasting volatility for {len(returns_test)} steps...")
    forecast = fitted_model.forecast(horizon=len(returns_test), reindex=False)
    
    # Get forecasted variance (need to take square root for volatility)
    forecasted_variance = forecast.variance.values[-1, :]
    forecasted_volatility = np.sqrt(forecasted_variance) / 100  # Scale back
    
    # Calculate actual realized volatility (squared returns)
    actual_volatility = np.abs(returns_test.values)
    
    # Calculate metrics
    metrics = calculate_metrics(actual_volatility, forecasted_volatility)
    
    print(f"\n[OK] GARCH Forecast Completed")
    print(f"Volatility Forecast Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")
    
    return forecasted_volatility, metrics

def plot_forecast_results(test, predictions, model_name="ARIMA"):
    """Plot forecast results vs actual values"""
    print(f"\nGenerating forecast plot for {model_name}...")
    
    returns_test = test['Returns'].dropna()
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot actual vs predicted
    axes[0].plot(returns_test.index, returns_test.values, linewidth=1.5, 
                label='Actual', color='#2E86AB', alpha=0.7)
    axes[0].plot(returns_test.index, predictions, linewidth=1.5, 
                label='Forecast', color='#D62828', linestyle='--', alpha=0.7)
    axes[0].set_title(f'{model_name}: Actual vs Forecasted Returns', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Date', fontsize=12)
    axes[0].set_ylabel('Returns', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Plot forecast errors
    errors = returns_test.values - predictions
    axes[1].plot(returns_test.index, errors, linewidth=1, color='#F18F01', alpha=0.7)
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[1].fill_between(returns_test.index, errors, 0, alpha=0.3, color='#F18F01')
    axes[1].set_title(f'{model_name}: Forecast Errors', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].set_ylabel('Error (Actual - Forecast)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_volatility_forecast(test, forecasted_vol):
    """Plot volatility forecast"""
    print(f"\nGenerating volatility forecast plot...")
    
    returns_test = test['Returns'].dropna()
    actual_vol = np.abs(returns_test.values)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot returns with forecasted volatility bands
    axes[0].plot(returns_test.index, returns_test.values, linewidth=0.8, 
                label='Returns', color='#2E86AB', alpha=0.6)
    axes[0].plot(returns_test.index, forecasted_vol, linewidth=2, 
                label='Forecasted Volatility', color='#D62828')
    axes[0].plot(returns_test.index, -forecasted_vol, linewidth=2, color='#D62828')
    axes[0].fill_between(returns_test.index, forecasted_vol, -forecasted_vol, 
                         alpha=0.2, color='#D62828')
    axes[0].set_title('GARCH: Returns with Forecasted Volatility Bands', 
                     fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Date', fontsize=12)
    axes[0].set_ylabel('Returns / Volatility', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Compare actual vs forecasted volatility
    axes[1].plot(returns_test.index, actual_vol, linewidth=1.5, 
                label='Actual Volatility (|returns|)', color='#2E86AB', alpha=0.7)
    axes[1].plot(returns_test.index, forecasted_vol, linewidth=1.5, 
                label='Forecasted Volatility', color='#D62828', linestyle='--', alpha=0.7)
    axes[1].set_title('GARCH: Actual vs Forecasted Volatility', 
                     fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].set_ylabel('Volatility', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def directional_accuracy(actual, predicted):
    """Calculate directional accuracy (% of correct sign predictions)"""
    actual_direction = np.sign(actual)
    predicted_direction = np.sign(predicted)
    
    correct = np.sum(actual_direction == predicted_direction)
    total = len(actual)
    accuracy = (correct / total) * 100
    
    return accuracy

def compare_forecast_methods(test, arima_pred, arima_metrics):
    """Compare different forecasting approaches"""
    print(f"\n{'='*60}")
    print("FORECAST METHOD COMPARISON")
    print(f"{'='*60}")
    
    returns_test = test['Returns'].dropna().values
    
    # Naive forecast (random walk - last value)
    naive_pred = np.roll(returns_test, 1)
    naive_pred[0] = 0
    naive_metrics = calculate_metrics(returns_test, naive_pred)
    
    # Mean forecast (historical mean)
    mean_pred = np.full(len(returns_test), np.mean(returns_test))
    mean_metrics = calculate_metrics(returns_test, mean_pred)
    
    # Create comparison table
    comparison = pd.DataFrame({
        'Model': ['ARIMA(2,0,3)', 'Naive (Random Walk)', 'Historical Mean'],
        'MAE': [arima_metrics['MAE'], naive_metrics['MAE'], mean_metrics['MAE']],
        'RMSE': [arima_metrics['RMSE'], naive_metrics['RMSE'], mean_metrics['RMSE']],
        'MAPE': [arima_metrics['MAPE'], naive_metrics['MAPE'], mean_metrics['MAPE']],
        'R2': [arima_metrics['R2'], naive_metrics['R2'], mean_metrics['R2']]
    })
    
    # Add directional accuracy
    comparison['Dir_Accuracy'] = [
        directional_accuracy(returns_test, arima_pred),
        directional_accuracy(returns_test, naive_pred),
        directional_accuracy(returns_test, mean_pred)
    ]
    
    print(comparison.to_string(index=False))
    
    return comparison

def plot_forecast_comparison(comparison_df):
    """Plot forecast method comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    models = comparison_df['Model']
    
    # MAE
    axes[0, 0].bar(range(len(models)), comparison_df['MAE'], 
                   color=['#2E86AB', '#F18F01', '#06A77D'], alpha=0.7, edgecolor='black')
    axes[0, 0].set_xticks(range(len(models)))
    axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
    axes[0, 0].set_title('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('MAE', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # RMSE
    axes[0, 1].bar(range(len(models)), comparison_df['RMSE'], 
                   color=['#2E86AB', '#F18F01', '#06A77D'], alpha=0.7, edgecolor='black')
    axes[0, 1].set_xticks(range(len(models)))
    axes[0, 1].set_xticklabels(models, rotation=45, ha='right')
    axes[0, 1].set_title('Root Mean Squared Error (RMSE)', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('RMSE', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # R-squared
    axes[1, 0].bar(range(len(models)), comparison_df['R2'], 
                   color=['#2E86AB', '#F18F01', '#06A77D'], alpha=0.7, edgecolor='black')
    axes[1, 0].set_xticks(range(len(models)))
    axes[1, 0].set_xticklabels(models, rotation=45, ha='right')
    axes[1, 0].set_title('R-squared', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('R²', fontsize=10)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # Directional Accuracy
    axes[1, 1].bar(range(len(models)), comparison_df['Dir_Accuracy'], 
                   color=['#2E86AB', '#F18F01', '#06A77D'], alpha=0.7, edgecolor='black')
    axes[1, 1].set_xticks(range(len(models)))
    axes[1, 1].set_xticklabels(models, rotation=45, ha='right')
    axes[1, 1].set_title('Directional Accuracy (%)', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Accuracy (%)', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].axhline(y=50, color='red', linestyle='--', linewidth=1, label='Random (50%)')
    axes[1, 1].legend()
    
    plt.tight_layout()
    return fig

def main():
    """Main execution function"""
    print("="*60)
    print("MSBD5006: Forecasting and Evaluation")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Split data - use last 252 trading days (~12 months) for testing
    train, test = split_data(df, test_days=252)
    
    # ARIMA rolling forecast
    arima_pred, arima_metrics = arima_rolling_forecast(train, test, order=(2, 0, 3))
    
    # Plot ARIMA forecast
    fig1 = plot_forecast_results(test, arima_pred, "ARIMA(2,0,3)")
    fig1.savefig('output/figures/22_arima_forecast.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n[OK] Saved: output/figures/22_arima_forecast.png")
    
    # GARCH volatility forecast
    garch_vol_pred, garch_metrics = garch_forecast(train, test)
    
    # Plot GARCH forecast
    fig2 = plot_volatility_forecast(test, garch_vol_pred)
    fig2.savefig('output/figures/23_garch_volatility_forecast.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: output/figures/23_garch_volatility_forecast.png")
    
    # Compare forecast methods
    comparison_df = compare_forecast_methods(test, arima_pred, arima_metrics)
    comparison_df.to_csv('output/forecast_comparison.csv', index=False)
    print("\n[OK] Saved: output/forecast_comparison.csv")
    
    # Plot comparison
    fig3 = plot_forecast_comparison(comparison_df)
    fig3.savefig('output/figures/24_forecast_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: output/figures/24_forecast_comparison.png")
    
    # Save predictions
    returns_test = test['Returns'].dropna()
    forecast_df = pd.DataFrame({
        'Date': returns_test.index,
        'Actual_Returns': returns_test.values,
        'ARIMA_Forecast': arima_pred,
        'GARCH_Volatility_Forecast': garch_vol_pred
    })
    forecast_df.to_csv('output/forecast_results.csv', index=False)
    print("[OK] Saved: output/forecast_results.csv")
    
    print("\n" + "="*60)
    print("FORECASTING COMPLETED!")
    print("="*60)
    print("\nGenerated Files:")
    print("- ARIMA forecast plots")
    print("- GARCH volatility forecast plots")
    print("- Forecast comparison plots")
    print("- Forecast results and metrics CSV files")
    print("\nNext Step: Create comprehensive Jupyter notebook")

if __name__ == "__main__":
    main()


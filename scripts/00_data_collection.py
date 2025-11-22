"""
Script 00: Data Collection

This script fetches Hang Seng Index (HSI) data from Yahoo Finance
using the yfinance API and performs basic data validation.


"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import os

# Create output directory if it doesn't exist
os.makedirs('data', exist_ok=True)
os.makedirs('output', exist_ok=True)

def fetch_hsi_data(start_date='2015-01-01', end_date='2025-11-22'):
    """
    Fetch Hang Seng Index data from Yahoo Finance
    
    Parameters:
    -----------
    start_date : str
        Start date for data collection (YYYY-MM-DD)
    end_date : str
        End date for data collection (YYYY-MM-DD)
    
    Returns:
    --------
    pd.DataFrame
        HSI data with OHLCV columns
    """
    print(f"Fetching HSI data from {start_date} to {end_date}...")
    
    # Download HSI data (^HSI is the ticker for Hang Seng Index)
    hsi = yf.download('^HSI', start=start_date, end=end_date, progress=True)
    
    print(f"\nData fetched successfully!")
    print(f"Total rows: {len(hsi)}")
    print(f"Date range: {hsi.index[0]} to {hsi.index[-1]}")
    
    return hsi

def validate_data(df):
    """
    Perform data quality checks
    
    Parameters:
    -----------
    df : pd.DataFrame
        HSI data
    
    Returns:
    --------
    dict
        Validation results
    """
    print("\n" + "="*60)
    print("DATA VALIDATION REPORT")
    print("="*60)
    
    validation = {}
    
    # Check for missing values
    missing = df.isnull().sum()
    validation['missing_values'] = missing
    print("\nMissing Values:")
    print(missing)
    
    # Check for duplicates
    duplicates = df.index.duplicated().sum()
    validation['duplicates'] = duplicates
    print(f"\nDuplicate dates: {duplicates}")
    
    # Check data types
    print("\nData Types:")
    print(df.dtypes)
    
    # Basic statistics
    print("\nBasic Statistics:")
    print(df.describe())
    
    # Check for negative values (shouldn't exist in price data)
    negative_values = (df < 0).sum()
    validation['negative_values'] = negative_values
    if negative_values.sum() > 0:
        print("\n[WARNING] Negative values found!")
        print(negative_values[negative_values > 0])
    else:
        print("\n[OK] No negative values found")
    
    # Check for outliers using IQR method
    print("\nChecking for potential outliers in returns...")
    returns = df['Close'].pct_change()
    Q1 = returns.quantile(0.25)
    Q3 = returns.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((returns < (Q1 - 3 * IQR)) | (returns > (Q3 + 3 * IQR))).sum()
    validation['outliers'] = outliers
    print(f"Potential outliers (3*IQR rule): {outliers}")
    
    return validation

def save_data(df, filename='hsi_raw.csv'):
    """
    Save data to CSV file
    
    Parameters:
    -----------
    df : pd.DataFrame
        HSI data
    filename : str
        Output filename
    """
    filepath = os.path.join('data', filename)
    df.to_csv(filepath)
    print(f"\n[OK] Data saved to: {filepath}")
    
    # Also save data info
    info_path = os.path.join('output', 'data_info.txt')
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("HSI DATA INFORMATION\n")
        f.write("="*60 + "\n\n")
        f.write(f"Data Source: Yahoo Finance (^HSI)\n")
        f.write(f"Collection Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Observations: {len(df)}\n")
        f.write(f"Date Range: {df.index[0]} to {df.index[-1]}\n")
        # Handle multi-level columns
        if isinstance(df.columns, pd.MultiIndex):
            col_str = ', '.join([str(col) for col in df.columns])
        else:
            col_str = ', '.join(df.columns)
        f.write(f"Columns: {col_str}\n\n")
        f.write("Summary Statistics:\n")
        f.write(str(df.describe()))
    
    print(f"[OK] Data info saved to: {info_path}")

def main():
    """Main execution function"""
    print("="*60)
    print("MSBD5006: HSI Data Collection")
    print("="*60)
    
    # Fetch data
    hsi_data = fetch_hsi_data()
    
    # Validate data
    validation_results = validate_data(hsi_data)
    
    # Save data
    save_data(hsi_data)
    
    print("\n" + "="*60)
    print("DATA COLLECTION COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nNext Steps:")
    print("1. Run scripts/01_eda.py for Exploratory Data Analysis")
    print("2. Check output/data_info.txt for detailed data information")

if __name__ == "__main__":
    main()


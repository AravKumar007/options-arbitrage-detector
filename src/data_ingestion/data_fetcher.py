"""
Data Fetcher - Processes and cleans raw options data.
Prepares data for storage and analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.data_ingestion.api_client import OptionsAPIClient
from src.pricing.black_scholes import BlackScholes
from src.pricing.implied_vol import ImpliedVolatility
from src.utils.logger import get_logger
from src.utils.helpers import calculate_time_to_maturity, calculate_moneyness

logger = get_logger(__name__)


class OptionsDataFetcher:
    """
    Fetch, process, and enrich options chain data.
    """
    
    def __init__(self, api_source: str = "yfinance", risk_free_rate: float = 0.05):
        """
        Initialize data fetcher.
        
        Args:
            api_source: API source for data
            risk_free_rate: Risk-free interest rate
        """
        self.api_client = OptionsAPIClient(source=api_source)
        self.bs_model = BlackScholes()
        self.iv_calculator = ImpliedVolatility()
        self.risk_free_rate = risk_free_rate
        logger.info(f"OptionsDataFetcher initialized with r={risk_free_rate}")
    
    def fetch_and_process(self, ticker: str, expiration: str = None) -> pd.DataFrame:
        """
        Fetch and process options chain with enriched data.
        
        Args:
            ticker: Stock ticker symbol
            expiration: Expiration date (YYYY-MM-DD) or None for nearest
        
        Returns:
            Processed DataFrame with options and calculated fields
        """
        # Fetch raw data
        logger.info(f"Fetching options data for {ticker}...")
        chain = self.api_client.get_option_chain(ticker, expiration)
        spot_price = self.api_client.get_current_price(ticker)
        
        if spot_price is None:
            logger.error(f"Could not fetch spot price for {ticker}")
            return pd.DataFrame()
        
        # Combine calls and puts
        all_options = pd.concat([chain['calls'], chain['puts']], ignore_index=True)
        
        if all_options.empty:
            logger.warning(f"No options data found for {ticker}")
            return pd.DataFrame()
        
        # Add spot price
        all_options['spot_price'] = spot_price
        
        # Process and enrich
        processed = self._enrich_data(all_options)
        
        logger.info(f"Processed {len(processed)} options contracts for {ticker}")
        return processed
    
    def _enrich_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add calculated fields to options data.
        
        Args:
            df: Raw options DataFrame
        
        Returns:
            Enriched DataFrame
        """
        if df.empty:
            return df
        
        # Make a copy to avoid modifying original
        enriched = df.copy()
        
        # Calculate time to maturity
        enriched['time_to_maturity'] = enriched['expiration'].apply(
            lambda x: calculate_time_to_maturity(x)
        )
        
        # Calculate moneyness
        enriched['moneyness'] = enriched.apply(
            lambda row: calculate_moneyness(row['spot_price'], row['strike']),
            axis=1
        )
        
        # Classify options (ITM/ATM/OTM)
        enriched['classification'] = enriched.apply(
            lambda row: self._classify_option(
                row['spot_price'], 
                row['strike'], 
                row['option_type']
            ),
            axis=1
        )
        
        # Calculate mid price
        enriched['mid_price'] = (enriched['bid'] + enriched['ask']) / 2
        
        # Calculate bid-ask spread
        enriched['spread'] = enriched['ask'] - enriched['bid']
        enriched['spread_pct'] = enriched['spread'] / enriched['mid_price'] * 100
        
        # Calculate intrinsic and time value
        enriched['intrinsic_value'] = enriched.apply(
            lambda row: self._calculate_intrinsic_value(
                row['spot_price'],
                row['strike'],
                row['option_type']
            ),
            axis=1
        )
        
        enriched['time_value'] = enriched['mid_price'] - enriched['intrinsic_value']
        
        # Calculate Greeks using our Black-Scholes model
        logger.info("Calculating Greeks for all options...")
        enriched = self._calculate_greeks(enriched)
        
        # Add data quality score
        enriched['quality_score'] = enriched.apply(self._calculate_quality_score, axis=1)
        
        return enriched
    
    def _classify_option(self, spot: float, strike: float, option_type: str) -> str:
        """
        Classify option as ITM, ATM, or OTM.
        
        Args:
            spot: Spot price
            strike: Strike price
            option_type: 'call' or 'put'
        
        Returns:
            Classification string
        """
        moneyness = spot / strike
        
        if abs(moneyness - 1.0) < 0.05:  # Within 5%
            return 'ATM'
        
        if option_type == 'call':
            return 'ITM' if spot > strike else 'OTM'
        else:  # put
            return 'ITM' if spot < strike else 'OTM'
    
    def _calculate_intrinsic_value(self, spot: float, strike: float, option_type: str) -> float:
        """
        Calculate intrinsic value of option.
        
        Args:
            spot: Spot price
            strike: Strike price
            option_type: 'call' or 'put'
        
        Returns:
            Intrinsic value
        """
        if option_type == 'call':
            return max(spot - strike, 0)
        else:  # put
            return max(strike - spot, 0)
    
    def _calculate_greeks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Greeks for all options using Black-Scholes.
        
        Args:
            df: Options DataFrame
        
        Returns:
            DataFrame with Greeks columns added
        """
        greeks_list = []
        
        for idx, row in df.iterrows():
            try:
                S = row['spot_price']
                K = row['strike']
                T = row['time_to_maturity']
                r = self.risk_free_rate
                sigma = row.get('impliedVolatility', 0.3)  # Use IV if available
                option_type = row['option_type']
                
                # Skip if invalid parameters
                if T <= 0 or sigma <= 0:
                    greeks = {
                        'delta': np.nan, 'gamma': np.nan, 'vega': np.nan,
                        'theta': np.nan, 'rho': np.nan
                    }
                else:
                    greeks = self.bs_model.greeks(S, K, T, r, sigma, option_type)
                
                greeks_list.append(greeks)
                
            except Exception as e:
                logger.warning(f"Error calculating Greeks for row {idx}: {e}")
                greeks_list.append({
                    'delta': np.nan, 'gamma': np.nan, 'vega': np.nan,
                    'theta': np.nan, 'rho': np.nan
                })
        
        # Add Greeks to DataFrame
        greeks_df = pd.DataFrame(greeks_list)
        for col in greeks_df.columns:
            df[col] = greeks_df[col]
        
        return df
    
    def _calculate_quality_score(self, row: pd.Series) -> float:
        """
        Calculate data quality score (0-100).
        
        Args:
            row: DataFrame row
        
        Returns:
            Quality score
        """
        score = 100.0
        
        # Penalize wide spreads
        if not pd.isna(row.get('spread_pct')):
            if row['spread_pct'] > 10:
                score -= 20
            elif row['spread_pct'] > 5:
                score -= 10
        
        # Penalize low volume
        if not pd.isna(row.get('volume')):
            if row['volume'] == 0:
                score -= 30
            elif row['volume'] < 10:
                score -= 15
        
        # Penalize low open interest
        if not pd.isna(row.get('openInterest')):
            if row['openInterest'] == 0:
                score -= 20
            elif row['openInterest'] < 50:
                score -= 10
        
        # Penalize missing implied volatility
        if pd.isna(row.get('impliedVolatility')) or row.get('impliedVolatility', 0) == 0:
            score -= 15
        
        return max(score, 0)
    
    def fetch_multiple_expirations(self, ticker: str, 
                                   num_expirations: int = 3) -> pd.DataFrame:
        """
        Fetch and process multiple expiration dates.
        
        Args:
            ticker: Stock ticker symbol
            num_expirations: Number of expirations to fetch
        
        Returns:
            Combined DataFrame for all expirations
        """
        expirations = self.api_client.get_available_expirations(ticker)
        
        if not expirations:
            logger.error(f"No expirations found for {ticker}")
            return pd.DataFrame()
        
        # Limit to requested number
        expirations = expirations[:num_expirations]
        
        all_data = []
        for exp in expirations:
            logger.info(f"Processing expiration {exp}...")
            data = self.fetch_and_process(ticker, exp)
            if not data.empty:
                all_data.append(data)
        
        if not all_data:
            return pd.DataFrame()
        
        combined = pd.concat(all_data, ignore_index=True)
        logger.info(f"Combined {len(combined)} options across {len(expirations)} expirations")
        
        return combined
    
    def get_atm_options(self, df: pd.DataFrame, tolerance: float = 0.05) -> pd.DataFrame:
        """
        Filter for at-the-money options.
        
        Args:
            df: Options DataFrame
            tolerance: ATM tolerance (default 5%)
        
        Returns:
            Filtered DataFrame
        """
        atm = df[df['classification'] == 'ATM'].copy()
        logger.info(f"Found {len(atm)} ATM options")
        return atm
    
    def get_liquid_options(self, df: pd.DataFrame, 
                          min_volume: int = 10,
                          min_open_interest: int = 50) -> pd.DataFrame:
        """
        Filter for liquid options.
        
        Args:
            df: Options DataFrame
            min_volume: Minimum volume
            min_open_interest: Minimum open interest
        
        Returns:
            Filtered DataFrame
        """
        liquid = df[
            (df['volume'] >= min_volume) & 
            (df['openInterest'] >= min_open_interest)
        ].copy()
        
        logger.info(f"Found {len(liquid)} liquid options")
        return liquid


# Example usage and testing
if __name__ == "__main__":
    print(f"\n{'='*70}")
    print(f"Options Data Fetcher Test")
    print(f"{'='*70}")
    
    # Initialize fetcher
    fetcher = OptionsDataFetcher()
    
    # Test ticker
    ticker = "AAPL"
    
    # Fetch and process options
    print(f"\n1. Fetching and processing options for {ticker}...")
    options_df = fetcher.fetch_and_process(ticker)
    
    if not options_df.empty:
        print(f"\nProcessed {len(options_df)} options contracts")
        print(f"\nColumns: {list(options_df.columns)}")
        
        print(f"\n2. Sample processed data (first 5 rows):")
        display_cols = ['strike', 'option_type', 'bid', 'ask', 'mid_price', 
                       'moneyness', 'classification', 'delta', 'quality_score']
        print(options_df[display_cols].head())
        
        # Get ATM options
        print(f"\n3. At-The-Money Options:")
        atm = fetcher.get_atm_options(options_df)
        if not atm.empty:
            print(atm[display_cols].head())
        
        # Get liquid options
        print(f"\n4. Liquid Options (volume >= 10, OI >= 50):")
        liquid = fetcher.get_liquid_options(options_df)
        print(f"Found {len(liquid)} liquid options")
        
        # Quality statistics
        print(f"\n5. Data Quality Statistics:")
        print(f"Average quality score: {options_df['quality_score'].mean():.1f}")
        print(f"Options with quality > 80: {(options_df['quality_score'] > 80).sum()}")
        
        # Greeks statistics
        print(f"\n6. Greeks Statistics:")
        print(f"Average Delta (calls): {options_df[options_df['option_type']=='call']['delta'].mean():.4f}")
        print(f"Average Delta (puts): {options_df[options_df['option_type']=='put']['delta'].mean():.4f}")
    
    logger.info("Data fetcher test completed successfully")
    
    
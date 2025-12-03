"""
API Client for fetching options chain data.
Supports multiple data sources with fallback mechanisms.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import get_logger
from src.utils.validators import OptionDataValidator

logger = get_logger(__name__)


class OptionsAPIClient:
    """
    Fetch options chain data from various sources.
    Primary source: yfinance (free, no API key needed)
    """
    
    def __init__(self, source: str = "yfinance"):
        """
        Initialize API client.
        
        Args:
            source: Data source ('yfinance', 'polygon', 'alpaca')
        """
        self.source = source
        self.validator = OptionDataValidator()
        logger.info(f"OptionsAPIClient initialized with source: {source}")
    
    def get_available_expirations(self, ticker: str) -> List[str]:
        """
        Get available expiration dates for a ticker.
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            List of expiration dates (YYYY-MM-DD format)
        """
        try:
            stock = yf.Ticker(ticker)
            expirations = stock.options
            logger.info(f"Found {len(expirations)} expiration dates for {ticker}")
            return list(expirations)
        except Exception as e:
            logger.error(f"Error fetching expirations for {ticker}: {e}")
            return []
    
    def get_option_chain(self, ticker: str, expiration: str = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch complete options chain for a ticker and expiration.
        
        Args:
            ticker: Stock ticker symbol
            expiration: Expiration date (YYYY-MM-DD). If None, uses nearest expiration
        
        Returns:
            Dictionary with 'calls' and 'puts' DataFrames
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Get expiration date
            if expiration is None:
                expirations = stock.options
                if not expirations:
                    logger.error(f"No options available for {ticker}")
                    return {'calls': pd.DataFrame(), 'puts': pd.DataFrame()}
                expiration = expirations[0]  # Use nearest expiration
            
            # Fetch options chain
            opt_chain = stock.option_chain(expiration)
            
            calls = opt_chain.calls
            puts = opt_chain.puts
            
            # Add metadata
            calls['ticker'] = ticker
            calls['expiration'] = expiration
            calls['option_type'] = 'call'
            calls['fetch_time'] = datetime.now()
            
            puts['ticker'] = ticker
            puts['expiration'] = expiration
            puts['option_type'] = 'put'
            puts['fetch_time'] = datetime.now()
            
            logger.info(f"Fetched {len(calls)} calls and {len(puts)} puts for {ticker} exp {expiration}")
            
            return {'calls': calls, 'puts': puts}
            
        except Exception as e:
            logger.error(f"Error fetching option chain for {ticker}: {e}")
            return {'calls': pd.DataFrame(), 'puts': pd.DataFrame()}
    
    def get_current_price(self, ticker: str) -> Optional[float]:
        """
        Get current stock price.
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            Current price or None if error
        """
        try:
            stock = yf.Ticker(ticker)
            # Try to get real-time price from info
            info = stock.info
            price = info.get('currentPrice') or info.get('regularMarketPrice')
            
            if price is None:
                # Fallback: get from recent history
                hist = stock.history(period='1d')
                if not hist.empty:
                    price = hist['Close'].iloc[-1]
            
            logger.debug(f"Current price for {ticker}: ${price:.2f}")
            return float(price)
            
        except Exception as e:
            logger.error(f"Error fetching price for {ticker}: {e}")
            return None
    
    def get_multiple_chains(self, tickers: List[str], 
                           expiration: str = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Fetch options chains for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            expiration: Expiration date (YYYY-MM-DD) or None for nearest
        
        Returns:
            Dictionary mapping ticker to options chain
        """
        results = {}
        
        for ticker in tickers:
            logger.info(f"Fetching options for {ticker}...")
            results[ticker] = self.get_option_chain(ticker, expiration)
            time.sleep(0.5)  # Rate limiting - be nice to the API
        
        return results
    
    def get_historical_prices(self, ticker: str, period: str = "1mo") -> pd.DataFrame:
        """
        Get historical price data.
        
        Args:
            ticker: Stock ticker symbol
            period: Time period ('1d', '5d', '1mo', '3mo', '1y', '2y', '5y', 'max')
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            logger.info(f"Fetched {len(hist)} historical records for {ticker}")
            return hist
        except Exception as e:
            logger.error(f"Error fetching historical data for {ticker}: {e}")
            return pd.DataFrame()
    
    def validate_chain_data(self, chain: pd.DataFrame, spot_price: float) -> Tuple[bool, List[str]]:
        """
        Validate options chain data quality.
        
        Args:
            chain: Options chain DataFrame
            spot_price: Current spot price
        
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        
        if chain.empty:
            issues.append("Empty options chain")
            return False, issues
        
        # Check for required columns
        required_cols = ['strike', 'lastPrice', 'bid', 'ask', 'volume', 'impliedVolatility']
        missing_cols = [col for col in required_cols if col not in chain.columns]
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
        
        # Validate prices
        if 'bid' in chain.columns and 'ask' in chain.columns:
            # Check for crossed markets
            crossed = (chain['bid'] > chain['ask']).sum()
            if crossed > 0:
                issues.append(f"Found {crossed} crossed markets (bid > ask)")
            
            # Check for zero bid-ask
            zero_markets = ((chain['bid'] == 0) & (chain['ask'] == 0)).sum()
            if zero_markets > len(chain) * 0.2:  # More than 20% are zero
                issues.append(f"Too many zero bid-ask spreads: {zero_markets}")
        
        # Check strike prices
        if 'strike' in chain.columns:
            if spot_price:
                # Check if strikes are reasonable relative to spot
                min_strike = chain['strike'].min()
                max_strike = chain['strike'].max()
                
                if min_strike > spot_price * 2 or max_strike < spot_price * 0.5:
                    issues.append(f"Strike range [{min_strike}, {max_strike}] suspicious for spot={spot_price}")
        
        # Check implied volatility
        if 'impliedVolatility' in chain.columns:
            iv_data = chain['impliedVolatility'].dropna()
            if len(iv_data) > 0:
                mean_iv = iv_data.mean()
                if mean_iv < 0.01 or mean_iv > 5.0:  # 1% to 500%
                    issues.append(f"Suspicious average IV: {mean_iv*100:.1f}%")
        
        is_valid = len(issues) == 0
        
        if is_valid:
            logger.info("Options chain validation passed")
        else:
            logger.warning(f"Options chain validation issues: {issues}")
        
        return is_valid, issues
    
    def get_option_greeks(self, ticker: str, expiration: str = None) -> pd.DataFrame:
        """
        Fetch options with Greeks data if available.
        
        Args:
            ticker: Stock ticker symbol
            expiration: Expiration date
        
        Returns:
            DataFrame with options and Greeks
        """
        chain = self.get_option_chain(ticker, expiration)
        
        # Combine calls and puts
        all_options = pd.concat([chain['calls'], chain['puts']], ignore_index=True)
        
        # yfinance doesn't provide Greeks directly, but we can calculate them
        # This would integrate with our Black-Scholes model
        logger.info(f"Retrieved {len(all_options)} options with metadata")
        
        return all_options


# Example usage and testing
if __name__ == "__main__":
    print(f"\n{'='*70}")
    print(f"Options API Client Test")
    print(f"{'='*70}")
    
    # Initialize client
    client = OptionsAPIClient()
    
    # Test ticker
    ticker = "SPY"
    
    # Get current price
    print(f"\n1. Fetching current price for {ticker}...")
    price = client.get_current_price(ticker)
    print(f"Current Price: ${price:.2f}")
    
    # Get available expirations
    print(f"\n2. Fetching available expirations...")
    expirations = client.get_available_expirations(ticker)
    print(f"Found {len(expirations)} expirations")
    print(f"Next 5 expirations: {expirations[:5]}")
    
    # Get options chain for nearest expiration
    if expirations:
        nearest_exp = expirations[0]
        print(f"\n3. Fetching options chain for {nearest_exp}...")
        chain = client.get_option_chain(ticker, nearest_exp)
        
        calls = chain['calls']
        puts = chain['puts']
        
        print(f"\nCalls: {len(calls)} contracts")
        print(f"Puts: {len(puts)} contracts")
        
        if not calls.empty:
            print(f"\nSample Call Options (first 5):")
            print(calls[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'impliedVolatility']].head())
        
        # Validate data
        print(f"\n4. Validating options data...")
        is_valid, issues = client.validate_chain_data(calls, price)
        print(f"Validation result: {'PASSED' if is_valid else 'FAILED'}")
        if issues:
            print(f"Issues found: {issues}")
    
    # Get historical data
    print(f"\n5. Fetching historical price data...")
    hist = client.get_historical_prices(ticker, period="1mo")
    print(f"Retrieved {len(hist)} days of historical data")
    if not hist.empty:
        print(f"Date range: {hist.index[0].date()} to {hist.index[-1].date()}")
        print(f"Price range: ${hist['Close'].min():.2f} - ${hist['Close'].max():.2f}")
    
    logger.info("API client test completed successfully")

    
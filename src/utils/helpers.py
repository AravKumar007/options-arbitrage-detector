"""
Helper utility functions for the options arbitrage project.
Common calculations and data transformations.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Union, List, Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_time_to_maturity(expiration_date: Union[str, datetime], 
                               current_date: Union[str, datetime] = None) -> float:
    """
    Calculate time to maturity in years.
    
    Args:
        expiration_date: Option expiration date
        current_date: Current date (defaults to today)
    
    Returns:
        Time to maturity in years
    """
    if isinstance(expiration_date, str):
        expiration_date = pd.to_datetime(expiration_date)
    
    if current_date is None:
        current_date = datetime.now()
    elif isinstance(current_date, str):
        current_date = pd.to_datetime(current_date)
    
    days_to_maturity = (expiration_date - current_date).days
    
    # Use trading days approximation (252 trading days per year)
    years_to_maturity = days_to_maturity / 365.0
    
    return max(years_to_maturity, 0)  # Ensure non-negative


def calculate_moneyness(S: float, K: float, method: str = 'simple') -> float:
    """
    Calculate option moneyness.
    
    Args:
        S: Spot price
        K: Strike price
        method: 'simple' (S/K) or 'log' (ln(S/K))
    
    Returns:
        Moneyness value
    """
    if method == 'simple':
        return S / K
    elif method == 'log':
        return np.log(S / K)
    else:
        raise ValueError(f"Unknown moneyness method: {method}")


def classify_option(S: float, K: float, threshold: float = 0.05) -> str:
    """
    Classify option as ITM, ATM, or OTM.
    
    Args:
        S: Spot price
        K: Strike price
        threshold: ATM threshold (default 5%)
    
    Returns:
        Classification: 'ITM', 'ATM', or 'OTM'
    """
    moneyness = S / K
    
    if abs(moneyness - 1.0) < threshold:
        return 'ATM'
    elif moneyness > 1.0:
        return 'ITM'  # For calls
    else:
        return 'OTM'  # For calls


def annualized_return(initial_value: float, final_value: float, 
                     time_period: float) -> float:
    """
    Calculate annualized return.
    
    Args:
        initial_value: Starting value
        final_value: Ending value
        time_period: Time period in years
    
    Returns:
        Annualized return
    """
    if initial_value <= 0 or time_period <= 0:
        return 0.0
    
    total_return = (final_value - initial_value) / initial_value
    annualized = (1 + total_return) ** (1 / time_period) - 1
    
    return annualized


def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio.
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate (annualized)
    
    Returns:
        Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate
    
    if np.std(excess_returns) == 0:
        return 0.0
    
    return np.mean(excess_returns) / np.std(excess_returns)


def max_drawdown(equity_curve: np.ndarray) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown from equity curve.
    
    Args:
        equity_curve: Array of portfolio values over time
    
    Returns:
        Tuple of (max_drawdown, start_index, end_index)
    """
    cumulative_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - cumulative_max) / cumulative_max
    
    max_dd = np.min(drawdown)
    end_idx = np.argmin(drawdown)
    start_idx = np.argmax(equity_curve[:end_idx]) if end_idx > 0 else 0
    
    return abs(max_dd), start_idx, end_idx


def calculate_spread_cost(bid: float, ask: float) -> float:
    """
    Calculate bid-ask spread as percentage of mid price.
    
    Args:
        bid: Bid price
        ask: Ask price
    
    Returns:
        Spread cost as percentage
    """
    mid = (bid + ask) / 2
    if mid == 0:
        return 0.0
    
    return (ask - bid) / mid


def interpolate_iv(strikes: np.ndarray, ivs: np.ndarray, 
                   target_strike: float, method: str = 'linear') -> float:
    """
    Interpolate implied volatility for a given strike.
    
    Args:
        strikes: Array of strikes
        ivs: Array of implied volatilities
        target_strike: Strike to interpolate
        method: Interpolation method ('linear', 'cubic')
    
    Returns:
        Interpolated IV
    """
    # Remove NaN values
    mask = ~np.isnan(ivs)
    strikes = strikes[mask]
    ivs = ivs[mask]
    
    if len(strikes) == 0:
        logger.warning("No valid IV data for interpolation")
        return np.nan
    
    if target_strike < strikes.min() or target_strike > strikes.max():
        logger.warning(f"Target strike {target_strike} outside data range")
        return np.nan
    
    return np.interp(target_strike, strikes, ivs)


def bootstrap_confidence_interval(data: np.ndarray, statistic_func, 
                                  n_bootstrap: int = 1000, 
                                  confidence_level: float = 0.95) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval for a statistic.
    
    Args:
        data: Input data
        statistic_func: Function to calculate statistic (e.g., np.mean)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95%)
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_stats.append(statistic_func(sample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_stats, alpha/2 * 100)
    upper = np.percentile(bootstrap_stats, (1 - alpha/2) * 100)
    
    return lower, upper


def format_large_number(num: float) -> str:
    """
    Format large numbers with K, M, B suffixes.
    
    Args:
        num: Number to format
    
    Returns:
        Formatted string
    """
    if abs(num) >= 1e9:
        return f"${num/1e9:.2f}B"
    elif abs(num) >= 1e6:
        return f"${num/1e6:.2f}M"
    elif abs(num) >= 1e3:
        return f"${num/1e3:.2f}K"
    else:
        return f"${num:.2f}"


def calculate_transaction_cost(price: float, quantity: int, 
                               commission_per_contract: float = 0.65,
                               exchange_fee: float = 0.10) -> float:
    """
    Calculate total transaction cost for options trade.
    
    Args:
        price: Option price
        quantity: Number of contracts
        commission_per_contract: Broker commission per contract
        exchange_fee: Exchange fee per contract
    
    Returns:
        Total transaction cost
    """
    commission = quantity * commission_per_contract
    fees = quantity * exchange_fee
    total_cost = commission + fees
    
    return total_cost


def days_to_expiration(expiry_date: Union[str, datetime]) -> int:
    """
    Calculate days remaining to expiration.
    
    Args:
        expiry_date: Expiration date
    
    Returns:
        Number of days to expiration
    """
    if isinstance(expiry_date, str):
        expiry_date = pd.to_datetime(expiry_date)
    
    today = datetime.now()
    days = (expiry_date - today).days
    
    return max(days, 0)


# Example usage and testing
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"Helper Functions Tests")
    print(f"{'='*60}")
    
    # Test time to maturity
    print("\n1. Time to Maturity:")
    expiry = datetime.now() + timedelta(days=30)
    ttm = calculate_time_to_maturity(expiry)
    print(f"Time to maturity (30 days): {ttm:.4f} years ({ttm*365:.1f} days)")
    
    # Test moneyness
    print("\n2. Moneyness:")
    S, K = 100, 105
    print(f"Simple moneyness (S={S}, K={K}): {calculate_moneyness(S, K, 'simple'):.4f}")
    print(f"Log moneyness (S={S}, K={K}): {calculate_moneyness(S, K, 'log'):.4f}")
    
    # Test classification
    print("\n3. Option Classification:")
    print(f"S=100, K=100: {classify_option(100, 100)}")
    print(f"S=110, K=100: {classify_option(110, 100)}")
    print(f"S=90, K=100: {classify_option(90, 100)}")
    
    # Test Sharpe ratio
    print("\n4. Sharpe Ratio:")
    returns = np.random.normal(0.001, 0.02, 252)  # Daily returns for 1 year
    sr = sharpe_ratio(returns)
    print(f"Sharpe Ratio (random data): {sr:.4f}")
    
    # Test max drawdown
    print("\n5. Maximum Drawdown:")
    equity = np.array([100, 110, 105, 120, 115, 90, 95, 100, 110])
    mdd, start, end = max_drawdown(equity)
    print(f"Max Drawdown: {mdd*100:.2f}%")
    print(f"Drawdown period: index {start} to {end}")
    
    # Test spread cost
    print("\n6. Bid-Ask Spread:")
    bid, ask = 10.0, 10.5
    spread = calculate_spread_cost(bid, ask)
    print(f"Bid: ${bid}, Ask: ${ask}")
    print(f"Spread cost: {spread*100:.2f}%")
    
    # Test transaction cost
    print("\n7. Transaction Cost:")
    cost = calculate_transaction_cost(price=10.0, quantity=10)
    print(f"Transaction cost for 10 contracts at $10: ${cost:.2f}")
    
    # Test large number formatting
    print("\n8. Number Formatting:")
    print(f"1,500: {format_large_number(1500)}")
    print(f"1,500,000: {format_large_number(1500000)}")
    print(f"1,500,000,000: {format_large_number(1500000000)}")
    
    logger.info("Helper functions tests completed")
    
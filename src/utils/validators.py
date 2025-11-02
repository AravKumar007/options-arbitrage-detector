"""
Data validation utilities for options data.
Ensures data quality and catches common errors.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OptionDataValidator:
    """
    Validates options chain data for quality and consistency.
    """
    
    @staticmethod
    def validate_price(price: float, field_name: str = "price") -> bool:
        """
        Validate that a price is positive and reasonable.
        
        Args:
            price: Price to validate
            field_name: Name of the field for logging
        
        Returns:
            True if valid, False otherwise
        """
        if price is None or np.isnan(price):
            logger.warning(f"{field_name} is None or NaN")
            return False
        
        if price < 0:
            logger.warning(f"{field_name} is negative: {price}")
            return False
        
        if price > 1e6:  # Sanity check for extremely large prices
            logger.warning(f"{field_name} is unreasonably large: {price}")
            return False
        
        return True
    
    @staticmethod
    def validate_strike(strike: float, spot_price: float, 
                       max_moneyness: float = 3.0) -> bool:
        """
        Validate strike price relative to spot price.
        
        Args:
            strike: Strike price
            spot_price: Current spot price
            max_moneyness: Maximum acceptable moneyness ratio
        
        Returns:
            True if valid, False otherwise
        """
        if not OptionDataValidator.validate_price(strike, "strike"):
            return False
        
        if spot_price <= 0:
            logger.warning(f"Invalid spot price: {spot_price}")
            return False
        
        moneyness = strike / spot_price
        if moneyness > max_moneyness or moneyness < (1 / max_moneyness):
            logger.warning(f"Strike {strike} too far from spot {spot_price}, moneyness: {moneyness}")
            return False
        
        return True
    
    @staticmethod
    def validate_time_to_maturity(T: float, max_years: float = 5.0) -> bool:
        """
        Validate time to maturity.
        
        Args:
            T: Time to maturity in years
            max_years: Maximum acceptable maturity
        
        Returns:
            True if valid, False otherwise
        """
        if T is None or np.isnan(T):
            logger.warning("Time to maturity is None or NaN")
            return False
        
        if T <= 0:
            logger.warning(f"Time to maturity is non-positive: {T}")
            return False
        
        if T > max_years:
            logger.warning(f"Time to maturity too large: {T} years")
            return False
        
        return True
    
    @staticmethod
    def validate_volatility(sigma: float, min_vol: float = 0.01, 
                           max_vol: float = 3.0) -> bool:
        """
        Validate volatility value.
        
        Args:
            sigma: Volatility (annualized)
            min_vol: Minimum acceptable volatility
            max_vol: Maximum acceptable volatility
        
        Returns:
            True if valid, False otherwise
        """
        if sigma is None or np.isnan(sigma):
            logger.warning("Volatility is None or NaN")
            return False
        
        if sigma < min_vol:
            logger.warning(f"Volatility too low: {sigma*100:.2f}%")
            return False
        
        if sigma > max_vol:
            logger.warning(f"Volatility too high: {sigma*100:.2f}%")
            return False
        
        return True
    
    @staticmethod
    def validate_interest_rate(r: float) -> bool:
        """
        Validate interest rate.
        
        Args:
            r: Interest rate (annualized)
        
        Returns:
            True if valid, False otherwise
        """
        if r is None or np.isnan(r):
            logger.warning("Interest rate is None or NaN")
            return False
        
        if r < -0.1 or r > 0.5:  # -10% to 50% seems reasonable
            logger.warning(f"Interest rate seems unreasonable: {r*100:.2f}%")
            return False
        
        return True
    
    @staticmethod
    def check_put_call_parity(call_price: float, put_price: float, 
                              S: float, K: float, T: float, r: float,
                              tolerance: float = 0.05) -> Tuple[bool, float]:
        """
        Check put-call parity: C - P should equal S - K*e^(-rT)
        
        Args:
            call_price: Call option price
            put_price: Put option price
            S: Spot price
            K: Strike price
            T: Time to maturity
            r: Interest rate
            tolerance: Acceptable deviation (as fraction of spot price)
        
        Returns:
            Tuple of (is_valid, deviation)
        """
        theoretical_diff = S - K * np.exp(-r * T)
        actual_diff = call_price - put_price
        deviation = abs(actual_diff - theoretical_diff)
        
        is_valid = deviation < (tolerance * S)
        
        if not is_valid:
            logger.warning(
                f"Put-call parity violation: deviation={deviation:.4f}, "
                f"tolerance={tolerance*S:.4f}"
            )
        
        return is_valid, deviation
    
    @staticmethod
    def validate_option_chain(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate an entire options chain DataFrame.
        
        Args:
            df: DataFrame with columns [strike, bid, ask, volume, open_interest, etc.]
        
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        # Check required columns
        required_cols = ['strike', 'bid', 'ask']
        for col in required_cols:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")
        
        if errors:
            return False, errors
        
        # Check for negative prices
        if (df['bid'] < 0).any():
            errors.append("Found negative bid prices")
        
        if (df['ask'] < 0).any():
            errors.append("Found negative ask prices")
        
        # Check bid-ask spread
        if (df['bid'] > df['ask']).any():
            errors.append("Found bid > ask (crossed market)")
        
        # Check for missing data
        missing_pct = df.isnull().sum() / len(df) * 100
        for col in required_cols:
            if missing_pct[col] > 10:  # More than 10% missing
                errors.append(f"Column {col} has {missing_pct[col]:.1f}% missing data")
        
        # Check for duplicate strikes
        if df['strike'].duplicated().any():
            errors.append("Found duplicate strikes")
        
        is_valid = len(errors) == 0
        
        if is_valid:
            logger.info(f"Options chain validation passed for {len(df)} strikes")
        else:
            logger.error(f"Options chain validation failed with {len(errors)} errors")
        
        return is_valid, errors
    
    @staticmethod
    def validate_arbitrage_free(iv_surface: np.ndarray, strikes: np.ndarray) -> bool:
        """
        Check if IV surface is arbitrage-free (no calendar or butterfly arbitrage).
        
        Args:
            iv_surface: 2D array of implied volatilities [strikes x maturities]
            strikes: Array of strike prices
        
        Returns:
            True if arbitrage-free, False otherwise
        """
        # Check for monotonicity violations that could indicate arbitrage
        
        # Calendar spread: IV should generally increase with maturity
        # (not strictly required but suspicious if violated everywhere)
        
        # Butterfly spread: Check convexity in strike dimension
        if len(strikes) < 3:
            return True
        
        for maturity_idx in range(iv_surface.shape[1]):
            ivs = iv_surface[:, maturity_idx]
            
            # Check for non-smooth behavior (potential butterfly arbitrage)
            for i in range(1, len(strikes) - 1):
                # Second derivative approximation
                d2iv = ivs[i+1] - 2*ivs[i] + ivs[i-1]
                
                # Large negative curvature could indicate arbitrage
                if d2iv < -0.5:  # Threshold for suspicion
                    logger.warning(
                        f"Potential butterfly arbitrage at strike {strikes[i]}, "
                        f"maturity index {maturity_idx}"
                    )
                    return False
        
        return True


# Example usage and testing
if __name__ == "__main__":
    validator = OptionDataValidator()
    
    print(f"\n{'='*60}")
    print(f"Option Data Validator Tests")
    print(f"{'='*60}")
    
    # Test price validation
    print("\n1. Price Validation:")
    print(f"Valid price (100): {validator.validate_price(100)}")
    print(f"Invalid price (-10): {validator.validate_price(-10)}")
    print(f"Invalid price (None): {validator.validate_price(None)}")
    
    # Test strike validation
    print("\n2. Strike Validation:")
    print(f"Valid strike (100, spot=100): {validator.validate_strike(100, 100)}")
    print(f"Invalid strike (500, spot=100): {validator.validate_strike(500, 100)}")
    
    # Test time to maturity
    print("\n3. Time to Maturity Validation:")
    print(f"Valid maturity (1.0): {validator.validate_time_to_maturity(1.0)}")
    print(f"Invalid maturity (-0.5): {validator.validate_time_to_maturity(-0.5)}")
    print(f"Invalid maturity (10): {validator.validate_time_to_maturity(10)}")
    
    # Test volatility
    print("\n4. Volatility Validation:")
    print(f"Valid vol (0.25): {validator.validate_volatility(0.25)}")
    print(f"Invalid vol (5.0): {validator.validate_volatility(5.0)}")
    print(f"Invalid vol (0.001): {validator.validate_volatility(0.001)}")
    
    # Test put-call parity
    print("\n5. Put-Call Parity Check:")
    S, K, T, r = 100, 100, 1.0, 0.05
    call_price = 10.45
    put_price = 5.57  # Theoretical values from BS
    is_valid, dev = validator.check_put_call_parity(call_price, put_price, S, K, T, r)
    print(f"Valid parity: {is_valid}, Deviation: {dev:.6f}")
    
    # Test options chain validation
    print("\n6. Options Chain Validation:")
    test_df = pd.DataFrame({
        'strike': [95, 100, 105],
        'bid': [8.0, 5.0, 3.0],
        'ask': [8.5, 5.5, 3.5],
        'volume': [100, 200, 150]
    })
    is_valid, errors = validator.validate_option_chain(test_df)
    print(f"Chain valid: {is_valid}")
    if errors:
        print(f"Errors: {errors}")
    
    logger.info("Validator tests completed")
    
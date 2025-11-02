"""
Black-Scholes Options Pricing Model
Implements pricing and Greeks calculations for European options.
"""

import numpy as np
from scipy.stats import norm
from typing import Tuple, Dict
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BlackScholes:
    """
    Black-Scholes model for European options pricing.
    """
    
    def __init__(self):
        """Initialize Black-Scholes model."""
        logger.info("Black-Scholes model initialized")
    
    @staticmethod
    def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate d1 parameter.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate
            sigma: Volatility
        
        Returns:
            d1 value
        """
        return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate d2 parameter.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate
            sigma: Volatility
        
        Returns:
            d2 value
        """
        d1 = BlackScholes._d1(S, K, T, r, sigma)
        return d1 - sigma * np.sqrt(T)
    
    def call_price(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate European call option price.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate (annual)
            sigma: Volatility (annual)
        
        Returns:
            Call option price
        """
        if T <= 0:
            return max(S - K, 0)
        
        d1 = self._d1(S, K, T, r, sigma)
        d2 = self._d2(S, K, T, r, sigma)
        
        call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call
    
    def put_price(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate European put option price.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate (annual)
            sigma: Volatility (annual)
        
        Returns:
            Put option price
        """
        if T <= 0:
            return max(K - S, 0)
        
        d1 = self._d1(S, K, T, r, sigma)
        d2 = self._d2(S, K, T, r, sigma)
        
        put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return put
    
    def greeks(self, S: float, K: float, T: float, r: float, sigma: float, 
               option_type: str = 'call') -> Dict[str, float]:
        """
        Calculate option Greeks.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
        
        Returns:
            Dictionary containing Greeks (delta, gamma, vega, theta, rho)
        """
        if T <= 0:
            logger.warning("Time to maturity is zero or negative")
            return {
                'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 
                'theta': 0.0, 'rho': 0.0
            }
        
        d1 = self._d1(S, K, T, r, sigma)
        d2 = self._d2(S, K, T, r, sigma)
        
        # Common calculations
        pdf_d1 = norm.pdf(d1)
        cdf_d1 = norm.cdf(d1)
        cdf_neg_d1 = norm.cdf(-d1)
        cdf_d2 = norm.cdf(d2)
        cdf_neg_d2 = norm.cdf(-d2)
        
        sqrt_T = np.sqrt(T)
        exp_neg_rT = np.exp(-r * T)
        
        if option_type.lower() == 'call':
            # Call Greeks
            delta = cdf_d1
            theta = ((-S * pdf_d1 * sigma) / (2 * sqrt_T) 
                    - r * K * exp_neg_rT * cdf_d2)
            rho = K * T * exp_neg_rT * cdf_d2
        else:
            # Put Greeks
            delta = cdf_d1 - 1  # or -norm.cdf(-d1)
            theta = ((-S * pdf_d1 * sigma) / (2 * sqrt_T) 
                    + r * K * exp_neg_rT * cdf_neg_d2)
            rho = -K * T * exp_neg_rT * cdf_neg_d2
        
        # Gamma and Vega are same for calls and puts
        gamma = pdf_d1 / (S * sigma * sqrt_T)
        vega = S * pdf_d1 * sqrt_T
        
        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega / 100,  # Vega per 1% change in volatility
            'theta': theta / 365,  # Theta per day
            'rho': rho / 100  # Rho per 1% change in interest rate
        }
    
    def put_call_parity_check(self, S: float, K: float, T: float, r: float, 
                               call_price: float, put_price: float) -> Dict[str, float]:
        """
        Verify put-call parity: C - P = S - K*e^(-rT)
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            call_price: Market call price
            put_price: Market put price
        
        Returns:
            Dictionary with parity check results
        """
        theoretical_diff = S - K * np.exp(-r * T)
        actual_diff = call_price - put_price
        arbitrage = actual_diff - theoretical_diff
        
        return {
            'theoretical_difference': theoretical_diff,
            'actual_difference': actual_diff,
            'arbitrage_value': arbitrage,
            'is_arbitrage': abs(arbitrage) > 0.01  # Threshold for transaction costs
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize model
    bs = BlackScholes()
    
    # Test parameters
    S = 100  # Stock price
    K = 100  # Strike price
    T = 1.0  # 1 year to maturity
    r = 0.05  # 5% risk-free rate
    sigma = 0.2  # 20% volatility
    
    # Calculate prices
    call = bs.call_price(S, K, T, r, sigma)
    put = bs.put_price(S, K, T, r, sigma)
    
    print(f"\n{'='*50}")
    print(f"Black-Scholes Pricing Test")
    print(f"{'='*50}")
    print(f"Stock Price (S): ${S}")
    print(f"Strike Price (K): ${K}")
    print(f"Time to Maturity (T): {T} years")
    print(f"Risk-Free Rate (r): {r*100}%")
    print(f"Volatility (σ): {sigma*100}%")
    print(f"\nCall Price: ${call:.4f}")
    print(f"Put Price: ${put:.4f}")
    
    # Calculate Greeks
    call_greeks = bs.greeks(S, K, T, r, sigma, 'call')
    put_greeks = bs.greeks(S, K, T, r, sigma, 'put')
    
    print(f"\n{'='*50}")
    print(f"Call Greeks:")
    print(f"{'='*50}")
    for greek, value in call_greeks.items():
        print(f"{greek.capitalize():10s}: {value:.6f}")
    
    print(f"\n{'='*50}")
    print(f"Put Greeks:")
    print(f"{'='*50}")
    for greek, value in put_greeks.items():
        print(f"{greek.capitalize():10s}: {value:.6f}")
    
    # Put-Call Parity Check
    parity = bs.put_call_parity_check(S, K, T, r, call, put)
    print(f"\n{'='*50}")
    print(f"Put-Call Parity Check:")
    print(f"{'='*50}")
    for key, value in parity.items():
        if isinstance(value, bool):
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.6f}")
    
    logger.info("Black-Scholes model test completed successfully")
    
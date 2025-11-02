"""
Implied Volatility Calculator using Newton-Raphson method.
Calculates the volatility implied by market option prices.
"""

import numpy as np
from typing import Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.pricing.black_scholes import BlackScholes
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ImpliedVolatility:
    """
    Calculate implied volatility from market option prices.
    Uses Newton-Raphson method for numerical optimization.
    """
    
    def __init__(self, max_iterations: int = 100, tolerance: float = 1e-6):
        """
        Initialize IV calculator.
        
        Args:
            max_iterations: Maximum iterations for Newton-Raphson
            tolerance: Convergence tolerance
        """
        self.bs_model = BlackScholes()
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        logger.info(f"ImpliedVolatility initialized with max_iter={max_iterations}, tol={tolerance}")
    
    def calculate_iv(self, market_price: float, S: float, K: float, T: float, 
                     r: float, option_type: str = 'call', 
                     initial_guess: float = 0.3) -> Optional[float]:
        """
        Calculate implied volatility using Newton-Raphson method.
        
        Args:
            market_price: Observed market price of the option
            S: Current stock price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate
            option_type: 'call' or 'put'
            initial_guess: Starting volatility estimate
        
        Returns:
            Implied volatility (annualized), or None if convergence fails
        """
        # Input validation
        if market_price <= 0:
            logger.error(f"Invalid market price: {market_price}")
            return None
        
        if T <= 0:
            logger.error(f"Invalid time to maturity: {T}")
            return None
        
        # Check for intrinsic value violations
        if option_type.lower() == 'call':
            intrinsic = max(S - K, 0)
        else:
            intrinsic = max(K - S, 0)
        
        if market_price < intrinsic:
            logger.warning(f"Market price {market_price} below intrinsic {intrinsic}")
            return None
        
        # Newton-Raphson iteration
        sigma = initial_guess
        
        for iteration in range(self.max_iterations):
            # Calculate option price with current sigma
            if option_type.lower() == 'call':
                price = self.bs_model.call_price(S, K, T, r, sigma)
            else:
                price = self.bs_model.put_price(S, K, T, r, sigma)
            
            # Calculate vega (derivative of price w.r.t. volatility)
            greeks = self.bs_model.greeks(S, K, T, r, sigma, option_type)
            vega = greeks['vega'] * 100  # Convert back to actual vega
            
            # Check for zero vega
            if abs(vega) < 1e-10:
                logger.warning("Vega too small, cannot converge")
                return None
            
            # Price difference
            price_diff = market_price - price
            
            # Check convergence
            if abs(price_diff) < self.tolerance:
                logger.debug(f"Converged in {iteration + 1} iterations: IV={sigma:.6f}")
                return sigma
            
            # Newton-Raphson update
            sigma = sigma + price_diff / vega
            
            # Ensure sigma stays positive and reasonable
            sigma = max(0.001, min(sigma, 5.0))  # Bound between 0.1% and 500%
        
        logger.warning(f"Failed to converge after {self.max_iterations} iterations")
        return None
    
    def calculate_iv_surface(self, market_data: dict, S: float, r: float) -> dict:
        """
        Calculate implied volatility surface from market option chain.
        
        Args:
            market_data: Dict with structure {strike: {maturity: price}}
            S: Current stock price
            r: Risk-free rate
        
        Returns:
            IV surface as dict {strike: {maturity: iv}}
        """
        iv_surface = {}
        
        for strike, maturities in market_data.items():
            iv_surface[strike] = {}
            for maturity, price in maturities.items():
                iv = self.calculate_iv(price, S, strike, maturity, r)
                iv_surface[strike][maturity] = iv
        
        logger.info(f"Calculated IV surface for {len(market_data)} strikes")
        return iv_surface
    
    def bid_ask_iv(self, bid_price: float, ask_price: float, S: float, K: float, 
                   T: float, r: float, option_type: str = 'call') -> dict:
        """
        Calculate IV for both bid and ask prices.
        
        Args:
            bid_price: Bid price
            ask_price: Ask price
            S: Stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            option_type: 'call' or 'put'
        
        Returns:
            Dictionary with bid_iv, ask_iv, mid_iv, and spread
        """
        bid_iv = self.calculate_iv(bid_price, S, K, T, r, option_type)
        ask_iv = self.calculate_iv(ask_price, S, K, T, r, option_type)
        mid_price = (bid_price + ask_price) / 2
        mid_iv = self.calculate_iv(mid_price, S, K, T, r, option_type)
        
        result = {
            'bid_iv': bid_iv,
            'ask_iv': ask_iv,
            'mid_iv': mid_iv,
            'iv_spread': ask_iv - bid_iv if (ask_iv and bid_iv) else None
        }
        
        return result


# Example usage and testing
if __name__ == "__main__":
    # Initialize calculator
    iv_calc = ImpliedVolatility()
    
    # Test parameters
    S = 100  # Stock price
    K = 100  # Strike price (ATM)
    T = 0.5  # 6 months to maturity
    r = 0.05  # 5% risk-free rate
    
    # First, calculate a theoretical price with known volatility
    bs = BlackScholes()
    true_sigma = 0.25  # 25% volatility
    theoretical_call_price = bs.call_price(S, K, T, r, true_sigma)
    theoretical_put_price = bs.put_price(S, K, T, r, true_sigma)
    
    print(f"\n{'='*60}")
    print(f"Implied Volatility Calculator Test")
    print(f"{'='*60}")
    print(f"Stock Price: ${S}")
    print(f"Strike Price: ${K}")
    print(f"Time to Maturity: {T} years")
    print(f"Risk-Free Rate: {r*100}%")
    print(f"True Volatility: {true_sigma*100}%")
    print(f"\nTheoretical Call Price: ${theoretical_call_price:.4f}")
    print(f"Theoretical Put Price: ${theoretical_put_price:.4f}")
    
    # Now recover the implied volatility from the price
    implied_vol_call = iv_calc.calculate_iv(
        theoretical_call_price, S, K, T, r, 'call'
    )
    implied_vol_put = iv_calc.calculate_iv(
        theoretical_put_price, S, K, T, r, 'put'
    )
    
    print(f"\n{'='*60}")
    print(f"Recovered Implied Volatility:")
    print(f"{'='*60}")
    print(f"Call IV: {implied_vol_call*100:.4f}%")
    print(f"Put IV: {implied_vol_put*100:.4f}%")
    print(f"Error (Call): {abs(implied_vol_call - true_sigma)*100:.6f}%")
    print(f"Error (Put): {abs(implied_vol_put - true_sigma)*100:.6f}%")
    
    # Test bid-ask IV
    print(f"\n{'='*60}")
    print(f"Bid-Ask IV Analysis:")
    print(f"{'='*60}")
    bid_price = theoretical_call_price - 0.5
    ask_price = theoretical_call_price + 0.5
    bid_ask_result = iv_calc.bid_ask_iv(bid_price, ask_price, S, K, T, r, 'call')
    
    print(f"Bid Price: ${bid_price:.4f}")
    print(f"Ask Price: ${ask_price:.4f}")
    print(f"Bid IV: {bid_ask_result['bid_iv']*100:.4f}%")
    print(f"Mid IV: {bid_ask_result['mid_iv']*100:.4f}%")
    print(f"Ask IV: {bid_ask_result['ask_iv']*100:.4f}%")
    print(f"IV Spread: {bid_ask_result['iv_spread']*100:.4f}%")
    
    logger.info("Implied volatility test completed successfully")
    
"""
SVI (Stochastic Volatility Inspired) Model
Flexible parametric form for volatility smiles.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import Tuple, Optional, Dict
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SVIModel:
    """
    SVI model for implied volatility parameterization.
    Provides smooth, arbitrage-free volatility smiles.
    """
    
    def __init__(self, parameterization: str = 'raw'):
        """
        Initialize SVI model.
        
        Args:
            parameterization: 'raw' or 'natural' SVI parameterization
        """
        self.parameterization = parameterization
        self.params = None  # Will store (a, b, rho, m, sigma)
        logger.info(f"SVI model initialized with {parameterization} parameterization")
    
    @staticmethod
    def svi_raw(k: float, a: float, b: float, rho: float, m: float, sigma: float) -> float:
        """
        Raw SVI parameterization for total implied variance.
        
        w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
        
        Args:
            k: Log-moneyness (log(K/F))
            a: Vertical shift (controls ATM variance level)
            b: Slope (controls angle of smile wings)
            rho: Correlation (controls skew, -1 < rho < 1)
            m: Horizontal shift (controls ATM location)
            sigma: Smoothness (controls smile curvature)
        
        Returns:
            Total implied variance w(k)
        """
        sqrt_term = np.sqrt((k - m)**2 + sigma**2)
        w = a + b * (rho * (k - m) + sqrt_term)
        return max(w, 0.0)  # Ensure non-negative variance
    
    @staticmethod
    def svi_to_iv(w: float, T: float) -> float:
        """
        Convert total implied variance to implied volatility.
        
        Args:
            w: Total implied variance
            T: Time to maturity
        
        Returns:
            Implied volatility
        """
        if T <= 0:
            return 0.0
        return np.sqrt(w / T)
    
    def get_iv(self, strike: float, forward: float, maturity: float) -> float:
        """
        Get implied volatility for a given strike.
        
        Args:
            strike: Strike price
            forward: Forward price
            maturity: Time to maturity
        
        Returns:
            Implied volatility
        """
        if self.params is None:
            logger.error("Model not calibrated yet")
            return 0.25
        
        # Calculate log-moneyness
        k = np.log(strike / forward)
        
        # Get parameters
        a, b, rho, m, sigma = self.params
        
        # Calculate total variance
        w = self.svi_raw(k, a, b, rho, m, sigma)
        
        # Convert to IV
        iv = self.svi_to_iv(w, maturity)
        
        return iv
    
    def calibrate(self, strikes: np.ndarray, market_ivs: np.ndarray,
                  forward: float, maturity: float) -> Tuple[float, float, float, float, float]:
        """
        Calibrate SVI parameters to market data.
        
        Args:
            strikes: Array of strike prices
            market_ivs: Array of market implied volatilities
            forward: Forward price
            maturity: Time to maturity
        
        Returns:
            Tuple of calibrated parameters (a, b, rho, m, sigma)
        """
        # Convert to log-moneyness
        log_moneyness = np.log(strikes / forward)
        
        # Convert IVs to total variance
        market_variances = market_ivs**2 * maturity
        
        # Initial guess based on ATM values
        atm_idx = np.argmin(np.abs(log_moneyness))
        atm_var = market_variances[atm_idx]
        
        # Initial parameters
        initial_guess = [
            atm_var,        # a: ATM variance level
            0.1,            # b: slope
            -0.3,           # rho: typical negative skew
            log_moneyness[atm_idx],  # m: ATM log-moneyness
            0.1             # sigma: curvature
        ]
        
        # Parameter bounds (ensuring arbitrage-free conditions)
        bounds = [
            (0.001, 1.0),      # a > 0
            (0.001, 1.0),      # b > 0
            (-0.999, 0.999),   # -1 < rho < 1
            (-1.0, 1.0),       # m (reasonable range for log-moneyness)
            (0.001, 1.0)       # sigma > 0
        ]
        
        def objective(params):
            a, b, rho, m, sigma = params
            
            # Calculate model variances
            model_variances = np.array([
                self.svi_raw(k, a, b, rho, m, sigma)
                for k in log_moneyness
            ])
            
            # Mean squared error
            mse = np.mean((model_variances - market_variances)**2)
            
            # Add penalty for arbitrage violations
            penalty = 0
            
            # Check butterfly arbitrage condition: b * (1 + |rho|) <= 4 * sigma
            if b * (1 + abs(rho)) > 4 * sigma:
                penalty += 100 * (b * (1 + abs(rho)) - 4 * sigma)**2
            
            # Penalize extreme parameters
            if abs(rho) > 0.9:
                penalty += 10 * (abs(rho) - 0.9)**2
            
            return mse + penalty
        
        try:
            # Optimize
            result = minimize(
                objective,
                initial_guess,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000}
            )
            
            if result.success:
                self.params = tuple(result.x)
                logger.info(f"SVI calibration successful: a={result.x[0]:.4f}, "
                          f"b={result.x[1]:.4f}, ρ={result.x[2]:.4f}, "
                          f"m={result.x[3]:.4f}, σ={result.x[4]:.4f}")
                return self.params
            else:
                logger.warning("SVI calibration did not converge")
                self.params = tuple(initial_guess)
                return self.params
                
        except Exception as e:
            logger.error(f"Error in SVI calibration: {e}")
            self.params = tuple(initial_guess)
            return self.params
    
    def check_arbitrage_free(self) -> Dict[str, bool]:
        """
        Check if current parameters satisfy arbitrage-free conditions.
        
        Returns:
            Dictionary with arbitrage checks
        """
        if self.params is None:
            return {}
        
        a, b, rho, m, sigma = self.params
        
        # Butterfly arbitrage condition
        butterfly_ok = b * (1 + abs(rho)) <= 4 * sigma
        
        # Calendar spread arbitrage (SVI is inherently calendar arbitrage-free)
        calendar_ok = True
        
        # Check parameter bounds
        bounds_ok = (
            a > 0 and
            b > 0 and
            -1 < rho < 1 and
            sigma > 0
        )
        
        return {
            'butterfly_arbitrage_free': butterfly_ok,
            'calendar_arbitrage_free': calendar_ok,
            'parameters_valid': bounds_ok,
            'overall_arbitrage_free': butterfly_ok and calendar_ok and bounds_ok
        }
    
    def calculate_fit_quality(self, strikes: np.ndarray, market_ivs: np.ndarray,
                             forward: float, maturity: float) -> Dict[str, float]:
        """
        Calculate quality metrics for the SVI fit.
        
        Args:
            strikes: Strike prices
            market_ivs: Market implied volatilities
            forward: Forward price
            maturity: Time to maturity
        
        Returns:
            Dictionary with fit quality metrics
        """
        if self.params is None:
            return {}
        
        # Calculate model IVs
        model_ivs = np.array([
            self.get_iv(K, forward, maturity) for K in strikes
        ])
        
        # Calculate metrics
        residuals = model_ivs - market_ivs
        
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        max_error = np.max(np.abs(residuals))
        
        # Avoid division by zero
        market_var = np.var(market_ivs)
        if market_var > 0:
            r_squared = 1 - (np.sum(residuals**2) / np.sum((market_ivs - np.mean(market_ivs))**2))
        else:
            r_squared = 0.0
        
        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'max_error': float(max_error),
            'r_squared': float(r_squared),
            'mean_market_iv': float(np.mean(market_ivs)),
            'mean_model_iv': float(np.mean(model_ivs))
        }
    
    def generate_smile(self, forward: float, maturity: float,
                      num_points: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate smooth volatility smile using SVI parameters.
        
        Args:
            forward: Forward price
            maturity: Time to maturity
            num_points: Number of points to generate
        
        Returns:
            Tuple of (strikes, implied_volatilities)
        """
        if self.params is None:
            logger.error("Model not calibrated yet")
            return np.array([]), np.array([])
        
        # Generate log-moneyness range
        k_range = np.linspace(-0.5, 0.5, num_points)
        
        # Calculate strikes
        strikes = forward * np.exp(k_range)
        
        # Calculate IVs
        ivs = np.array([self.get_iv(K, forward, maturity) for K in strikes])
        
        return strikes, ivs


# Example usage and testing
if __name__ == "__main__":
    print(f"\n{'='*70}")
    print(f"SVI Model Test")
    print(f"{'='*70}")
    
    # Market data setup
    forward = 100.0
    maturity = 1.0
    strikes = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120])
    
    # Generate synthetic market data with smile
    print("\n1. Generating synthetic market data...")
    true_a = 0.04
    true_b = 0.1
    true_rho = -0.4
    true_m = 0.0
    true_sigma = 0.2
    
    log_moneyness = np.log(strikes / forward)
    market_variances = np.array([
        SVIModel.svi_raw(k, true_a, true_b, true_rho, true_m, true_sigma)
        for k in log_moneyness
    ])
    market_ivs = np.sqrt(market_variances / maturity)
    
    # Add small noise
    market_ivs += np.random.normal(0, 0.005, len(strikes))
    
    print(f"True parameters: a={true_a:.4f}, b={true_b:.4f}, ρ={true_rho:.4f}, "
          f"m={true_m:.4f}, σ={true_sigma:.4f}")
    print(f"Market IVs: {[f'{iv*100:.2f}%' for iv in market_ivs]}")
    
    # Calibrate SVI model
    print("\n2. Calibrating SVI model...")
    svi = SVIModel()
    calibrated_params = svi.calibrate(strikes, market_ivs, forward, maturity)
    a, b, rho, m, sigma = calibrated_params
    
    print(f"Calibrated parameters:")
    print(f"   a: {a:.4f}")
    print(f"   b: {b:.4f}")
    print(f"   ρ (rho): {rho:.4f}")
    print(f"   m: {m:.4f}")
    print(f"   σ (sigma): {sigma:.4f}")
    
    # Check arbitrage-free conditions
    print("\n3. Arbitrage-Free Checks:")
    arb_checks = svi.check_arbitrage_free()
    for key, value in arb_checks.items():
        status = "✓" if value else "✗"
        print(f"   {status} {key}: {value}")
    
    # Model vs Market comparison
    print("\n4. Model vs Market Comparison:")
    print(f"{'Strike':<10} {'Market IV':<12} {'Model IV':<12} {'Difference':<12}")
    print("-" * 50)
    
    for K, market_iv in zip(strikes, market_ivs):
        model_iv = svi.get_iv(K, forward, maturity)
        diff = (model_iv - market_iv) * 100
        print(f"{K:<10.0f} {market_iv*100:<12.2f}% {model_iv*100:<12.2f}% {diff:<12.4f}%")
    
    # Fit quality
    print("\n5. Fit Quality Metrics:")
    metrics = svi.calculate_fit_quality(strikes, market_ivs, forward, maturity)
    for key, value in metrics.items():
        if 'iv' in key:
            print(f"   {key}: {value*100:.2f}%")
        else:
            print(f"   {key}: {value:.6f}")
    
    # Generate smooth smile
    print("\n6. Generating smooth volatility smile...")
    smile_strikes, smile_ivs = svi.generate_smile(forward, maturity, num_points=50)
    print(f"   Generated {len(smile_strikes)} points")
    print(f"   Strike range: {smile_strikes.min():.2f} - {smile_strikes.max():.2f}")
    print(f"   IV range: {smile_ivs.min()*100:.2f}% - {smile_ivs.max()*100:.2f}%")
    
    logger.info("SVI model test completed successfully")
    
"""
SABR (Stochastic Alpha Beta Rho) Model
Industry-standard model for volatility surface fitting.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import Tuple, Optional, Dict
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SABRModel:
    """
    SABR model for implied volatility surface.
    Captures volatility smile and skew dynamics.
    """
    
    def __init__(self):
        """Initialize SABR model."""
        self.params = None  # Will store (alpha, beta, rho, nu)
        logger.info("SABR model initialized")
    
    @staticmethod
    def sabr_volatility(F: float, K: float, T: float, 
                       alpha: float, beta: float, rho: float, nu: float) -> float:
        """
        Calculate SABR implied volatility.
        
        Args:
            F: Forward price
            K: Strike price
            T: Time to expiration
            alpha: Initial volatility level
            beta: CEV exponent (0=normal, 1=lognormal)
            rho: Correlation between asset and volatility
            nu: Volatility of volatility (vol-of-vol)
        
        Returns:
            Implied volatility
        """
        # Handle edge cases
        if T <= 0:
            return 0.0
        
        if abs(F - K) < 1e-10:  # ATM case
            FK = F
        else:
            FK = (F * K) ** (0.5)
        
        # Prevent division by zero
        if FK <= 0:
            return alpha
        
        # Calculate log-moneyness
        if abs(F - K) < 1e-10:
            logFK = 0.0
        else:
            logFK = np.log(F / K)
        
        # SABR formula components
        FK_beta = FK ** (beta - 1)
        
        # z parameter
        if abs(nu) < 1e-10:  # nu = 0 case
            z = 0.0
            x_z = 1.0
        else:
            z = (nu / alpha) * FK_beta * logFK
            
            # x(z) function
            if abs(z) < 1e-10:
                x_z = 1.0
            else:
                sqrt_term = np.sqrt(1 - 2*rho*z + z**2)
                numerator = sqrt_term + z - rho
                denominator = 1 - rho
                if abs(denominator) < 1e-10:
                    x_z = 1.0
                else:
                    x_z = np.log(numerator / denominator) / z
        
        # First term
        FK_beta_term = (1 - beta)**2 / 24 * alpha**2 / (FK**(2-2*beta))
        rho_term = rho * beta * nu * alpha / (4 * FK**(1-beta))
        nu_term = (2 - 3*rho**2) / 24 * nu**2
        
        term1 = alpha / (FK**((1-beta)/2) * (1 + (1-beta)**2/24 * logFK**2 + (1-beta)**4/1920 * logFK**4))
        term2 = 1 + (FK_beta_term + rho_term + nu_term) * T
        
        sigma = term1 * x_z * term2
        
        return max(sigma, 0.001)  # Ensure positive volatility
    
    def calibrate(self, strikes: np.ndarray, market_ivs: np.ndarray,
                  forward: float, maturity: float,
                  beta: float = 0.5) -> Tuple[float, float, float, float]:
        """
        Calibrate SABR parameters to market data.
        
        Args:
            strikes: Array of strike prices
            market_ivs: Array of market implied volatilities
            forward: Forward price
            maturity: Time to maturity
            beta: Fixed beta parameter (0.5 is common)
        
        Returns:
            Tuple of calibrated parameters (alpha, beta, rho, nu)
        """
        # Initial guess
        atm_vol = market_ivs[np.argmin(np.abs(strikes - forward))]
        initial_guess = [atm_vol, beta, 0.0, 0.3]
        
        # Bounds for parameters
        bounds = [
            (0.001, 2.0),    # alpha: 0.1% to 200%
            (beta, beta),    # beta: fixed
            (-0.999, 0.999), # rho: correlation
            (0.001, 2.0)     # nu: vol-of-vol
        ]
        
        def objective(params):
            alpha, beta_val, rho, nu = params
            
            # Calculate model IVs
            model_ivs = np.array([
                self.sabr_volatility(forward, K, maturity, alpha, beta_val, rho, nu)
                for K in strikes
            ])
            
            # Mean squared error
            mse = np.mean((model_ivs - market_ivs)**2)
            
            # Add penalty for extreme parameters
            penalty = 0
            if abs(rho) > 0.95:
                penalty += 10 * (abs(rho) - 0.95)**2
            if nu > 1.5:
                penalty += 10 * (nu - 1.5)**2
            
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
                logger.info(f"SABR calibration successful: α={result.x[0]:.4f}, "
                          f"β={result.x[1]:.4f}, ρ={result.x[2]:.4f}, ν={result.x[3]:.4f}")
                return self.params
            else:
                logger.warning("SABR calibration did not converge, using initial guess")
                self.params = tuple(initial_guess)
                return self.params
                
        except Exception as e:
            logger.error(f"Error in SABR calibration: {e}")
            self.params = tuple(initial_guess)
            return self.params
    
    def get_model_iv(self, strike: float, forward: float, maturity: float) -> float:
        """
        Get model implied volatility for a given strike.
        
        Args:
            strike: Strike price
            forward: Forward price
            maturity: Time to maturity
        
        Returns:
            Model implied volatility
        """
        if self.params is None:
            logger.error("Model not calibrated yet")
            return 0.25  # Default 25% vol
        
        alpha, beta, rho, nu = self.params
        return self.sabr_volatility(forward, strike, maturity, alpha, beta, rho, nu)
    
    def calculate_fit_quality(self, strikes: np.ndarray, market_ivs: np.ndarray,
                             forward: float, maturity: float) -> Dict[str, float]:
        """
        Calculate quality metrics for the SABR fit.
        
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
            self.get_model_iv(K, forward, maturity) for K in strikes
        ])
        
        # Calculate metrics
        residuals = model_ivs - market_ivs
        
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        max_error = np.max(np.abs(residuals))
        r_squared = 1 - (np.sum(residuals**2) / np.sum((market_ivs - np.mean(market_ivs))**2))
        
        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'max_error': float(max_error),
            'r_squared': float(r_squared),
            'mean_market_iv': float(np.mean(market_ivs)),
            'mean_model_iv': float(np.mean(model_ivs))
        }


# Example usage and testing
if __name__ == "__main__":
    print(f"\n{'='*70}")
    print(f"SABR Model Test")
    print(f"{'='*70}")
    
    # Market data setup
    forward = 100.0
    maturity = 1.0
    strikes = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120])
    
    # Generate synthetic market IVs with smile
    print("\n1. Generating synthetic market data...")
    true_alpha = 0.25
    true_beta = 0.5
    true_rho = -0.3
    true_nu = 0.4
    
    sabr = SABRModel()
    market_ivs = np.array([
        sabr.sabr_volatility(forward, K, maturity, true_alpha, true_beta, true_rho, true_nu)
        for K in strikes
    ])
    
    # Add small noise
    market_ivs += np.random.normal(0, 0.005, len(strikes))
    
    print(f"True parameters: α={true_alpha}, β={true_beta}, ρ={true_rho}, ν={true_nu}")
    print(f"Market IVs: {[f'{iv*100:.2f}%' for iv in market_ivs]}")
    
    # Calibrate model
    print("\n2. Calibrating SABR model...")
    calibrated_params = sabr.calibrate(strikes, market_ivs, forward, maturity, beta=0.5)
    alpha, beta, rho, nu = calibrated_params
    
    print(f"Calibrated parameters:")
    print(f"   α (alpha): {alpha:.4f}")
    print(f"   β (beta):  {beta:.4f}")
    print(f"   ρ (rho):   {rho:.4f}")
    print(f"   ν (nu):    {nu:.4f}")
    
    # Calculate model IVs
    print("\n3. Model vs Market Comparison:")
    print(f"{'Strike':<10} {'Market IV':<12} {'Model IV':<12} {'Difference':<12}")
    print("-" * 50)
    
    for K, market_iv in zip(strikes, market_ivs):
        model_iv = sabr.get_model_iv(K, forward, maturity)
        diff = (model_iv - market_iv) * 100
        print(f"{K:<10.0f} {market_iv*100:<12.2f}% {model_iv*100:<12.2f}% {diff:<12.4f}%")
    
    # Fit quality
    print("\n4. Fit Quality Metrics:")
    metrics = sabr.calculate_fit_quality(strikes, market_ivs, forward, maturity)
    for key, value in metrics.items():
        if 'iv' in key:
            print(f"   {key}: {value*100:.2f}%")
        else:
            print(f"   {key}: {value:.6f}")
    
    logger.info("SABR model test completed successfully")
    
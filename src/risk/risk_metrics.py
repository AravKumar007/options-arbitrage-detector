"""
Risk Metrics Calculator
Calculate portfolio risk metrics and Value at Risk (VaR).
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from scipy import stats
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RiskMetrics:
    """
    Calculate comprehensive risk metrics for options portfolio.
    """
    
    def __init__(self):
        """Initialize risk metrics calculator."""
        logger.info("RiskMetrics initialized")
    
    def calculate_var(self, returns: np.ndarray, confidence_level: float = 0.95,
                     method: str = 'historical') -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns: Array of returns
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            method: 'historical', 'parametric', or 'monte_carlo'
        
        Returns:
            VaR value (positive number representing potential loss)
        """
        if len(returns) == 0:
            return 0.0
        
        if method == 'historical':
            # Historical VaR: percentile of actual returns
            var = -np.percentile(returns, (1 - confidence_level) * 100)
            
        elif method == 'parametric':
            # Parametric VaR: assumes normal distribution
            mean = np.mean(returns)
            std = np.std(returns)
            z_score = stats.norm.ppf(1 - confidence_level)
            var = -(mean + z_score * std)
            
        else:  # monte_carlo
            # Monte Carlo VaR: simulate from fitted distribution
            mean = np.mean(returns)
            std = np.std(returns)
            simulated_returns = np.random.normal(mean, std, 10000)
            var = -np.percentile(simulated_returns, (1 - confidence_level) * 100)
        
        return max(var, 0)
    
    def calculate_cvar(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.
        
        Average loss beyond VaR threshold.
        
        Args:
            returns: Array of returns
            confidence_level: Confidence level
        
        Returns:
            CVaR value
        """
        if len(returns) == 0:
            return 0.0
        
        var = self.calculate_var(returns, confidence_level, method='historical')
        
        # Get returns worse than VaR
        threshold = -var
        tail_returns = returns[returns <= threshold]
        
        if len(tail_returns) > 0:
            cvar = -np.mean(tail_returns)
        else:
            cvar = var
        
        return cvar
    
    def calculate_sharpe_ratio(self, returns: np.ndarray, 
                              risk_free_rate: float = 0.05) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Array of returns
            risk_free_rate: Annual risk-free rate
        
        Returns:
            Sharpe ratio
        """
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        
        return sharpe
    
    def calculate_sortino_ratio(self, returns: np.ndarray,
                                risk_free_rate: float = 0.05,
                                target_return: float = 0.0) -> float:
        """
        Calculate Sortino ratio (uses downside deviation).
        
        Args:
            returns: Array of returns
            risk_free_rate: Annual risk-free rate
            target_return: Minimum acceptable return
        
        Returns:
            Sortino ratio
        """
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        
        # Downside deviation (only negative returns)
        downside_returns = excess_returns[excess_returns < target_return]
        
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0
        
        downside_std = np.std(downside_returns)
        sortino = np.mean(excess_returns) / downside_std * np.sqrt(252)
        
        return sortino
    
    def calculate_max_drawdown(self, equity_curve: np.ndarray) -> Dict[str, float]:
        """
        Calculate maximum drawdown and related metrics.
        
        Args:
            equity_curve: Array of portfolio values over time
        
        Returns:
            Dictionary with drawdown metrics
        """
        if len(equity_curve) == 0:
            return {'max_drawdown': 0, 'max_drawdown_pct': 0, 'drawdown_duration': 0}
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(equity_curve)
        
        # Calculate drawdown
        drawdown = (equity_curve - running_max) / running_max
        
        # Maximum drawdown
        max_dd = np.min(drawdown)
        max_dd_idx = np.argmin(drawdown)
        
        # Find peak before max drawdown
        peak_idx = np.argmax(equity_curve[:max_dd_idx+1]) if max_dd_idx > 0 else 0
        
        # Duration in days (assuming daily data)
        dd_duration = max_dd_idx - peak_idx
        
        return {
            'max_drawdown': abs(max_dd),
            'max_drawdown_pct': abs(max_dd) * 100,
            'drawdown_duration': dd_duration,
            'peak_idx': peak_idx,
            'trough_idx': max_dd_idx
        }
    
    def calculate_volatility(self, returns: np.ndarray, 
                            annualize: bool = True) -> float:
        """
        Calculate volatility (standard deviation of returns).
        
        Args:
            returns: Array of returns
            annualize: Whether to annualize (assumes daily returns)
        
        Returns:
            Volatility
        """
        if len(returns) == 0:
            return 0.0
        
        vol = np.std(returns)
        
        if annualize:
            vol *= np.sqrt(252)  # Annualize daily returns
        
        return vol
    
    def calculate_greeks_risk(self, portfolio_greeks: Dict[str, float],
                             spot_move: float = 0.01,
                             vol_move: float = 0.01) -> Dict[str, float]:
        """
        Calculate risk from Greeks exposure.
        
        Args:
            portfolio_greeks: Dictionary with portfolio Greeks
            spot_move: Spot price move (1% default)
            vol_move: Volatility move (1% default)
        
        Returns:
            Dictionary with risk estimates
        """
        delta = portfolio_greeks.get('delta', 0)
        gamma = portfolio_greeks.get('gamma', 0)
        vega = portfolio_greeks.get('vega', 0)
        theta = portfolio_greeks.get('theta', 0)
        
        # Delta risk (first-order spot risk)
        delta_risk = delta * spot_move * 100  # Assuming spot = 100
        
        # Gamma risk (second-order spot risk)
        gamma_risk = 0.5 * gamma * (spot_move * 100) ** 2
        
        # Vega risk (volatility risk)
        vega_risk = vega * vol_move
        
        # Theta risk (time decay per day)
        theta_risk = theta
        
        # Total spot risk (delta + gamma)
        total_spot_risk = delta_risk + gamma_risk
        
        return {
            'delta_risk': delta_risk,
            'gamma_risk': gamma_risk,
            'total_spot_risk': total_spot_risk,
            'vega_risk': vega_risk,
            'theta_risk': theta_risk
        }
    
    def calculate_portfolio_metrics(self, returns: np.ndarray,
                                    equity_curve: np.ndarray,
                                    risk_free_rate: float = 0.05) -> Dict:
        """
        Calculate comprehensive portfolio risk metrics.
        
        Args:
            returns: Array of returns
            equity_curve: Portfolio value over time
            risk_free_rate: Annual risk-free rate
        
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # Return metrics
        if len(returns) > 0:
            metrics['total_return'] = (equity_curve[-1] - equity_curve[0]) / equity_curve[0] * 100
            metrics['avg_daily_return'] = np.mean(returns) * 100
            metrics['annualized_return'] = np.mean(returns) * 252 * 100
        
        # Risk metrics
        metrics['volatility'] = self.calculate_volatility(returns)
        metrics['var_95'] = self.calculate_var(returns, 0.95)
        metrics['cvar_95'] = self.calculate_cvar(returns, 0.95)
        
        # Risk-adjusted metrics
        metrics['sharpe_ratio'] = self.calculate_sharpe_ratio(returns, risk_free_rate)
        metrics['sortino_ratio'] = self.calculate_sortino_ratio(returns, risk_free_rate)
        
        # Drawdown metrics
        dd_metrics = self.calculate_max_drawdown(equity_curve)
        metrics.update(dd_metrics)
        
        # Win rate (if available)
        if len(returns) > 0:
            winning_days = (returns > 0).sum()
            metrics['win_rate'] = (winning_days / len(returns)) * 100
        
        return metrics
    
    def check_risk_limits(self, portfolio_greeks: Dict[str, float],
                         limits: Dict[str, float]) -> Dict[str, bool]:
        """
        Check if portfolio Greeks are within risk limits.
        
        Args:
            portfolio_greeks: Current portfolio Greeks
            limits: Dictionary with limit values
        
        Returns:
            Dictionary indicating which limits are breached
        """
        breaches = {}
        
        for greek, value in portfolio_greeks.items():
            limit = limits.get(f'max_{greek}', float('inf'))
            breaches[greek] = abs(value) > limit
        
        return breaches


# Example usage and testing
if __name__ == "__main__":
    print(f"\n{'='*70}")
    print(f"Risk Metrics Calculator Test")
    print(f"{'='*70}")
    
    # Generate sample returns
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 252)  # Daily returns for 1 year
    
    # Generate equity curve
    initial_capital = 100000
    equity_curve = initial_capital * np.cumprod(1 + returns)
    
    # Initialize calculator
    risk = RiskMetrics()
    
    # Calculate VaR
    print("\n1. Value at Risk:")
    var_hist = risk.calculate_var(returns, 0.95, method='historical')
    var_param = risk.calculate_var(returns, 0.95, method='parametric')
    print(f"   Historical VaR (95%): ${var_hist*initial_capital:.2f}")
    print(f"   Parametric VaR (95%): ${var_param*initial_capital:.2f}")
    
    # Calculate CVaR
    cvar = risk.calculate_cvar(returns, 0.95)
    print(f"\n2. Conditional VaR (95%): ${cvar*initial_capital:.2f}")
    
    # Sharpe and Sortino ratios
    print("\n3. Risk-Adjusted Returns:")
    sharpe = risk.calculate_sharpe_ratio(returns)
    sortino = risk.calculate_sortino_ratio(returns)
    print(f"   Sharpe Ratio: {sharpe:.2f}")
    print(f"   Sortino Ratio: {sortino:.2f}")
    
    # Max drawdown
    print("\n4. Drawdown Analysis:")
    dd = risk.calculate_max_drawdown(equity_curve)
    print(f"   Max Drawdown: {dd['max_drawdown_pct']:.2f}%")
    print(f"   Duration: {dd['drawdown_duration']} days")
    
    # Volatility
    print("\n5. Volatility:")
    vol = risk.calculate_volatility(returns)
    print(f"   Annualized Volatility: {vol*100:.2f}%")
    
    # Greeks risk
    print("\n6. Greeks Risk Analysis:")
    portfolio_greeks = {
        'delta': 150,
        'gamma': 20,
        'vega': 500,
        'theta': -50
    }
    greeks_risk = risk.calculate_greeks_risk(portfolio_greeks)
    for key, value in greeks_risk.items():
        print(f"   {key}: ${value:.2f}")
    
    # Comprehensive metrics
    print("\n7. Comprehensive Portfolio Metrics:")
    metrics = risk.calculate_portfolio_metrics(returns, equity_curve)
    for key, value in metrics.items():
        if 'pct' in key or 'return' in key or 'rate' in key:
            print(f"   {key}: {value:.2f}%")
        elif 'ratio' in key:
            print(f"   {key}: {value:.2f}")
        elif 'var' in key.lower():
            print(f"   {key}: ${value*initial_capital:.2f}")
        elif 'duration' in key or 'idx' in key:
            print(f"   {key}: {value}")
        else:
            print(f"   {key}: {value:.2f}")
    
    # Check risk limits
    print("\n8. Risk Limit Checks:")
    limits = {
        'max_delta': 200,
        'max_gamma': 30,
        'max_vega': 1000,
        'max_theta': 100
    }
    breaches = risk.check_risk_limits(portfolio_greeks, limits)
    for greek, is_breached in breaches.items():
        status = "⚠ BREACH" if is_breached else "✓ OK"
        print(f"   {greek}: {status}")
    
    logger.info("Risk metrics calculator test completed successfully")
    
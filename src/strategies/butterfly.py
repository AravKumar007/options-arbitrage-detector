"""
Butterfly Spread Strategy
Identifies arbitrage opportunities in butterfly spreads.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.pricing.black_scholes import BlackScholes
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ButterflySpreadStrategy:
    """
    Detect and analyze butterfly spread arbitrage opportunities.
    
    Butterfly Spread: 
    - Long 1 lower strike
    - Short 2 middle strike
    - Long 1 higher strike
    
    Profits from low volatility and price staying near middle strike.
    """
    
    def __init__(self, transaction_cost: float = 0.01):
        """
        Initialize butterfly spread strategy.
        
        Args:
            transaction_cost: Transaction cost as fraction of price
        """
        self.transaction_cost = transaction_cost
        self.bs_model = BlackScholes()
        logger.info(f"ButterflySpreadStrategy initialized")
    
    def find_butterfly_spreads(self, df: pd.DataFrame, 
                              min_profit: float = 0.5) -> pd.DataFrame:
        """
        Find butterfly spread arbitrage opportunities.
        
        Args:
            df: DataFrame with options data
            min_profit: Minimum profit threshold
        
        Returns:
            DataFrame with butterfly spread opportunities
        """
        opportunities = []
        
        # Group by expiration and option type
        for (expiration, opt_type), group in df.groupby(['expiration', 'option_type']):
            # Need at least 3 different strikes
            strikes = sorted(group['strike'].unique())
            
            if len(strikes) < 3:
                continue
            
            # Check all combinations of 3 strikes
            for i in range(len(strikes) - 2):
                for j in range(i + 1, len(strikes) - 1):
                    for k in range(j + 1, len(strikes)):
                        K1, K2, K3 = strikes[i], strikes[j], strikes[k]
                        
                        # For true butterfly, K2 should be roughly midpoint
                        # But we check all combinations for arbitrage
                        
                        # Get options at these strikes
                        opt_K1 = group[group['strike'] == K1].iloc[0]
                        opt_K2 = group[group['strike'] == K2].iloc[0]
                        opt_K3 = group[group['strike'] == K3].iloc[0]
                        
                        # Butterfly: Buy K1, Sell 2xK2, Buy K3
                        buy_K1 = (opt_K1['bid'] + opt_K1['ask']) / 2
                        sell_K2 = (opt_K2['bid'] + opt_K2['ask']) / 2
                        buy_K3 = (opt_K3['bid'] + opt_K3['ask']) / 2
                        
                        # Net cost (debit)
                        net_cost = buy_K1 - 2*sell_K2 + buy_K3
                        
                        # Transaction costs
                        trans_cost = self.transaction_cost * (buy_K1 + 2*sell_K2 + buy_K3)
                        
                        # Maximum payoff occurs when spot = K2 at expiration
                        max_payoff = K2 - K1
                        
                        # Check for arbitrage
                        # 1. Net cost should be positive (debit spread)
                        # 2. Net cost should be less than max payoff
                        # 3. Profit after costs should be positive
                        
                        if net_cost <= 0:
                            # Negative cost = arbitrage! (getting paid to enter)
                            profit = abs(net_cost) - trans_cost
                            arb_type = "negative_cost"
                            
                            if profit > min_profit:
                                opportunities.append({
                                    'strike_low': K1,
                                    'strike_mid': K2,
                                    'strike_high': K3,
                                    'expiration': expiration,
                                    'option_type': opt_type,
                                    'cost_K1': buy_K1,
                                    'cost_K2': sell_K2,
                                    'cost_K3': buy_K3,
                                    'net_cost': net_cost,
                                    'max_payoff': max_payoff,
                                    'profit': profit,
                                    'arbitrage_type': arb_type,
                                    'signal_time': datetime.now()
                                })
                        
                        elif net_cost > max_payoff:
                            # Cost exceeds max payoff = potential arbitrage (mispricing)
                            # Reverse butterfly: Sell K1, Buy 2xK2, Sell K3
                            reverse_income = -net_cost  # Income from reverse position
                            reverse_max_loss = max_payoff
                            profit = reverse_income - reverse_max_loss - trans_cost
                            arb_type = "overpriced"
                            
                            if profit > min_profit:
                                opportunities.append({
                                    'strike_low': K1,
                                    'strike_mid': K2,
                                    'strike_high': K3,
                                    'expiration': expiration,
                                    'option_type': opt_type,
                                    'cost_K1': buy_K1,
                                    'cost_K2': sell_K2,
                                    'cost_K3': buy_K3,
                                    'net_cost': net_cost,
                                    'max_payoff': max_payoff,
                                    'profit': profit,
                                    'arbitrage_type': arb_type,
                                    'recommendation': 'sell_butterfly',
                                    'signal_time': datetime.now()
                                })
                        
                        else:
                            # Normal butterfly - check if profit potential exists
                            max_profit = max_payoff - net_cost - trans_cost
                            
                            if max_profit > min_profit:
                                # Calculate break-even points
                                be_lower = K1 + net_cost
                                be_upper = K3 - net_cost
                                
                                opportunities.append({
                                    'strike_low': K1,
                                    'strike_mid': K2,
                                    'strike_high': K3,
                                    'expiration': expiration,
                                    'option_type': opt_type,
                                    'cost_K1': buy_K1,
                                    'cost_K2': sell_K2,
                                    'cost_K3': buy_K3,
                                    'net_cost': net_cost,
                                    'max_payoff': max_payoff,
                                    'max_profit': max_profit,
                                    'break_even_lower': be_lower,
                                    'break_even_upper': be_upper,
                                    'arbitrage_type': 'normal',
                                    'recommendation': 'buy_butterfly',
                                    'signal_time': datetime.now()
                                })
        
        result_df = pd.DataFrame(opportunities)
        
        if not result_df.empty:
            result_df = result_df.sort_values('profit' if 'profit' in result_df.columns else 'max_profit', 
                                             ascending=False)
            logger.info(f"Found {len(result_df)} butterfly spread opportunities")
        
        return result_df
    
    def check_convexity_arbitrage(self, df: pd.DataFrame, spot_price: float) -> pd.DataFrame:
        """
        Check for convexity violations that indicate butterfly arbitrage.
        
        Option prices should be convex in strike. Violations = arbitrage.
        
        Args:
            df: DataFrame with options data
            spot_price: Current spot price
        
        Returns:
            DataFrame with convexity violations
        """
        violations = []
        
        # Group by expiration and option type
        for (expiration, opt_type), group in df.groupby(['expiration', 'option_type']):
            # Sort by strike
            sorted_group = group.sort_values('strike').reset_index(drop=True)
            
            if len(sorted_group) < 3:
                continue
            
            # Check convexity for each triplet
            for i in range(len(sorted_group) - 2):
                K1 = sorted_group.loc[i, 'strike']
                K2 = sorted_group.loc[i+1, 'strike']
                K3 = sorted_group.loc[i+2, 'strike']
                
                C1 = (sorted_group.loc[i, 'bid'] + sorted_group.loc[i, 'ask']) / 2
                C2 = (sorted_group.loc[i+1, 'bid'] + sorted_group.loc[i+1, 'ask']) / 2
                C3 = (sorted_group.loc[i+2, 'bid'] + sorted_group.loc[i+2, 'ask']) / 2
                
                # Convexity condition: C2 <= (C1*w + C3*(1-w))
                # where w = (K3-K2)/(K3-K1)
                
                if K3 != K1:
                    w = (K3 - K2) / (K3 - K1)
                    interpolated_price = C1 * w + C3 * (1 - w)
                    
                    # Check violation
                    violation_amount = C2 - interpolated_price
                    
                    if violation_amount > 0.01:  # Significant violation
                        # This means butterfly can be set up for arbitrage
                        butterfly_cost = C1 - 2*C2 + C3
                        
                        violations.append({
                            'strike_low': K1,
                            'strike_mid': K2,
                            'strike_high': K3,
                            'expiration': expiration,
                            'option_type': opt_type,
                            'price_K1': C1,
                            'price_K2': C2,
                            'price_K3': C3,
                            'interpolated_price': interpolated_price,
                            'violation_amount': violation_amount,
                            'butterfly_cost': butterfly_cost,
                            'arbitrage_profit': -butterfly_cost if butterfly_cost < 0 else 0,
                            'signal_time': datetime.now()
                        })
        
        result_df = pd.DataFrame(violations)
        
        if not result_df.empty:
            logger.info(f"Found {len(result_df)} convexity violations")
        
        return result_df
    
    def calculate_butterfly_greeks(self, K1: float, K2: float, K3: float,
                                   spot_price: float, T: float, r: float,
                                   sigma: float, option_type: str) -> Dict[str, float]:
        """
        Calculate Greeks for butterfly spread position.
        
        Args:
            K1, K2, K3: Strike prices (low, mid, high)
            spot_price: Current spot price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
        
        Returns:
            Dictionary with net Greeks
        """
        # Long K1
        greeks_K1 = self.bs_model.greeks(spot_price, K1, T, r, sigma, option_type)
        
        # Short 2xK2
        greeks_K2 = self.bs_model.greeks(spot_price, K2, T, r, sigma, option_type)
        
        # Long K3
        greeks_K3 = self.bs_model.greeks(spot_price, K3, T, r, sigma, option_type)
        
        # Net Greeks: +K1 - 2*K2 + K3
        net_greeks = {
            'net_delta': greeks_K1['delta'] - 2*greeks_K2['delta'] + greeks_K3['delta'],
            'net_gamma': greeks_K1['gamma'] - 2*greeks_K2['gamma'] + greeks_K3['gamma'],
            'net_vega': greeks_K1['vega'] - 2*greeks_K2['vega'] + greeks_K3['vega'],
            'net_theta': greeks_K1['theta'] - 2*greeks_K2['theta'] + greeks_K3['theta'],
            'net_rho': greeks_K1['rho'] - 2*greeks_K2['rho'] + greeks_K3['rho']
        }
        
        return net_greeks
    
    def analyze_profitability(self, butterfly_spread: Dict, 
                             spot_price: float) -> Dict[str, float]:
        """
        Analyze profitability at different price levels.
        
        Args:
            butterfly_spread: Butterfly spread details
            spot_price: Current spot price
        
        Returns:
            Dictionary with profit analysis
        """
        K1 = butterfly_spread['strike_low']
        K2 = butterfly_spread['strike_mid']
        K3 = butterfly_spread['strike_high']
        net_cost = butterfly_spread['net_cost']
        
        # Calculate P&L at various spot prices at expiration
        spot_range = np.linspace(K1 - 5, K3 + 5, 50)
        pnl_values = []
        
        for S in spot_range:
            if butterfly_spread['option_type'] == 'call':
                payoff_K1 = max(S - K1, 0)
                payoff_K2 = max(S - K2, 0)
                payoff_K3 = max(S - K3, 0)
            else:  # put
                payoff_K1 = max(K1 - S, 0)
                payoff_K2 = max(K2 - S, 0)
                payoff_K3 = max(K3 - S, 0)
            
            # Net payoff: +K1 - 2*K2 + K3
            net_payoff = payoff_K1 - 2*payoff_K2 + payoff_K3
            pnl = net_payoff - net_cost
            pnl_values.append(pnl)
        
        pnl_array = np.array(pnl_values)
        
        return {
            'max_profit': np.max(pnl_array),
            'max_loss': np.min(pnl_array),
            'profit_at_current_spot': np.interp(spot_price, spot_range, pnl_array),
            'optimal_spot': spot_range[np.argmax(pnl_array)],
            'profit_probability': (pnl_array > 0).sum() / len(pnl_array) * 100
        }


# Example usage and testing
if __name__ == "__main__":
    print(f"\n{'='*70}")
    print(f"Butterfly Spread Strategy Test")
    print(f"{'='*70}")
    
    # Create sample options data
    sample_data = []
    strikes = [95, 100, 105]
    spot = 100
    
    for strike in strikes:
        # Call options
        if strike == 95:
            mid_price = 8.0
        elif strike == 100:
            mid_price = 5.0
        else:
            mid_price = 3.0
        
        sample_data.append({
            'strike': strike,
            'expiration': '2024-06-20',
            'time_to_maturity': 0.5,
            'option_type': 'call',
            'bid': mid_price - 0.3,
            'ask': mid_price + 0.3,
            'impliedVolatility': 0.25
        })
    
    df = pd.DataFrame(sample_data)
    
    # Initialize strategy
    strategy = ButterflySpreadStrategy(transaction_cost=0.01)
    
    # Find opportunities
    print("\n1. Finding butterfly spread opportunities...")
    opportunities = strategy.find_butterfly_spreads(df, min_profit=0.1)
    
    if not opportunities.empty:
        print(f"\nFound {len(opportunities)} opportunities:")
        display_cols = ['strike_low', 'strike_mid', 'strike_high', 'net_cost', 
                       'max_payoff', 'recommendation']
        available_cols = [col for col in display_cols if col in opportunities.columns]
        print(opportunities[available_cols].to_string(index=False))
    else:
        print("No opportunities found")
    
    # Check convexity
    print("\n2. Checking for convexity violations...")
    violations = strategy.check_convexity_arbitrage(df, spot)
    
    if not violations.empty:
        print(f"Found {len(violations)} convexity violations")
        print(violations[['strike_low', 'strike_mid', 'strike_high', 
                         'violation_amount', 'butterfly_cost']].to_string(index=False))
    else:
        print("No convexity violations found")
    
    # Calculate Greeks
    print("\n3. Butterfly Greeks Analysis:")
    greeks = strategy.calculate_butterfly_greeks(95, 100, 105, spot, 0.5, 0.05, 0.25, 'call')
    for greek, value in greeks.items():
        print(f"   {greek}: {value:.6f}")
    
    # Profit analysis
    if not opportunities.empty:
        print("\n4. Profitability Analysis:")
        best_spread = opportunities.iloc[0].to_dict()
        profit_analysis = strategy.analyze_profitability(best_spread, spot)
        for key, value in profit_analysis.items():
            if 'probability' in key or 'spot' in key:
                print(f"   {key}: {value:.2f}")
            else:
                print(f"   {key}: ${value:.2f}")
    
    logger.info("Butterfly spread strategy test completed successfully")
    
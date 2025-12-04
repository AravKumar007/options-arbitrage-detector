"""
Calendar Spread Strategy
Identifies arbitrage opportunities in calendar spreads (time spreads).
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


class CalendarSpreadStrategy:
    """
    Detect and analyze calendar spread arbitrage opportunities.
    
    Calendar Spread: Buy longer-dated option, sell shorter-dated option (same strike).
    Profits from time decay and volatility changes.
    """
    
    def __init__(self, transaction_cost: float = 0.01):
        """
        Initialize calendar spread strategy.
        
        Args:
            transaction_cost: Transaction cost as fraction of price
        """
        self.transaction_cost = transaction_cost
        self.bs_model = BlackScholes()
        logger.info(f"CalendarSpreadStrategy initialized")
    
    def find_calendar_spreads(self, df: pd.DataFrame, 
                             min_profit: float = 0.5) -> pd.DataFrame:
        """
        Find calendar spread opportunities.
        
        Args:
            df: DataFrame with options data
            min_profit: Minimum profit threshold
        
        Returns:
            DataFrame with calendar spread opportunities
        """
        opportunities = []
        
        # Group by strike and option type
        grouped = df.groupby(['strike', 'option_type'])
        
        for (strike, opt_type), group in grouped:
            # Need at least 2 different expirations
            if len(group['expiration'].unique()) < 2:
                continue
            
            # Sort by expiration
            sorted_group = group.sort_values('time_to_maturity')
            
            # Check all pairs (near-term vs far-term)
            for i in range(len(sorted_group)):
                for j in range(i + 1, len(sorted_group)):
                    near_term = sorted_group.iloc[i]
                    far_term = sorted_group.iloc[j]
                    
                    # Calendar spread: Sell near-term, Buy far-term
                    near_premium = (near_term['bid'] + near_term['ask']) / 2
                    far_cost = (far_term['bid'] + far_term['ask']) / 2
                    
                    # Initial cost (debit spread)
                    initial_cost = far_cost - near_premium
                    
                    # Transaction costs
                    trans_cost = self.transaction_cost * (near_premium + far_cost)
                    
                    # Check if spread makes sense
                    if initial_cost <= 0:
                        continue  # No cost or credit - unusual
                    
                    # Estimate profit potential
                    # At near-term expiration, near option expires worthless
                    # Far option retains time value
                    
                    T_near = near_term['time_to_maturity']
                    T_far = far_term['time_to_maturity']
                    time_diff = T_far - T_near
                    
                    # Estimate far option value at near expiration
                    # Using same IV assumption
                    iv = far_term.get('impliedVolatility', 0.25)
                    
                    # Simple estimation: far option will have ~sqrt(T_remaining/T_original) of value
                    if T_far > 0:
                        value_retention = np.sqrt(time_diff / T_far)
                        estimated_far_value = far_cost * value_retention
                    else:
                        estimated_far_value = 0
                    
                    # Expected profit
                    expected_profit = estimated_far_value - initial_cost - trans_cost
                    
                    # Return on investment
                    roi = (expected_profit / initial_cost) * 100 if initial_cost > 0 else 0
                    
                    if expected_profit > min_profit:
                        opportunities.append({
                            'strike': strike,
                            'option_type': opt_type,
                            'near_expiration': near_term['expiration'],
                            'far_expiration': far_term['expiration'],
                            'near_maturity': T_near,
                            'far_maturity': T_far,
                            'time_difference': time_diff,
                            'near_premium': near_premium,
                            'far_cost': far_cost,
                            'initial_cost': initial_cost,
                            'estimated_value': estimated_far_value,
                            'expected_profit': expected_profit,
                            'roi_percent': roi,
                            'near_iv': near_term.get('impliedVolatility', np.nan),
                            'far_iv': far_term.get('impliedVolatility', np.nan),
                            'iv_differential': far_term.get('impliedVolatility', 0) - near_term.get('impliedVolatility', 0),
                            'signal_time': datetime.now()
                        })
        
        result_df = pd.DataFrame(opportunities)
        
        if not result_df.empty:
            result_df = result_df.sort_values('expected_profit', ascending=False)
            logger.info(f"Found {len(result_df)} calendar spread opportunities")
        
        return result_df
    
    def analyze_vol_term_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze volatility term structure for calendar spread signals.
        
        Args:
            df: DataFrame with options data
        
        Returns:
            DataFrame with term structure analysis
        """
        analysis = []
        
        # Group by strike and option type
        for (strike, opt_type), group in df.groupby(['strike', 'option_type']):
            # Sort by maturity
            sorted_group = group.sort_values('time_to_maturity')
            
            if len(sorted_group) < 2:
                continue
            
            # Check if term structure is inverted (bearish signal)
            maturities = sorted_group['time_to_maturity'].values
            ivs = sorted_group['impliedVolatility'].values
            
            # Remove NaN
            valid_mask = ~np.isnan(ivs)
            maturities = maturities[valid_mask]
            ivs = ivs[valid_mask]
            
            if len(ivs) < 2:
                continue
            
            # Calculate term structure slope
            # Positive slope = normal (far > near)
            # Negative slope = inverted (near > far) - unusual, potential arb
            
            slope = (ivs[-1] - ivs[0]) / (maturities[-1] - maturities[0]) if maturities[-1] > maturities[0] else 0
            
            # Count inversions
            inversions = 0
            for i in range(len(ivs) - 1):
                if ivs[i] > ivs[i+1]:
                    inversions += 1
            
            analysis.append({
                'strike': strike,
                'option_type': opt_type,
                'num_expirations': len(ivs),
                'near_iv': ivs[0],
                'far_iv': ivs[-1],
                'iv_spread': ivs[-1] - ivs[0],
                'term_structure_slope': slope,
                'inversions': inversions,
                'is_inverted': slope < 0,
                'signal': 'inverted' if slope < -0.01 else 'normal' if slope > 0.01 else 'flat'
            })
        
        result_df = pd.DataFrame(analysis)
        
        if not result_df.empty:
            inverted = result_df[result_df['is_inverted']]
            if len(inverted) > 0:
                logger.info(f"Found {len(inverted)} inverted term structures (potential arbitrage)")
        
        return result_df
    
    def calculate_greeks_pnl(self, near_option: Dict, far_option: Dict,
                            spot_price: float, r: float) -> Dict[str, float]:
        """
        Calculate Greeks exposure for calendar spread position.
        
        Args:
            near_option: Dict with near-term option details
            far_option: Dict with far-term option details
            spot_price: Current spot price
            r: Risk-free rate
        
        Returns:
            Dictionary with net Greeks
        """
        # Near option (short position)
        near_greeks = self.bs_model.greeks(
            spot_price,
            near_option['strike'],
            near_option['time_to_maturity'],
            r,
            near_option.get('implied_vol', 0.25),
            near_option['option_type']
        )
        
        # Far option (long position)
        far_greeks = self.bs_model.greeks(
            spot_price,
            far_option['strike'],
            far_option['time_to_maturity'],
            r,
            far_option.get('implied_vol', 0.25),
            far_option['option_type']
        )
        
        # Net Greeks (long far - short near)
        net_greeks = {
            'net_delta': far_greeks['delta'] - near_greeks['delta'],
            'net_gamma': far_greeks['gamma'] - near_greeks['gamma'],
            'net_vega': far_greeks['vega'] - near_greeks['vega'],
            'net_theta': far_greeks['theta'] - near_greeks['theta'],
            'net_rho': far_greeks['rho'] - near_greeks['rho']
        }
        
        return net_greeks
    
    def recommend_adjustments(self, spread_position: Dict) -> List[str]:
        """
        Recommend position adjustments based on market conditions.
        
        Args:
            spread_position: Current calendar spread position
        
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Check profitability
        if spread_position.get('expected_profit', 0) < 0:
            recommendations.append("CLOSE: Position shows negative expected profit")
        
        # Check IV spread
        iv_diff = spread_position.get('iv_differential', 0)
        if iv_diff < 0:
            recommendations.append("WARNING: Inverted IV term structure detected")
        
        # Check time to expiration
        near_maturity = spread_position.get('near_maturity', 0)
        if near_maturity < 0.05:  # Less than ~18 days
            recommendations.append("MONITOR: Near-term expiration approaching")
        
        # Check ROI
        roi = spread_position.get('roi_percent', 0)
        if roi > 20:
            recommendations.append("STRONG: High ROI potential")
        elif roi > 10:
            recommendations.append("MODERATE: Acceptable ROI")
        else:
            recommendations.append("WEAK: Low ROI potential")
        
        return recommendations


# Example usage and testing
if __name__ == "__main__":
    print(f"\n{'='*70}")
    print(f"Calendar Spread Strategy Test")
    print(f"{'='*70}")
    
    # Create sample options data with multiple expirations
    sample_data = []
    strike = 100
    spot = 100
    
    # Near-term (1 month)
    sample_data.append({
        'strike': strike,
        'expiration': '2024-05-20',
        'time_to_maturity': 1/12,
        'option_type': 'call',
        'bid': 4.8,
        'ask': 5.2,
        'impliedVolatility': 0.30
    })
    
    # Mid-term (3 months)
    sample_data.append({
        'strike': strike,
        'expiration': '2024-07-20',
        'time_to_maturity': 3/12,
        'option_type': 'call',
        'bid': 7.8,
        'ask': 8.2,
        'impliedVolatility': 0.28
    })
    
    # Far-term (6 months)
    sample_data.append({
        'strike': strike,
        'expiration': '2024-10-20',
        'time_to_maturity': 6/12,
        'option_type': 'call',
        'bid': 10.8,
        'ask': 11.2,
        'impliedVolatility': 0.26
    })
    
    df = pd.DataFrame(sample_data)
    
    # Initialize strategy
    strategy = CalendarSpreadStrategy(transaction_cost=0.01)
    
    # Find opportunities
    print("\n1. Finding calendar spread opportunities...")
    opportunities = strategy.find_calendar_spreads(df, min_profit=0.1)
    
    if not opportunities.empty:
        print(f"\nFound {len(opportunities)} opportunities:")
        print(opportunities[['strike', 'near_expiration', 'far_expiration', 
                           'initial_cost', 'expected_profit', 'roi_percent']].to_string(index=False))
    else:
        print("No opportunities found")
    
    # Analyze term structure
    print("\n2. Volatility Term Structure Analysis:")
    term_analysis = strategy.analyze_vol_term_structure(df)
    print(term_analysis[['strike', 'near_iv', 'far_iv', 'term_structure_slope', 'signal']].to_string(index=False))
    
    # Get recommendations
    if not opportunities.empty:
        print("\n3. Position Recommendations:")
        best_spread = opportunities.iloc[0].to_dict()
        recommendations = strategy.recommend_adjustments(best_spread)
        for rec in recommendations:
            print(f"   • {rec}")
    
    logger.info("Calendar spread strategy test completed successfully")
    
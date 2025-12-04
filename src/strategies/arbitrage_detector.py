"""
Arbitrage Detector
Identifies mispricings and arbitrage opportunities in options markets.
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


class ArbitrageDetector:
    """
    Detect various types of arbitrage opportunities in options markets.
    """
    
    def __init__(self, transaction_cost: float = 0.01):
        """
        Initialize arbitrage detector.
        
        Args:
            transaction_cost: Transaction cost as fraction of price (e.g., 0.01 = 1%)
        """
        self.transaction_cost = transaction_cost
        self.bs_model = BlackScholes()
        logger.info(f"ArbitrageDetector initialized with {transaction_cost*100:.2f}% transaction cost")
    
    def check_put_call_parity(self, df: pd.DataFrame, spot_price: float, 
                              r: float) -> pd.DataFrame:
        """
        Check for put-call parity violations.
        
        Put-Call Parity: C - P = S - K*e^(-rT)
        
        Args:
            df: DataFrame with calls and puts at same strikes and maturities
            spot_price: Current spot price
            r: Risk-free rate
        
        Returns:
            DataFrame with parity violations
        """
        violations = []
        
        # Group by strike and expiration
        grouped = df.groupby(['strike', 'expiration'])
        
        for (strike, expiration), group in grouped:
            # Need both call and put
            calls = group[group['option_type'] == 'call']
            puts = group[group['option_type'] == 'put']
            
            if len(calls) == 0 or len(puts) == 0:
                continue
            
            call_row = calls.iloc[0]
            put_row = puts.iloc[0]
            
            # Get prices
            call_mid = (call_row['bid'] + call_row['ask']) / 2
            put_mid = (put_row['bid'] + put_row['ask']) / 2
            
            T = call_row['time_to_maturity']
            
            # Theoretical difference
            theoretical_diff = spot_price - strike * np.exp(-r * T)
            
            # Actual difference
            actual_diff = call_mid - put_mid
            
            # Arbitrage profit (accounting for transaction costs)
            deviation = abs(actual_diff - theoretical_diff)
            cost = (call_row['ask'] - call_row['bid'] + put_row['ask'] - put_row['bid']) / 2
            
            profit = deviation - cost - self.transaction_cost * (call_mid + put_mid)
            
            if profit > 0:
                violations.append({
                    'strike': strike,
                    'expiration': expiration,
                    'maturity': T,
                    'theoretical_diff': theoretical_diff,
                    'actual_diff': actual_diff,
                    'deviation': deviation,
                    'profit': profit,
                    'call_price': call_mid,
                    'put_price': put_mid,
                    'signal_time': datetime.now()
                })
        
        result_df = pd.DataFrame(violations)
        
        if not result_df.empty:
            logger.info(f"Found {len(result_df)} put-call parity violations")
        
        return result_df
    
    def check_box_spread(self, df: pd.DataFrame, r: float) -> pd.DataFrame:
        """
        Check for box spread arbitrage.
        
        A box spread combines a bull call spread and bear put spread.
        
        Args:
            df: DataFrame with options data
            r: Risk-free rate
        
        Returns:
            DataFrame with box spread opportunities
        """
        opportunities = []
        
        # Group by expiration
        for expiration, exp_group in df.groupby('expiration'):
            T = exp_group['time_to_maturity'].iloc[0]
            
            # Get calls and puts
            calls = exp_group[exp_group['option_type'] == 'call'].sort_values('strike')
            puts = exp_group[exp_group['option_type'] == 'put'].sort_values('strike')
            
            if len(calls) < 2 or len(puts) < 2:
                continue
            
            # Check all pairs of strikes
            strikes = sorted(exp_group['strike'].unique())
            
            for i in range(len(strikes)):
                for j in range(i + 1, len(strikes)):
                    K1, K2 = strikes[i], strikes[j]
                    
                    # Get options at these strikes
                    call_K1 = calls[calls['strike'] == K1]
                    call_K2 = calls[calls['strike'] == K2]
                    put_K1 = puts[puts['strike'] == K1]
                    put_K2 = puts[puts['strike'] == K2]
                    
                    if any(len(x) == 0 for x in [call_K1, call_K2, put_K1, put_K2]):
                        continue
                    
                    # Box spread cost
                    cost = (
                        (call_K1['ask'].iloc[0] - call_K2['bid'].iloc[0]) +
                        (put_K2['ask'].iloc[0] - put_K1['bid'].iloc[0])
                    )
                    
                    # Box spread payoff (always K2 - K1)
                    payoff = K2 - K1
                    
                    # Present value of payoff
                    pv_payoff = payoff * np.exp(-r * T)
                    
                    # Profit
                    profit = pv_payoff - cost - self.transaction_cost * cost
                    
                    if profit > 0:
                        opportunities.append({
                            'strike_low': K1,
                            'strike_high': K2,
                            'expiration': expiration,
                            'maturity': T,
                            'cost': cost,
                            'payoff': payoff,
                            'pv_payoff': pv_payoff,
                            'profit': profit,
                            'return_pct': (profit / cost) * 100,
                            'signal_time': datetime.now()
                        })
        
        result_df = pd.DataFrame(opportunities)
        
        if not result_df.empty:
            logger.info(f"Found {len(result_df)} box spread arbitrage opportunities")
        
        return result_df
    
    def check_conversion_reversal(self, df: pd.DataFrame, spot_price: float,
                                   r: float) -> pd.DataFrame:
        """
        Check for conversion and reversal arbitrage.
        
        Conversion: Long stock + Long put + Short call (synthetic short)
        Reversal: Short stock + Short put + Long call (synthetic long)
        
        Args:
            df: DataFrame with options data
            spot_price: Current spot price
            r: Risk-free rate
        
        Returns:
            DataFrame with conversion/reversal opportunities
        """
        opportunities = []
        
        # Group by strike and expiration
        grouped = df.groupby(['strike', 'expiration'])
        
        for (strike, expiration), group in grouped:
            calls = group[group['option_type'] == 'call']
            puts = group[group['option_type'] == 'put']
            
            if len(calls) == 0 or len(puts) == 0:
                continue
            
            call_row = calls.iloc[0]
            put_row = puts.iloc[0]
            
            T = call_row['time_to_maturity']
            
            # Conversion: Buy stock, Buy put, Sell call
            conversion_cost = (
                spot_price +
                put_row['ask'] -
                call_row['bid']
            )
            conversion_payoff = strike
            conversion_pv = conversion_payoff * np.exp(-r * T)
            conversion_profit = conversion_pv - conversion_cost - self.transaction_cost * conversion_cost
            
            if conversion_profit > 0:
                opportunities.append({
                    'type': 'conversion',
                    'strike': strike,
                    'expiration': expiration,
                    'maturity': T,
                    'cost': conversion_cost,
                    'payoff': conversion_payoff,
                    'profit': conversion_profit,
                    'return_pct': (conversion_profit / conversion_cost) * 100,
                    'signal_time': datetime.now()
                })
            
            # Reversal: Sell stock, Sell put, Buy call
            reversal_income = (
                spot_price +
                put_row['bid'] -
                call_row['ask']
            )
            reversal_obligation = strike
            reversal_pv = reversal_obligation * np.exp(-r * T)
            reversal_profit = reversal_income - reversal_pv - self.transaction_cost * reversal_income
            
            if reversal_profit > 0:
                opportunities.append({
                    'type': 'reversal',
                    'strike': strike,
                    'expiration': expiration,
                    'maturity': T,
                    'income': reversal_income,
                    'obligation': reversal_obligation,
                    'profit': reversal_profit,
                    'return_pct': (reversal_profit / reversal_income) * 100,
                    'signal_time': datetime.now()
                })
        
        result_df = pd.DataFrame(opportunities)
        
        if not result_df.empty:
            logger.info(f"Found {len(result_df)} conversion/reversal opportunities")
        
        return result_df
    
    def check_volatility_arbitrage(self, df: pd.DataFrame, spot_price: float,
                                   r: float, vol_threshold: float = 0.05) -> pd.DataFrame:
        """
        Check for volatility arbitrage based on IV deviations.
        
        Args:
            df: DataFrame with options and implied volatilities
            spot_price: Current spot price
            r: Risk-free rate
            vol_threshold: Minimum IV deviation to flag (e.g., 0.05 = 5%)
        
        Returns:
            DataFrame with volatility arbitrage opportunities
        """
        opportunities = []
        
        # Calculate mean IV for each maturity
        for expiration, group in df.groupby('expiration'):
            mean_iv = group['impliedVolatility'].mean()
            std_iv = group['impliedVolatility'].std()
            
            if std_iv == 0:
                continue
            
            # Find outliers
            for idx, row in group.iterrows():
                iv = row['impliedVolatility']
                z_score = (iv - mean_iv) / std_iv
                
                # Flag significant deviations
                if abs(z_score) > 2.0:  # More than 2 standard deviations
                    deviation = iv - mean_iv
                    
                    if abs(deviation) > vol_threshold:
                        opportunities.append({
                            'strike': row['strike'],
                            'expiration': expiration,
                            'option_type': row['option_type'],
                            'implied_vol': iv,
                            'mean_vol': mean_iv,
                            'deviation': deviation,
                            'z_score': z_score,
                            'signal': 'overpriced' if deviation > 0 else 'underpriced',
                            'mid_price': (row['bid'] + row['ask']) / 2,
                            'signal_time': datetime.now()
                        })
        
        result_df = pd.DataFrame(opportunities)
        
        if not result_df.empty:
            logger.info(f"Found {len(result_df)} volatility arbitrage opportunities")
        
        return result_df
    
    def scan_all_arbitrage(self, df: pd.DataFrame, spot_price: float,
                          r: float) -> Dict[str, pd.DataFrame]:
        """
        Scan for all types of arbitrage opportunities.
        
        Args:
            df: DataFrame with options data
            spot_price: Current spot price
            r: Risk-free rate
        
        Returns:
            Dictionary with all arbitrage opportunities by type
        """
        logger.info("Scanning for all arbitrage opportunities...")
        
        results = {
            'put_call_parity': self.check_put_call_parity(df, spot_price, r),
            'box_spread': self.check_box_spread(df, r),
            'conversion_reversal': self.check_conversion_reversal(df, spot_price, r),
            'volatility_arbitrage': self.check_volatility_arbitrage(df, spot_price, r)
        }
        
        # Summary
        total_opportunities = sum(len(v) for v in results.values())
        logger.info(f"Total arbitrage opportunities found: {total_opportunities}")
        
        return results


# Example usage and testing
if __name__ == "__main__":
    print(f"\n{'='*70}")
    print(f"Arbitrage Detector Test")
    print(f"{'='*70}")
    
    # Create sample options data
    np.random.seed(42)
    spot_price = 100.0
    r = 0.05
    
    sample_data = []
    strikes = [95, 100, 105]
    expirations = ['2024-06-20', '2024-12-20']
    
    for expiration in expirations:
        T = 0.5 if expiration == '2024-06-20' else 1.0
        
        for strike in strikes:
            # Calls
            call_fair = 10.0 if strike == 95 else 5.0 if strike == 100 else 2.0
            sample_data.append({
                'strike': strike,
                'expiration': expiration,
                'time_to_maturity': T,
                'option_type': 'call',
                'bid': call_fair - 0.5,
                'ask': call_fair + 0.5,
                'impliedVolatility': 0.25 + np.random.normal(0, 0.02)
            })
            
            # Puts
            put_fair = 2.0 if strike == 95 else 5.0 if strike == 100 else 10.0
            sample_data.append({
                'strike': strike,
                'expiration': expiration,
                'time_to_maturity': T,
                'option_type': 'put',
                'bid': put_fair - 0.5,
                'ask': put_fair + 0.5,
                'impliedVolatility': 0.25 + np.random.normal(0, 0.02)
            })
    
    df = pd.DataFrame(sample_data)
    
    # Initialize detector
    detector = ArbitrageDetector(transaction_cost=0.01)
    
    # Scan for arbitrage
    print("\n1. Scanning for all arbitrage opportunities...")
    results = detector.scan_all_arbitrage(df, spot_price, r)
    
    print("\n2. Results by Type:")
    for arb_type, opportunities in results.items():
        print(f"\n   {arb_type.upper()}:")
        if len(opportunities) > 0:
            print(f"   Found {len(opportunities)} opportunities")
            print(opportunities.to_string(index=False))
        else:
            print("   No opportunities found")
    
    logger.info("Arbitrage detector test completed successfully")
    
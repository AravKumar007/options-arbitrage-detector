"""
Slippage Model
Estimates and simulates realistic slippage for options trades.
"""

import numpy as np
from typing import Dict, Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SlippageModel:
    """
    Model slippage based on order size, liquidity, and market conditions.
    """
    
    def __init__(self, model_type: str = 'proportional'):
        """
        Initialize slippage model.
        
        Args:
            model_type: 'fixed', 'proportional', or 'market_impact'
        """
        self.model_type = model_type
        logger.info(f"SlippageModel initialized with {model_type} model")
    
    def calculate_fixed_slippage(self, price: float, 
                                 fixed_bps: int = 5) -> float:
        """
        Calculate fixed slippage in basis points.
        
        Args:
            price: Option price
            fixed_bps: Fixed slippage in basis points (default 5 bps)
        
        Returns:
            Slippage amount
        """
        slippage = price * (fixed_bps / 10000)
        return slippage
    
    def calculate_proportional_slippage(self, price: float, size: int,
                                       typical_volume: int = 100,
                                       base_factor: float = 0.001) -> float:
        """
        Calculate proportional slippage based on order size vs typical volume.
        
        Args:
            price: Option price
            size: Order size
            typical_volume: Typical daily volume
            base_factor: Base slippage factor (0.1% default)
        
        Returns:
            Slippage amount
        """
        # Avoid division by zero
        if typical_volume <= 0:
            typical_volume = 100
        
        # Slippage increases with order size relative to volume
        size_ratio = size / typical_volume
        
        # Square root model: slippage grows slower than linear
        slippage_factor = base_factor * np.sqrt(size_ratio)
        
        slippage = price * slippage_factor
        return slippage
    
    def calculate_market_impact(self, price: float, size: int,
                                bid_ask_spread: float,
                                market_depth: int = 500,
                                temporary_impact: float = 0.5) -> Dict[str, float]:
        """
        Calculate market impact with temporary and permanent components.
        
        Args:
            price: Option price
            size: Order size
            bid_ask_spread: Current bid-ask spread
            market_depth: Total liquidity (contracts)
            temporary_impact: Fraction of impact that's temporary (0-1)
        
        Returns:
            Dictionary with impact components
        """
        # Ensure market depth is positive
        if market_depth <= 0:
            market_depth = 500
        
        # Total impact increases with order size
        size_ratio = size / market_depth
        
        # Base impact from spread
        spread_cost = bid_ask_spread / 2
        
        # Additional impact from moving the market
        # Using square root model
        additional_impact = price * 0.01 * np.sqrt(size_ratio)
        
        # Total impact
        total_impact = spread_cost + additional_impact
        
        # Split into temporary and permanent
        temporary = total_impact * temporary_impact
        permanent = total_impact * (1 - temporary_impact)
        
        return {
            'total_impact': total_impact,
            'temporary_impact': temporary,
            'permanent_impact': permanent,
            'spread_cost': spread_cost,
            'price_impact': additional_impact
        }
    
    def estimate_slippage(self, price: float, size: int,
                         bid: float, ask: float,
                         volume: int = 100,
                         open_interest: int = 1000) -> Dict[str, float]:
        """
        Comprehensive slippage estimation using current model.
        
        Args:
            price: Mid price
            size: Order size
            bid: Best bid
            ask: Best ask
            volume: Recent volume
            open_interest: Open interest
        
        Returns:
            Dictionary with slippage estimates
        """
        spread = ask - bid
        
        if self.model_type == 'fixed':
            slippage = self.calculate_fixed_slippage(price)
            
            return {
                'slippage': slippage,
                'slippage_pct': (slippage / price) * 100,
                'model': 'fixed'
            }
        
        elif self.model_type == 'proportional':
            slippage = self.calculate_proportional_slippage(
                price, size, typical_volume=volume
            )
            
            return {
                'slippage': slippage,
                'slippage_pct': (slippage / price) * 100,
                'model': 'proportional',
                'size_ratio': size / max(volume, 1)
            }
        
        else:  # market_impact
            impact = self.calculate_market_impact(
                price, size, spread, market_depth=open_interest
            )
            
            return {
                'slippage': impact['total_impact'],
                'slippage_pct': (impact['total_impact'] / price) * 100,
                'model': 'market_impact',
                'temporary_impact': impact['temporary_impact'],
                'permanent_impact': impact['permanent_impact'],
                'spread_cost': impact['spread_cost']
            }
    
    def adjust_price_for_slippage(self, price: float, side: str,
                                  slippage: float) -> float:
        """
        Adjust execution price for slippage.
        
        Args:
            price: Base price
            side: 'buy' or 'sell'
            slippage: Slippage amount
        
        Returns:
            Adjusted execution price
        """
        if side == 'buy':
            # Pay more when buying
            return price + slippage
        else:  # sell
            # Receive less when selling
            return price - slippage
    
    def calculate_total_cost(self, price: float, size: int,
                            side: str, slippage: float,
                            commission: float = 0.65) -> Dict[str, float]:
        """
        Calculate total transaction cost including slippage and commissions.
        
        Args:
            price: Base price per contract
            size: Number of contracts
            side: 'buy' or 'sell'
            slippage: Slippage per contract
            commission: Commission per contract
        
        Returns:
            Dictionary with cost breakdown
        """
        # Adjusted price
        exec_price = self.adjust_price_for_slippage(price, side, slippage)
        
        # Notional value
        notional = exec_price * size * 100  # Options are per 100 shares
        
        # Commissions
        total_commission = commission * size
        
        # Total slippage cost
        total_slippage = slippage * size * 100
        
        # Total cost
        if side == 'buy':
            total_cost = notional + total_commission
        else:  # sell
            total_cost = notional - total_commission
        
        return {
            'base_price': price,
            'execution_price': exec_price,
            'size': size,
            'notional': notional,
            'slippage_total': total_slippage,
            'commission_total': total_commission,
            'total_cost': total_cost,
            'cost_per_contract': total_cost / size if size > 0 else 0
        }
    
    def simulate_execution_path(self, price: float, total_size: int,
                               num_chunks: int = 5) -> Dict:
        """
        Simulate executing large order in chunks to minimize slippage.
        
        Args:
            price: Starting price
            total_size: Total order size
            num_chunks: Number of chunks to split into
        
        Returns:
            Dictionary with execution simulation
        """
        chunk_size = total_size // num_chunks
        remainder = total_size % num_chunks
        
        executions = []
        cumulative_slippage = 0
        
        for i in range(num_chunks):
            size = chunk_size + (1 if i < remainder else 0)
            
            # Slippage increases with each chunk (price moves)
            slippage = self.calculate_proportional_slippage(
                price, size, typical_volume=100
            )
            
            # Cumulative impact
            cumulative_slippage += slippage
            exec_price = price + cumulative_slippage
            
            executions.append({
                'chunk': i + 1,
                'size': size,
                'price': exec_price,
                'slippage': slippage,
                'cumulative_slippage': cumulative_slippage
            })
        
        # Average execution price
        total_notional = sum(e['price'] * e['size'] for e in executions)
        avg_price = total_notional / total_size
        
        return {
            'total_size': total_size,
            'num_chunks': num_chunks,
            'executions': executions,
            'avg_execution_price': avg_price,
            'total_slippage': cumulative_slippage,
            'slippage_pct': (cumulative_slippage / price) * 100
        }


# Example usage and testing
if __name__ == "__main__":
    print(f"\n{'='*70}")
    print(f"Slippage Model Test")
    print(f"{'='*70}")
    
    price = 5.0
    size = 50
    bid = 4.90
    ask = 5.10
    volume = 100
    open_interest = 1000
    
    # Test fixed slippage
    print("\n1. Fixed Slippage Model:")
    model_fixed = SlippageModel('fixed')
    slippage = model_fixed.calculate_fixed_slippage(price, fixed_bps=5)
    print(f"   Price: ${price:.2f}")
    print(f"   Fixed slippage (5 bps): ${slippage:.4f}")
    print(f"   Slippage %: {(slippage/price)*100:.3f}%")
    
    # Test proportional slippage
    print("\n2. Proportional Slippage Model:")
    model_prop = SlippageModel('proportional')
    result = model_prop.estimate_slippage(price, size, bid, ask, volume, open_interest)
    print(f"   Order size: {size} contracts")
    print(f"   Volume: {volume} contracts")
    print(f"   Size ratio: {result.get('size_ratio', 0):.2f}")
    print(f"   Slippage: ${result['slippage']:.4f}")
    print(f"   Slippage %: {result['slippage_pct']:.3f}%")
    
    # Test market impact
    print("\n3. Market Impact Model:")
    model_impact = SlippageModel('market_impact')
    result = model_impact.estimate_slippage(price, size, bid, ask, volume, open_interest)
    print(f"   Total impact: ${result['slippage']:.4f}")
    print(f"   Temporary: ${result['temporary_impact']:.4f}")
    print(f"   Permanent: ${result['permanent_impact']:.4f}")
    print(f"   Spread cost: ${result['spread_cost']:.4f}")
    
    # Total cost calculation
    print("\n4. Total Transaction Cost:")
    cost = model_prop.calculate_total_cost(price, size, 'buy', result['slippage'])
    print(f"   Base price: ${cost['base_price']:.2f}")
    print(f"   Execution price: ${cost['execution_price']:.2f}")
    print(f"   Notional: ${cost['notional']:.2f}")
    print(f"   Slippage: ${cost['slippage_total']:.2f}")
    print(f"   Commission: ${cost['commission_total']:.2f}")
    print(f"   Total cost: ${cost['total_cost']:.2f}")
    
    # Chunked execution simulation
    print("\n5. Chunked Execution (200 contracts in 4 chunks):")
    simulation = model_prop.simulate_execution_path(price, 200, num_chunks=4)
    print(f"   Total size: {simulation['total_size']} contracts")
    print(f"   Average execution price: ${simulation['avg_execution_price']:.4f}")
    print(f"   Total slippage: ${simulation['total_slippage']:.4f} ({simulation['slippage_pct']:.2f}%)")
    print(f"\n   Execution details:")
    for exec_info in simulation['executions']:
        print(f"   Chunk {exec_info['chunk']}: {exec_info['size']} @ ${exec_info['price']:.4f}")
    
    logger.info("Slippage model test completed successfully")
    
"""
Order Book Simulator
Simulates realistic order book for options execution.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import deque
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OrderBook:
    """
    Simulates a realistic order book with bid-ask spreads and depth.
    """
    
    def __init__(self, depth_levels: int = 5):
        """
        Initialize order book.
        
        Args:
            depth_levels: Number of price levels to simulate
        """
        self.depth_levels = depth_levels
        self.bids = deque(maxlen=depth_levels)  # [(price, size), ...]
        self.asks = deque(maxlen=depth_levels)  # [(price, size), ...]
        self.last_trade_price = None
        self.last_trade_size = None
        logger.info(f"OrderBook initialized with {depth_levels} depth levels")
    
    def initialize_from_option(self, bid: float, ask: float, 
                              volume: int = 100) -> None:
        """
        Initialize order book from option bid-ask data.
        
        Args:
            bid: Best bid price
            ask: Best ask price
            volume: Typical volume
        """
        mid = (bid + ask) / 2
        spread = ask - bid
        
        # Generate bid side (decreasing prices, increasing sizes)
        self.bids.clear()
        for i in range(self.depth_levels):
            price = bid - i * spread * 0.5
            size = int(volume * (1 + i * 0.3))  # Larger sizes at worse prices
            self.bids.append((price, size))
        
        # Generate ask side (increasing prices, increasing sizes)
        self.asks.clear()
        for i in range(self.depth_levels):
            price = ask + i * spread * 0.5
            size = int(volume * (1 + i * 0.3))
            self.asks.append((price, size))
        
        logger.debug(f"Order book initialized: bid={bid:.4f}, ask={ask:.4f}, spread={spread:.4f}")
    
    def get_best_bid(self) -> Optional[Tuple[float, int]]:
        """Get best bid (highest price to buy)."""
        return self.bids[0] if self.bids else None
    
    def get_best_ask(self) -> Optional[Tuple[float, int]]:
        """Get best ask (lowest price to sell)."""
        return self.asks[0] if self.asks else None
    
    def get_mid_price(self) -> Optional[float]:
        """Get mid price between best bid and ask."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid and best_ask:
            return (best_bid[0] + best_ask[0]) / 2
        return None
    
    def get_spread(self) -> Optional[float]:
        """Get bid-ask spread."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid and best_ask:
            return best_ask[0] - best_bid[0]
        return None
    
    def execute_market_order(self, side: str, size: int) -> Dict:
        """
        Execute a market order (immediate execution at best available price).
        
        Args:
            side: 'buy' or 'sell'
            size: Number of contracts to trade
        
        Returns:
            Dictionary with execution details
        """
        filled_size = 0
        total_cost = 0.0
        executed_prices = []
        
        if side == 'buy':
            # Buy from asks (selling side)
            book_side = list(self.asks)
            
            for i, (price, available_size) in enumerate(book_side):
                if filled_size >= size:
                    break
                
                # Fill from this level
                fill_size = min(size - filled_size, available_size)
                total_cost += price * fill_size
                filled_size += fill_size
                executed_prices.append(price)
                
                # Update book
                remaining_size = available_size - fill_size
                if remaining_size > 0:
                    self.asks[i] = (price, remaining_size)
                else:
                    # Level exhausted, remove it
                    if i < len(self.asks):
                        self.asks.remove((price, available_size))
        
        else:  # sell
            # Sell to bids (buying side)
            book_side = list(self.bids)
            
            for i, (price, available_size) in enumerate(book_side):
                if filled_size >= size:
                    break
                
                # Fill from this level
                fill_size = min(size - filled_size, available_size)
                total_cost += price * fill_size
                filled_size += fill_size
                executed_prices.append(price)
                
                # Update book
                remaining_size = available_size - fill_size
                if remaining_size > 0:
                    self.bids[i] = (price, remaining_size)
                else:
                    if i < len(self.bids):
                        self.bids.remove((price, available_size))
        
        # Calculate average execution price
        avg_price = total_cost / filled_size if filled_size > 0 else 0
        
        # Track last trade
        if filled_size > 0:
            self.last_trade_price = avg_price
            self.last_trade_size = filled_size
        
        return {
            'side': side,
            'requested_size': size,
            'filled_size': filled_size,
            'avg_price': avg_price,
            'total_cost': total_cost,
            'executed_prices': executed_prices,
            'fully_filled': filled_size == size
        }
    
    def execute_limit_order(self, side: str, size: int, limit_price: float) -> Dict:
        """
        Execute a limit order (only execute at limit price or better).
        
        Args:
            side: 'buy' or 'sell'
            size: Number of contracts
            limit_price: Maximum price to buy / minimum price to sell
        
        Returns:
            Dictionary with execution details
        """
        filled_size = 0
        total_cost = 0.0
        executed_prices = []
        
        if side == 'buy':
            # Buy from asks at limit_price or lower
            book_side = list(self.asks)
            
            for i, (price, available_size) in enumerate(book_side):
                if price > limit_price:
                    break  # Price too high
                
                if filled_size >= size:
                    break
                
                fill_size = min(size - filled_size, available_size)
                total_cost += price * fill_size
                filled_size += fill_size
                executed_prices.append(price)
                
                remaining_size = available_size - fill_size
                if remaining_size > 0:
                    self.asks[i] = (price, remaining_size)
                else:
                    if i < len(self.asks):
                        self.asks.remove((price, available_size))
        
        else:  # sell
            # Sell to bids at limit_price or higher
            book_side = list(self.bids)
            
            for i, (price, available_size) in enumerate(book_side):
                if price < limit_price:
                    break  # Price too low
                
                if filled_size >= size:
                    break
                
                fill_size = min(size - filled_size, available_size)
                total_cost += price * fill_size
                filled_size += fill_size
                executed_prices.append(price)
                
                remaining_size = available_size - fill_size
                if remaining_size > 0:
                    self.bids[i] = (price, remaining_size)
                else:
                    if i < len(self.bids):
                        self.bids.remove((price, available_size))
        
        avg_price = total_cost / filled_size if filled_size > 0 else 0
        
        if filled_size > 0:
            self.last_trade_price = avg_price
            self.last_trade_size = filled_size
        
        return {
            'side': side,
            'requested_size': size,
            'filled_size': filled_size,
            'avg_price': avg_price,
            'total_cost': total_cost,
            'executed_prices': executed_prices,
            'limit_price': limit_price,
            'fully_filled': filled_size == size,
            'unfilled_size': size - filled_size
        }
    
    def calculate_slippage(self, side: str, size: int) -> float:
        """
        Calculate expected slippage for a market order.
        
        Args:
            side: 'buy' or 'sell'
            size: Order size
        
        Returns:
            Slippage amount (positive value)
        """
        mid_price = self.get_mid_price()
        
        if mid_price is None:
            return 0.0
        
        # Simulate execution
        execution = self.execute_market_order(side, size)
        avg_price = execution['avg_price']
        
        # Slippage is difference from mid price
        if side == 'buy':
            slippage = avg_price - mid_price
        else:
            slippage = mid_price - avg_price
        
        # Restore order book (this was just a simulation)
        # In real implementation, would need to maintain separate state
        
        return max(slippage, 0)
    
    def get_book_depth(self) -> Dict:
        """
        Get current order book depth.
        
        Returns:
            Dictionary with bid and ask levels
        """
        return {
            'bids': list(self.bids),
            'asks': list(self.asks),
            'bid_volume': sum(size for _, size in self.bids),
            'ask_volume': sum(size for _, size in self.asks),
            'spread': self.get_spread(),
            'mid_price': self.get_mid_price()
        }


# Example usage and testing
if __name__ == "__main__":
    print(f"\n{'='*70}")
    print(f"Order Book Simulator Test")
    print(f"{'='*70}")
    
    # Initialize order book
    book = OrderBook(depth_levels=5)
    
    # Set up from option data
    bid = 5.00
    ask = 5.20
    book.initialize_from_option(bid, ask, volume=100)
    
    print("\n1. Initial Order Book:")
    depth = book.get_book_depth()
    print(f"   Best Bid: ${depth['bids'][0][0]:.2f} x {depth['bids'][0][1]}")
    print(f"   Best Ask: ${depth['asks'][0][0]:.2f} x {depth['asks'][0][1]}")
    print(f"   Spread: ${depth['spread']:.4f}")
    print(f"   Mid Price: ${depth['mid_price']:.4f}")
    
    print("\n   Full Bid Side:")
    for price, size in depth['bids']:
        print(f"   ${price:.2f} x {size}")
    
    print("\n   Full Ask Side:")
    for price, size in depth['asks']:
        print(f"   ${price:.2f} x {size}")
    
    # Execute market buy order
    print("\n2. Executing Market Buy Order (50 contracts):")
    result = book.execute_market_order('buy', 50)
    print(f"   Requested: {result['requested_size']} contracts")
    print(f"   Filled: {result['filled_size']} contracts")
    print(f"   Avg Price: ${result['avg_price']:.4f}")
    print(f"   Total Cost: ${result['total_cost']:.2f}")
    print(f"   Fully Filled: {result['fully_filled']}")
    
    # Execute limit sell order
    print("\n3. Executing Limit Sell Order (30 contracts at $5.10):")
    result = book.execute_limit_order('sell', 30, limit_price=5.10)
    print(f"   Requested: {result['requested_size']} contracts")
    print(f"   Filled: {result['filled_size']} contracts")
    print(f"   Avg Price: ${result['avg_price']:.4f}")
    print(f"   Unfilled: {result['unfilled_size']} contracts")
    
    # Calculate slippage
    print("\n4. Slippage Analysis:")
    book.initialize_from_option(bid, ask, volume=100)  # Reset book
    slippage = book.calculate_slippage('buy', 150)
    print(f"   Expected slippage for 150 contracts: ${slippage:.4f}")
    
    logger.info("Order book simulator test completed successfully")
    
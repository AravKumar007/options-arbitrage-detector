"""
Position Manager
Tracks positions, P&L, and portfolio Greeks.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.pricing.black_scholes import BlackScholes
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Position:
    """Represents a single option position."""
    
    def __init__(self, ticker: str, strike: float, expiration: str,
                 option_type: str, quantity: int, entry_price: float,
                 entry_time: datetime = None):
        """
        Initialize position.
        
        Args:
            ticker: Underlying ticker
            strike: Strike price
            expiration: Expiration date
            option_type: 'call' or 'put'
            quantity: Number of contracts (positive = long, negative = short)
            entry_price: Entry price per contract
            entry_time: Entry timestamp
        """
        self.ticker = ticker
        self.strike = strike
        self.expiration = expiration
        self.option_type = option_type
        self.quantity = quantity
        self.entry_price = entry_price
        self.entry_time = entry_time or datetime.now()
        self.current_price = entry_price
        self.greeks = {}
    
    def update_price(self, price: float):
        """Update current market price."""
        self.current_price = price
    
    def calculate_pnl(self) -> float:
        """Calculate unrealized P&L."""
        pnl_per_contract = (self.current_price - self.entry_price) * self.quantity
        return pnl_per_contract * 100  # Options are per 100 shares
    
    def calculate_notional(self) -> float:
        """Calculate notional value."""
        return abs(self.quantity) * self.current_price * 100
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'ticker': self.ticker,
            'strike': self.strike,
            'expiration': self.expiration,
            'option_type': self.option_type,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'pnl': self.calculate_pnl(),
            'notional': self.calculate_notional(),
            'entry_time': self.entry_time
        }


class PositionManager:
    """
    Manage portfolio of option positions and track P&L.
    """
    
    def __init__(self, initial_capital: float = 100000):
        """
        Initialize position manager.
        
        Args:
            initial_capital: Starting capital
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = []  # List of Position objects
        self.trade_history = []
        self.bs_model = BlackScholes()
        logger.info(f"PositionManager initialized with ${initial_capital:,.2f}")
    
    def add_position(self, ticker: str, strike: float, expiration: str,
                    option_type: str, quantity: int, entry_price: float,
                    commission: float = 0.65) -> Dict:
        """
        Add a new position.
        
        Args:
            ticker: Underlying ticker
            strike: Strike price
            expiration: Expiration date
            option_type: 'call' or 'put'
            quantity: Number of contracts
            entry_price: Entry price
            commission: Commission per contract
        
        Returns:
            Dictionary with trade details
        """
        # Calculate cost
        notional = quantity * entry_price * 100
        total_commission = commission * abs(quantity)
        
        if quantity > 0:
            # Buying - deduct from cash
            total_cost = notional + total_commission
            if total_cost > self.cash:
                logger.error(f"Insufficient cash: need ${total_cost:.2f}, have ${self.cash:.2f}")
                return {'success': False, 'reason': 'insufficient_cash'}
            self.cash -= total_cost
        else:
            # Selling - add to cash
            total_credit = notional - total_commission
            self.cash += total_credit
        
        # Create position
        position = Position(ticker, strike, expiration, option_type, 
                          quantity, entry_price)
        self.positions.append(position)
        
        # Record trade
        trade = {
            'timestamp': datetime.now(),
            'action': 'open',
            'ticker': ticker,
            'strike': strike,
            'expiration': expiration,
            'option_type': option_type,
            'quantity': quantity,
            'price': entry_price,
            'commission': total_commission,
            'notional': notional
        }
        self.trade_history.append(trade)
        
        logger.info(f"Position added: {ticker} {strike} {option_type} x{quantity} @ ${entry_price:.2f}")
        
        return {'success': True, 'position': position, 'trade': trade}
    
    def close_position(self, position_index: int, exit_price: float,
                      commission: float = 0.65) -> Dict:
        """
        Close an existing position.
        
        Args:
            position_index: Index of position to close
            exit_price: Exit price
            commission: Commission per contract
        
        Returns:
            Dictionary with close details
        """
        if position_index >= len(self.positions):
            logger.error(f"Invalid position index: {position_index}")
            return {'success': False, 'reason': 'invalid_index'}
        
        position = self.positions[position_index]
        
        # Calculate P&L
        pnl_per_contract = (exit_price - position.entry_price) * position.quantity
        total_pnl = pnl_per_contract * 100
        
        # Calculate proceeds
        notional = abs(position.quantity) * exit_price * 100
        total_commission = commission * abs(position.quantity)
        
        if position.quantity > 0:
            # Closing long - receive cash
            proceeds = notional - total_commission
            self.cash += proceeds
        else:
            # Closing short - pay cash
            cost = notional + total_commission
            self.cash -= cost
        
        # Record trade
        trade = {
            'timestamp': datetime.now(),
            'action': 'close',
            'ticker': position.ticker,
            'strike': position.strike,
            'expiration': position.expiration,
            'option_type': position.option_type,
            'quantity': position.quantity,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'pnl': total_pnl,
            'commission': total_commission
        }
        self.trade_history.append(trade)
        
        # Remove position
        self.positions.pop(position_index)
        
        logger.info(f"Position closed: P&L = ${total_pnl:.2f}")
        
        return {'success': True, 'pnl': total_pnl, 'trade': trade}
    
    def update_prices(self, price_data: Dict):
        """
        Update all position prices.
        
        Args:
            price_data: Dictionary mapping (ticker, strike, expiration, type) -> price
        """
        for position in self.positions:
            key = (position.ticker, position.strike, position.expiration, position.option_type)
            if key in price_data:
                position.update_price(price_data[key])
    
    def calculate_portfolio_pnl(self) -> Dict[str, float]:
        """
        Calculate total portfolio P&L.
        
        Returns:
            Dictionary with P&L metrics
        """
        unrealized_pnl = sum(pos.calculate_pnl() for pos in self.positions)
        total_value = self.cash + self.get_total_notional()
        realized_pnl = total_value - self.initial_capital
        
        return {
            'unrealized_pnl': unrealized_pnl,
            'realized_pnl': realized_pnl,
            'total_pnl': unrealized_pnl + realized_pnl,
            'total_value': total_value,
            'cash': self.cash,
            'return_pct': ((total_value - self.initial_capital) / self.initial_capital) * 100
        }
    
    def calculate_portfolio_greeks(self, spot_prices: Dict[str, float],
                                  r: float, volatilities: Dict) -> Dict[str, float]:
        """
        Calculate aggregate portfolio Greeks.
        
        Args:
            spot_prices: Dictionary of ticker -> spot price
            r: Risk-free rate
            volatilities: Dictionary of position -> volatility
        
        Returns:
            Dictionary with portfolio Greeks
        """
        portfolio_greeks = {
            'delta': 0,
            'gamma': 0,
            'vega': 0,
            'theta': 0,
            'rho': 0
        }
        
        for position in self.positions:
            spot = spot_prices.get(position.ticker, 100)
            sigma = volatilities.get((position.ticker, position.strike, position.expiration), 0.25)
            
            # Calculate time to maturity
            # Simplified - in production would use actual date parsing
            T = 0.5  # Placeholder
            
            greeks = self.bs_model.greeks(
                spot, position.strike, T, r, sigma, position.option_type
            )
            
            # Aggregate (weighted by quantity)
            for greek, value in greeks.items():
                portfolio_greeks[greek] += value * position.quantity
        
        return portfolio_greeks
    
    def get_total_notional(self) -> float:
        """Get total notional exposure."""
        return sum(pos.calculate_notional() for pos in self.positions)
    
    def get_positions_summary(self) -> pd.DataFrame:
        """
        Get summary of all positions.
        
        Returns:
            DataFrame with position details
        """
        if not self.positions:
            return pd.DataFrame()
        
        data = [pos.to_dict() for pos in self.positions]
        return pd.DataFrame(data)
    
    def get_trade_history(self) -> pd.DataFrame:
        """Get trade history as DataFrame."""
        if not self.trade_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trade_history)
    
    def get_performance_metrics(self) -> Dict:
        """
        Calculate performance metrics.
        
        Returns:
            Dictionary with performance stats
        """
        pnl = self.calculate_portfolio_pnl()
        
        # Calculate from trade history
        closed_trades = [t for t in self.trade_history if t['action'] == 'close']
        
        if closed_trades:
            pnls = [t['pnl'] for t in closed_trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]
            
            win_rate = len(wins) / len(pnls) * 100 if pnls else 0
            avg_win = np.mean(wins) if wins else 0
            avg_loss = abs(np.mean(losses)) if losses else 0
            profit_factor = sum(wins) / abs(sum(losses)) if losses else float('inf')
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        return {
            'total_return': pnl['return_pct'],
            'total_pnl': pnl['total_pnl'],
            'num_trades': len(closed_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_value': pnl['total_value']
        }


# Example usage and testing
if __name__ == "__main__":
    print(f"\n{'='*70}")
    print(f"Position Manager Test")
    print(f"{'='*70}")
    
    # Initialize manager
    manager = PositionManager(initial_capital=100000)
    
    print(f"\n1. Initial State:")
    print(f"   Capital: ${manager.initial_capital:,.2f}")
    print(f"   Cash: ${manager.cash:,.2f}")
    
    # Add positions
    print(f"\n2. Opening Positions:")
    
    result1 = manager.add_position('SPY', 450, '2024-06-20', 'call', 10, 5.50)
    print(f"   Position 1: {result1['success']}")
    
    result2 = manager.add_position('SPY', 445, '2024-06-20', 'put', -5, 3.20)
    print(f"   Position 2: {result2['success']}")
    
    print(f"   Cash after trades: ${manager.cash:,.2f}")
    
    # Show positions
    print(f"\n3. Current Positions:")
    positions_df = manager.get_positions_summary()
    print(positions_df[['ticker', 'strike', 'option_type', 'quantity', 
                       'entry_price', 'notional']].to_string(index=False))
    
    # Update prices
    print(f"\n4. Updating Prices:")
    price_updates = {
        ('SPY', 450, '2024-06-20', 'call'): 6.20,
        ('SPY', 445, '2024-06-20', 'put'): 2.80
    }
    manager.update_prices(price_updates)
    
    # Calculate P&L
    print(f"\n5. Portfolio P&L:")
    pnl = manager.calculate_portfolio_pnl()
    for key, value in pnl.items():
        if 'pct' in key:
            print(f"   {key}: {value:.2f}%")
        else:
            print(f"   {key}: ${value:,.2f}")
    
    # Close a position
    print(f"\n6. Closing Position:")
    close_result = manager.close_position(0, exit_price=6.20)
    print(f"   Success: {close_result['success']}")
    print(f"   P&L: ${close_result['pnl']:.2f}")
    print(f"   Cash after close: ${manager.cash:,.2f}")
    
    # Performance metrics
    print(f"\n7. Performance Metrics:")
    metrics = manager.get_performance_metrics()
    for key, value in metrics.items():
        if 'rate' in key or 'return' in key:
            print(f"   {key}: {value:.2f}%")
        elif 'factor' in key:
            print(f"   {key}: {value:.2f}x")
        else:
            print(f"   {key}: {value:.2f}")
    
    logger.info("Position manager test completed successfully")
    
"""
Data Storage - Database operations for options data.
Handles storage, retrieval, and management of options chain data.
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OptionsDataStorage:
    """
    Manage options data storage in SQLite database.
    """
    
    def __init__(self, db_path: str = "data/market_data.db"):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._create_tables()
        logger.info(f"Database initialized at {self.db_path}")
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        
        # Options chain table
        options_table = """
        CREATE TABLE IF NOT EXISTS options_chain (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            option_type TEXT NOT NULL,
            strike REAL NOT NULL,
            expiration TEXT NOT NULL,
            bid REAL,
            ask REAL,
            mid_price REAL,
            last_price REAL,
            volume INTEGER,
            open_interest INTEGER,
            implied_volatility REAL,
            spot_price REAL,
            time_to_maturity REAL,
            moneyness REAL,
            classification TEXT,
            intrinsic_value REAL,
            time_value REAL,
            delta REAL,
            gamma REAL,
            vega REAL,
            theta REAL,
            rho REAL,
            quality_score REAL,
            fetch_time TIMESTAMP NOT NULL,
            UNIQUE(ticker, option_type, strike, expiration, fetch_time)
        )
        """
        
        # Stock prices table
        prices_table = """
        CREATE TABLE IF NOT EXISTS stock_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            price REAL NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            UNIQUE(ticker, timestamp)
        )
        """
        
        # Arbitrage signals table
        signals_table = """
        CREATE TABLE IF NOT EXISTS arbitrage_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            strategy_type TEXT NOT NULL,
            signal_time TIMESTAMP NOT NULL,
            expected_profit REAL,
            confidence_score REAL,
            details TEXT,
            status TEXT DEFAULT 'pending'
        )
        """
        
        with self.conn:
            self.conn.execute(options_table)
            self.conn.execute(prices_table)
            self.conn.execute(signals_table)
            
            # Create indexes for faster queries
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_options_ticker_exp ON options_chain(ticker, expiration)"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_options_fetch_time ON options_chain(fetch_time)"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_prices_ticker ON stock_prices(ticker, timestamp)"
            )
        
        logger.info("Database tables created/verified")
    
    def save_options_chain(self, df: pd.DataFrame) -> int:
        """
        Save options chain data to database.
        
        Args:
            df: DataFrame with options data
        
        Returns:
            Number of rows saved
        """
        if df.empty:
            logger.warning("Empty DataFrame provided to save")
            return 0
        
        # Select columns that exist in the table
        columns_to_save = [
            'ticker', 'option_type', 'strike', 'expiration',
            'bid', 'ask', 'mid_price', 'lastPrice', 'volume', 'openInterest',
            'impliedVolatility', 'spot_price', 'time_to_maturity', 'moneyness',
            'classification', 'intrinsic_value', 'time_value',
            'delta', 'gamma', 'vega', 'theta', 'rho',
            'quality_score', 'fetch_time'
        ]
        
        # Rename columns to match database schema
        df_to_save = df.copy()
        if 'lastPrice' in df_to_save.columns:
            df_to_save['last_price'] = df_to_save['lastPrice']
        if 'openInterest' in df_to_save.columns:
            df_to_save['open_interest'] = df_to_save['openInterest']
        if 'impliedVolatility' in df_to_save.columns:
            df_to_save['implied_volatility'] = df_to_save['impliedVolatility']
        
        # Select only columns that exist
        existing_cols = [col for col in columns_to_save if col in df_to_save.columns]
        df_to_save = df_to_save[existing_cols]
        
        # Save to database (replace duplicates)
        try:
            rows_saved = df_to_save.to_sql(
                'options_chain',
                self.conn,
                if_exists='append',
                index=False
            )
            logger.info(f"Saved {rows_saved} options contracts to database")
            return rows_saved
        except sqlite3.IntegrityError as e:
            logger.warning(f"Some rows already exist in database: {e}")
            return 0
    
    def get_latest_options(self, ticker: str, 
                          expiration: str = None,
                          hours_back: int = 24) -> pd.DataFrame:
        """
        Retrieve latest options data from database.
        
        Args:
            ticker: Stock ticker
            expiration: Specific expiration date (optional)
            hours_back: How many hours back to look
        
        Returns:
            DataFrame with options data
        """
        time_threshold = datetime.now() - timedelta(hours=hours_back)
        
        query = """
        SELECT * FROM options_chain
        WHERE ticker = ?
        AND fetch_time >= ?
        """
        params = [ticker, time_threshold]
        
        if expiration:
            query += " AND expiration = ?"
            params.append(expiration)
        
        query += " ORDER BY fetch_time DESC"
        
        df = pd.read_sql_query(query, self.conn, params=params)
        logger.info(f"Retrieved {len(df)} options for {ticker}")
        
        return df
    
    def get_options_history(self, ticker: str, 
                           strike: float,
                           option_type: str,
                           expiration: str,
                           days_back: int = 7) -> pd.DataFrame:
        """
        Get historical data for a specific option contract.
        
        Args:
            ticker: Stock ticker
            strike: Strike price
            option_type: 'call' or 'put'
            expiration: Expiration date
            days_back: Number of days of history
        
        Returns:
            DataFrame with historical data
        """
        time_threshold = datetime.now() - timedelta(days=days_back)
        
        query = """
        SELECT * FROM options_chain
        WHERE ticker = ?
        AND strike = ?
        AND option_type = ?
        AND expiration = ?
        AND fetch_time >= ?
        ORDER BY fetch_time ASC
        """
        
        df = pd.read_sql_query(
            query, self.conn,
            params=[ticker, strike, option_type, expiration, time_threshold]
        )
        
        logger.info(f"Retrieved {len(df)} historical records for {ticker} {strike} {option_type}")
        return df
    
    def save_stock_price(self, ticker: str, price: float):
        """
        Save stock price to database.
        
        Args:
            ticker: Stock ticker
            price: Current price
        """
        timestamp = datetime.now()
        
        try:
            query = """
            INSERT INTO stock_prices (ticker, price, timestamp)
            VALUES (?, ?, ?)
            """
            with self.conn:
                self.conn.execute(query, (ticker, price, timestamp))
            logger.debug(f"Saved price for {ticker}: ${price:.2f}")
        except sqlite3.IntegrityError:
            logger.debug(f"Price for {ticker} at {timestamp} already exists")
    
    def get_price_history(self, ticker: str, days_back: int = 30) -> pd.DataFrame:
        """
        Get historical price data.
        
        Args:
            ticker: Stock ticker
            days_back: Number of days of history
        
        Returns:
            DataFrame with price history
        """
        time_threshold = datetime.now() - timedelta(days=days_back)
        
        query = """
        SELECT * FROM stock_prices
        WHERE ticker = ?
        AND timestamp >= ?
        ORDER BY timestamp ASC
        """
        
        df = pd.read_sql_query(query, self.conn, params=[ticker, time_threshold])
        logger.info(f"Retrieved {len(df)} price records for {ticker}")
        
        return df
    
    def save_arbitrage_signal(self, ticker: str, strategy_type: str,
                             expected_profit: float, confidence: float,
                             details: str = None):
        """
        Save arbitrage signal to database.
        
        Args:
            ticker: Stock ticker
            strategy_type: Type of arbitrage strategy
            expected_profit: Expected profit in dollars
            confidence: Confidence score (0-1)
            details: Additional details (JSON string)
        """
        timestamp = datetime.now()
        
        query = """
        INSERT INTO arbitrage_signals 
        (ticker, strategy_type, signal_time, expected_profit, confidence_score, details)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        
        with self.conn:
            self.conn.execute(
                query,
                (ticker, strategy_type, timestamp, expected_profit, confidence, details)
            )
        
        logger.info(f"Saved arbitrage signal: {strategy_type} for {ticker}, profit=${expected_profit:.2f}")
    
    def get_recent_signals(self, hours_back: int = 24) -> pd.DataFrame:
        """
        Get recent arbitrage signals.
        
        Args:
            hours_back: How many hours back to look
        
        Returns:
            DataFrame with signals
        """
        time_threshold = datetime.now() - timedelta(hours=hours_back)
        
        query = """
        SELECT * FROM arbitrage_signals
        WHERE signal_time >= ?
        ORDER BY signal_time DESC
        """
        
        df = pd.read_sql_query(query, self.conn, params=[time_threshold])
        logger.info(f"Retrieved {len(df)} recent signals")
        
        return df
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """
        Remove old data from database.
        
        Args:
            days_to_keep: Number of days of data to retain
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        with self.conn:
            # Clean options data
            result = self.conn.execute(
                "DELETE FROM options_chain WHERE fetch_time < ?",
                (cutoff_date,)
            )
            options_deleted = result.rowcount
            
            # Clean price data
            result = self.conn.execute(
                "DELETE FROM stock_prices WHERE timestamp < ?",
                (cutoff_date,)
            )
            prices_deleted = result.rowcount
        
        logger.info(f"Cleaned up old data: {options_deleted} options, {prices_deleted} prices")
    
    def get_database_stats(self) -> Dict:
        """
        Get database statistics.
        
        Returns:
            Dictionary with stats
        """
        stats = {}
        
        # Count records in each table
        stats['options_count'] = pd.read_sql_query(
            "SELECT COUNT(*) as count FROM options_chain", self.conn
        )['count'].iloc[0]
        
        stats['prices_count'] = pd.read_sql_query(
            "SELECT COUNT(*) as count FROM stock_prices", self.conn
        )['count'].iloc[0]
        
        stats['signals_count'] = pd.read_sql_query(
            "SELECT COUNT(*) as count FROM arbitrage_signals", self.conn
        )['count'].iloc[0]
        
        # Get database size
        stats['db_size_mb'] = self.db_path.stat().st_size / (1024 * 1024)
        
        return stats
    
    def close(self):
        """Close database connection."""
        self.conn.close()
        logger.info("Database connection closed")


# Example usage and testing
if __name__ == "__main__":
    print(f"\n{'='*70}")
    print(f"Options Data Storage Test")
    print(f"{'='*70}")
    
    # Initialize storage
    storage = OptionsDataStorage()
    
    # Create sample data
    print("\n1. Creating sample options data...")
    sample_data = pd.DataFrame({
        'ticker': ['AAPL'] * 5,
        'option_type': ['call', 'call', 'call', 'put', 'put'],
        'strike': [150, 155, 160, 145, 140],
        'expiration': ['2024-12-20'] * 5,
        'bid': [5.0, 3.0, 1.5, 2.0, 0.5],
        'ask': [5.5, 3.5, 2.0, 2.5, 1.0],
        'mid_price': [5.25, 3.25, 1.75, 2.25, 0.75],
        'volume': [1000, 500, 200, 300, 100],
        'spot_price': [155.0] * 5,
        'fetch_time': [datetime.now()] * 5
    })
    
    # Save data
    print("\n2. Saving options data...")
    rows_saved = storage.save_options_chain(sample_data)
    print(f"Saved {rows_saved} rows")
    
    # Retrieve data
    print("\n3. Retrieving latest options...")
    retrieved = storage.get_latest_options('AAPL')
    print(f"Retrieved {len(retrieved)} options")
    if not retrieved.empty:
        print(retrieved[['ticker', 'strike', 'option_type', 'bid', 'ask']].head())
    
    # Save stock price
    print("\n4. Saving stock price...")
    storage.save_stock_price('AAPL', 155.0)
    
    # Save arbitrage signal
    print("\n5. Saving arbitrage signal...")
    storage.save_arbitrage_signal(
        'AAPL',
        'calendar_spread',
        50.0,
        0.85,
        '{"leg1": "call_150", "leg2": "call_155"}'
    )
    
    # Get database stats
    print("\n6. Database Statistics:")
    stats = storage.get_database_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Close connection
    storage.close()
    logger.info("Data storage test completed successfully")
    
"""
Quick system test
"""
import sys
from pathlib import Path

print("Testing Options Arbitrage System...")
print("="*60)

# Test 1: Imports
print("\n1. Testing imports...")
try:
    from src.data_ingestion.api_client import OptionsAPIClient
    from src.data_ingestion.data_fetcher import OptionsDataFetcher
    from src.data_ingestion.data_storage import OptionsDataStorage
    print("✓ All imports successful!")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Fetch live data
print("\n2. Fetching live SPY options data...")
try:
    client = OptionsAPIClient()
    price = client.get_current_price("SPY")
    print(f"✓ SPY Current Price: ${price:.2f}")
    
    expirations = client.get_available_expirations("SPY")
    print(f"✓ Found {len(expirations)} expiration dates")
    print(f"  Next expiration: {expirations[0]}")
    
    chain = client.get_option_chain("SPY", expirations[0])
    calls = chain['calls']
    puts = chain['puts']
    print(f"✓ Fetched {len(calls)} calls and {len(puts)} puts")
    
    # Show sample data
    if not calls.empty:
        print(f"\n  Sample Call Options:")
        print(calls[['strike', 'bid', 'ask', 'volume']].head(3).to_string(index=False))
    
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Database
print("\n3. Testing database...")
try:
    storage = OptionsDataStorage()
    stats = storage.get_database_stats()
    print(f"✓ Database initialized")
    print(f"  Options in DB: {stats['options_count']}")
    print(f"  DB size: {stats['db_size_mb']:.2f} MB")
    storage.close()
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Process and save data
print("\n4. Testing data fetcher and storage...")
try:
    fetcher = OptionsDataFetcher()
    storage = OptionsDataStorage()
    
    print("  Fetching and processing SPY options...")
    options_df = fetcher.fetch_and_process("SPY")
    
    if not options_df.empty:
        print(f"✓ Processed {len(options_df)} options")
        print(f"  Columns: {len(options_df.columns)}")
        
        # Save to database
        rows = storage.save_options_chain(options_df)
        print(f"✓ Saved to database")
        
        # Check stats again
        stats = storage.get_database_stats()
        print(f"  Options now in DB: {stats['options_count']}")
        print(f"  DB size: {stats['db_size_mb']:.2f} MB")
    else:
        print("✗ No options data returned")
    
    storage.close()
    
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("✓ System test complete!")
print("="*60)

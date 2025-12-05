/*
 * Fast Execution Engine (C++)
 * High-performance order matching and execution for options trading
 * 
 * Compile: g++ -std=c++17 -O3 fast_execution.cpp -o fast_execution
 * Run: ./fast_execution
 */

#include <iostream>
#include <vector>
#include <queue>
#include <map>
#include <string>
#include <chrono>
#include <algorithm>
#include <iomanip>

using namespace std;

// Order structure
struct Order {
    int id;
    string side;  // "BUY" or "SELL"
    double price;
    int quantity;
    long long timestamp;
    
    Order(int id, string side, double price, int qty) 
        : id(id), side(side), price(price), quantity(qty) {
        auto now = chrono::system_clock::now();
        timestamp = chrono::duration_cast<chrono::microseconds>(
            now.time_since_epoch()
        ).count();
    }
};

// Trade execution result
struct Trade {
    int buy_order_id;
    int sell_order_id;
    double price;
    int quantity;
    long long timestamp;
    
    Trade(int buy_id, int sell_id, double px, int qty)
        : buy_order_id(buy_id), sell_order_id(sell_id), 
          price(px), quantity(qty) {
        auto now = chrono::system_clock::now();
        timestamp = chrono::duration_cast<chrono::microseconds>(
            now.time_since_epoch()
        ).count();
    }
};

// Order Book class
class OrderBook {
private:
    // Price-time priority queues
    // Buy orders: highest price first
    struct BuyComparator {
        bool operator()(const Order& a, const Order& b) const {
            if (a.price != b.price) return a.price < b.price;
            return a.timestamp > b.timestamp;
        }
    };
    
    // Sell orders: lowest price first
    struct SellComparator {
        bool operator()(const Order& a, const Order& b) const {
            if (a.price != b.price) return a.price > b.price;
            return a.timestamp > b.timestamp;
        }
    };
    
    priority_queue<Order, vector<Order>, BuyComparator> buy_orders;
    priority_queue<Order, vector<Order>, SellComparator> sell_orders;
    
    vector<Trade> trades;
    int next_order_id;
    
public:
    OrderBook() : next_order_id(1) {}
    
    // Add order and attempt to match
    vector<Trade> addOrder(string side, double price, int quantity) {
        vector<Trade> new_trades;
        
        if (side == "BUY") {
            // Try to match with sell orders
            while (!sell_orders.empty() && quantity > 0) {
                Order best_sell = sell_orders.top();
                
                // Check if prices cross
                if (price >= best_sell.price) {
                    sell_orders.pop();
                    
                    // Execute trade
                    int trade_qty = min(quantity, best_sell.quantity);
                    double trade_price = best_sell.price;
                    
                    Trade t(next_order_id, best_sell.id, trade_price, trade_qty);
                    new_trades.push_back(t);
                    trades.push_back(t);
                    
                    quantity -= trade_qty;
                    best_sell.quantity -= trade_qty;
                    
                    // Re-add partial order
                    if (best_sell.quantity > 0) {
                        sell_orders.push(best_sell);
                    }
                } else {
                    break;  // No match
                }
            }
            
            // Add remaining quantity to book
            if (quantity > 0) {
                buy_orders.push(Order(next_order_id++, side, price, quantity));
            }
            
        } else {  // SELL
            // Try to match with buy orders
            while (!buy_orders.empty() && quantity > 0) {
                Order best_buy = buy_orders.top();
                
                // Check if prices cross
                if (price <= best_buy.price) {
                    buy_orders.pop();
                    
                    // Execute trade
                    int trade_qty = min(quantity, best_buy.quantity);
                    double trade_price = best_buy.price;
                    
                    Trade t(best_buy.id, next_order_id, trade_price, trade_qty);
                    new_trades.push_back(t);
                    trades.push_back(t);
                    
                    quantity -= trade_qty;
                    best_buy.quantity -= trade_qty;
                    
                    // Re-add partial order
                    if (best_buy.quantity > 0) {
                        buy_orders.push(best_buy);
                    }
                } else {
                    break;  // No match
                }
            }
            
            // Add remaining quantity to book
            if (quantity > 0) {
                sell_orders.push(Order(next_order_id++, side, price, quantity));
            }
        }
        
        return new_trades;
    }
    
    // Get best bid price
    double getBestBid() const {
        if (buy_orders.empty()) return 0.0;
        return buy_orders.top().price;
    }
    
    // Get best ask price
    double getBestAsk() const {
        if (sell_orders.empty()) return 0.0;
        return sell_orders.top().price;
    }
    
    // Get spread
    double getSpread() const {
        if (buy_orders.empty() || sell_orders.empty()) return 0.0;
        return getBestAsk() - getBestBid();
    }
    
    // Get total trades
    int getTotalTrades() const {
        return trades.size();
    }
    
    // Get total volume
    int getTotalVolume() const {
        int volume = 0;
        for (const auto& t : trades) {
            volume += t.quantity;
        }
        return volume;
    }
    
    // Print statistics
    void printStats() const {
        cout << "\n=== Order Book Statistics ===" << endl;
        cout << "Best Bid: $" << fixed << setprecision(2) << getBestBid() << endl;
        cout << "Best Ask: $" << getBestAsk() << endl;
        cout << "Spread: $" << getSpread() << endl;
        cout << "Total Trades: " << getTotalTrades() << endl;
        cout << "Total Volume: " << getTotalVolume() << " contracts" << endl;
        cout << "Buy Orders in Book: " << buy_orders.size() << endl;
        cout << "Sell Orders in Book: " << sell_orders.size() << endl;
    }
    
    // Print recent trades
    void printRecentTrades(int n = 5) const {
        cout << "\n=== Recent Trades ===" << endl;
        int count = 0;
        for (auto it = trades.rbegin(); it != trades.rend() && count < n; ++it, ++count) {
            cout << "Trade " << count + 1 << ": "
                 << it->quantity << " @ $" << fixed << setprecision(2) 
                 << it->price << endl;
        }
    }
};

// Performance benchmark
class Benchmark {
private:
    chrono::high_resolution_clock::time_point start_time;
    
public:
    void start() {
        start_time = chrono::high_resolution_clock::now();
    }
    
    long long stop() {
        auto end_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(
            end_time - start_time
        ).count();
        return duration;
    }
    
    void printResult(const string& operation, int count, long long microseconds) {
        double ops_per_second = (count * 1000000.0) / microseconds;
        cout << "\n=== Performance: " << operation << " ===" << endl;
        cout << "Total operations: " << count << endl;
        cout << "Time taken: " << microseconds << " microseconds" << endl;
        cout << "Throughput: " << fixed << setprecision(0) 
             << ops_per_second << " ops/second" << endl;
    }
};

// Main function - demonstration
int main() {
    cout << "==================================================" << endl;
    cout << "   High-Performance Options Execution Engine" << endl;
    cout << "==================================================" << endl;
    
    OrderBook book;
    Benchmark benchmark;
    
    // Simulation parameters
    const int NUM_ORDERS = 10000;
    
    cout << "\n[1] Running performance test with " << NUM_ORDERS << " orders..." << endl;
    
    benchmark.start();
    
    // Generate and process orders
    for (int i = 0; i < NUM_ORDERS; i++) {
        // Alternate between buy and sell
        string side = (i % 2 == 0) ? "BUY" : "SELL";
        
        // Random price around 100
        double price = 100.0 + (rand() % 100) / 10.0 - 5.0;
        
        // Random quantity
        int quantity = 1 + (rand() % 10);
        
        // Add order
        book.addOrder(side, price, quantity);
    }
    
    long long elapsed = benchmark.stop();
    
    // Print results
    book.printStats();
    book.printRecentTrades(10);
    benchmark.printResult("Order Processing", NUM_ORDERS, elapsed);
    
    // Calculate average latency
    double avg_latency = (double)elapsed / NUM_ORDERS;
    cout << "\nAverage latency per order: " 
         << fixed << setprecision(2) << avg_latency << " microseconds" << endl;
    
    cout << "\n[2] Testing aggressive order matching..." << endl;
    
    OrderBook book2;
    
    // Add limit orders
    book2.addOrder("SELL", 105.0, 10);
    book2.addOrder("SELL", 104.0, 20);
    book2.addOrder("BUY", 103.0, 15);
    book2.addOrder("BUY", 102.0, 25);
    
    cout << "\nInitial book state:" << endl;
    book2.printStats();
    
    // Execute aggressive buy order that crosses spread
    cout << "\nExecuting aggressive BUY order: 30 contracts @ $105.0..." << endl;
    auto trades = book2.addOrder("BUY", 105.0, 30);
    
    cout << "Trades executed: " << trades.size() << endl;
    for (const auto& t : trades) {
        cout << "  - " << t.quantity << " @ $" << t.price << endl;
    }
    
    cout << "\nFinal book state:" << endl;
    book2.printStats();
    
    cout << "\n==================================================" << endl;
    cout << "   Execution engine test completed successfully!" << endl;
    cout << "==================================================" << endl;
    
    return 0;
}


import time
import random
import requests

API_URL = "http://localhost:8000/predict"

def generate_random_transaction():
    """Generates either a normal customer transaction or a high-risk fraudulent profile."""
    is_fraudulent_intent = random.random() < 0.20  # 20% chance of high-risk scenario
    
    if is_fraudulent_intent:
        # High-risk profile (late night, high amount, far from home, online)
        return {
            "transaction_amount": round(random.uniform(850.0, 3500.0), 2),
            "transaction_hour": random.choice([1, 2, 3, 23, 0]),
            "distance_from_home": round(random.uniform(60.0, 250.0), 2),
            "ratio_to_median_purchase_price": round(random.uniform(4.0, 9.5), 2),
            "used_chip": 0,
            "used_pin_number": 0,
            "online_order": 1
        }
    else:
        # Normal customer profile (daytime, reasonable amount, chip/PIN used)
        return {
            "transaction_amount": round(random.uniform(4.50, 120.0), 2),
            "transaction_hour": random.randint(8, 21),
            "distance_from_home": round(random.uniform(0.5, 15.0), 2),
            "ratio_to_median_purchase_price": round(random.uniform(0.5, 1.8), 2),
            "used_chip": 1,
            "used_pin_number": 1,
            "online_order": random.choice([0, 1])
        }

if __name__ == "__main__":
    print("=" * 65)
    print("🚀 STARTING REAL-TIME TRANSACTION TRAFFIC SIMULATOR")
    print("=" * 65)
    print("Target Endpoint: http://localhost:8000/predict")
    print("Press Ctrl+C to stop the simulation.\n")
    
    request_count = 0
    flagged_count = 0
    
    while True:
        try:
            payload = generate_random_transaction()
            response = requests.post(API_URL, json=payload, timeout=2.0)
            
            if response.status_code == 200:
                request_count += 1
                data = response.json()
                prob = data["fraud_probability"] * 100
                risk = data["risk_tier"]
                is_fraud = data["is_fraud"]
                latency = data["latency_ms"]
                
                if is_fraud:
                    flagged_count += 1
                    status_str = "🚨 [BLOCK / FRAUD]"
                else:
                    status_str = "✅ [APPROVE    ]"
                
                print(f"{status_str} | Amt: ${payload['transaction_amount']:<7.2f} | Risk: {risk:<11} | Prob: {prob:5.1f}% | Latency: {latency}ms")
            else:
                print(f"⚠️ API returned status code: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Connection refused. Make sure your FastAPI server is running!")
            break
        except Exception as e:
            print(f"⚠️ Error: {e}")
            
        # Wait 1.5 seconds between simulated transactions
        time.sleep(1.5)
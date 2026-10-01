# scripts/load_test.py
#!/usr/bin/env python3
"""
Load testing script for model-inference-service.

Usage:
    python scripts/load_test.py
"""

import asyncio
import time
import statistics
from typing import List, Tuple
import httpx


async def run_load_test(
    url: str = "http://localhost:8000",
    num_requests: int = 100,
    concurrency: int = 10
) -> None:
    """Run load test against inference service."""
    
    print(f"🔥 Load Testing {url}")
    print(f"   Requests: {num_requests}")
    print(f"   Concurrency: {concurrency}")
    print("=" * 60)
    
    latencies: List[float] = []
    errors: List[str] = []
    
    async with httpx.AsyncClient() as client:
        semaphore = asyncio.Semaphore(concurrency)
        
        async def make_request(idx: int) -> Tuple[float, bool]:
            """Make single inference request."""
            async with semaphore:
                try:
                    start = time.time()
                    response = await client.post(
                        f"{url}/v1/matching/predict",
                        json={
                            "entity_id_1": f"E_{idx}_1",
                            "entity_id_2": f"E_{idx}_2",
                            "include_shap": False
                        },
                        timeout=5.0
                    )
                    elapsed = (time.time() - start) * 1000  # milliseconds
                    
                    if response.status_code == 200:
                        return elapsed, True
                    else:
                        errors.append(f"HTTP {response.status_code}")
                        return elapsed, False
                
                except Exception as e:
                    errors.append(str(e))
                    return 0, False
        
        # Run all requests
        start_time = time.time()
        tasks = [make_request(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Collect successful latencies
        latencies = [lat for lat, success in results if success]
        successful = len(latencies)
        failed = len(results) - successful
        
        # Print results
        print(f"\n📊 Results:")
        print(f"   Successful: {successful}/{num_requests}")
        print(f"   Failed: {failed}/{num_requests}")
        print(f"   Success Rate: {(successful/num_requests*100):.1f}%")
        
        if latencies:
            print(f"\n⏱️  Latency (ms):")
            print(f"   Min: {min(latencies):.2f}")
            print(f"   Max: {max(latencies):.2f}")
            print(f"   Mean: {statistics.mean(latencies):.2f}")
            print(f"   Median: {statistics.median(latencies):.2f}")
            print(f"   P95: {sorted(latencies)[int(len(latencies)*0.95)]:.2f}")
            print(f"   P99: {sorted(latencies)[int(len(latencies)*0.99)]:.2f}")
        
        print(f"\n📈 Throughput:")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Requests/sec: {num_requests/total_time:.2f}")
        
        if errors:
            print(f"\n❌ Errors:")
            error_counts = {}
            for error in errors:
                error_counts[error] = error_counts.get(error, 0) + 1
            for error, count in error_counts.items():
                print(f"   {error}: {count}")


if __name__ == "__main__":
    asyncio.run(run_load_test())

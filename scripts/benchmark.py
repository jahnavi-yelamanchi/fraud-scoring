import argparse
import asyncio
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import httpx


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/v1/score")
    parser.add_argument("--health-url", default="http://localhost:8000/healthz")
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=25)
    parser.add_argument("--payload", default="data/processed/benchmark_payload.json")
    parser.add_argument("--startup-timeout", type=float, default=30)
    arguments = parser.parse_args()
    payload = json.loads(Path(arguments.payload).read_text())
    limiter = asyncio.Semaphore(arguments.concurrency)
    latencies = []
    async with httpx.AsyncClient(timeout=10) as client:
        deadline = perf_counter() + arguments.startup_timeout
        while True:
            try:
                if (await client.get(arguments.health_url)).is_success:
                    break
            except httpx.HTTPError:
                pass
            if perf_counter() >= deadline:
                raise RuntimeError(f"API did not become ready within {arguments.startup_timeout}s")
            await asyncio.sleep(0.25)
        async def request() -> int:
            async with limiter:
                start = perf_counter()
                response = await client.post(arguments.url, json=payload)
                response.raise_for_status()
                latencies.append((perf_counter() - start) * 1000)
                return response.status_code
        start = perf_counter()
        statuses = await asyncio.gather(*(request() for _ in range(arguments.requests)))
    elapsed = perf_counter() - start
    report = {"requests": arguments.requests, "concurrency": arguments.concurrency, "successes": sum(status == 200 for status in statuses), "throughput_rps": round(arguments.requests / elapsed, 2), "latency_ms": {"p50": round(float(np.percentile(latencies, 50)), 2), "p95": round(float(np.percentile(latencies, 95)), 2), "p99": round(float(np.percentile(latencies, 99)), 2)}}
    Path("reports").mkdir(exist_ok=True)
    Path("reports/latest_benchmark.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

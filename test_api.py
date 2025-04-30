from __future__ import annotations

import argparse
import time
from typing import List

import requests

# ---------------------------------------------------------------------------
# Sample bios for quick sanity‑check. Feel free to extend or replace.
# ---------------------------------------------------------------------------

TEST_BIOS: List[str] = [
    (
        "I am a theoretical biologist, interested in disease ecology. My tools are R, "
        "Clojure, compartmental disease modelling, and statistical GAM models. I work "
        "with geophysical, climate, biodiversity and land‑use data. I'm also fascinated "
        "by tech that tackles real‑world challenges in agriculture, conservation, and "
        "forecasting, plus AI and large language models."
    ),
    (
        "I'm a full‑stack engineer obsessed with TypeScript, React, and serverless. In my "
        "spare time I hack on open‑source dev‑tools, explore Rust, and follow web‑perf "
        "benchmarks and platform security news."
    ),
    (
        "As an embedded‑systems hobbyist I design open‑source hardware, prototype IoT "
        "gadgets on ESP32 and RP2040 boards, and love low‑power firmware tricks."
    ),
]


# ---------------------------------------------------------------------------
# Helper – single API call with latency measurement
# ---------------------------------------------------------------------------

def call_api(url: str, bio: str, top: int = 5) -> None:
    """POST *bio* to *url*, print top stories & latency (ms)."""

    payload = {"bio": bio}
    t0 = time.perf_counter()
    try:
        response = requests.post(url, json=payload, timeout=30)
    except requests.RequestException as exc:
        print(f"⚠️  Request failed: {exc}\n")
        return

    elapsed_ms = (time.perf_counter() - t0) * 1000

    if response.status_code != 200:
        print(f"❌ HTTP {response.status_code}: {response.text}\n")
        return

    try:
        items = response.json()
    except ValueError:
        print("❌ Response was not valid JSON\n")
        return

    print(f"Bio → '{bio}…'  |  {elapsed_ms:.1f} ms  |  showing top {top} results")
    for idx, item in enumerate(items[:top], 1):
        title = item.get("title", "<no title>")
        sim = item.get("similarity", 0.0)
        print(f"  {idx:2d}. {title}  (sim={sim:.3f})")
    print()


# ---------------------------------------------------------------------------
# CLI entry‑point
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Test the HN rerank API")
    parser.add_argument(
        "--url",
        default="http://localhost:8080/rerank",
        help="Endpoint URL of the /rerank route",
    )
    parser.add_argument(
        "--top", type=int, default=5, help="Number of items to print per bio"
    )

    args = parser.parse_args(argv)

    for bio in TEST_BIOS:
        call_api(args.url, bio, args.top)


if __name__ == "__main__":
    main()

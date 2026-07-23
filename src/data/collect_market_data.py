#!/usr/bin/env python3
"""
collect_market_data.py

Live L2 order-book depth + trade-print capture from Binance's public
websocket API, for a single symbol (default: BTCUSDT). No API key/auth
needed -- these are public market-data streams.

Why both streams:
  - `<symbol>@depth@100ms`  (Diff. Depth Stream) gives incremental L2
    order-book updates roughly every 100ms -- the raw material for
    computing mid-price, spread, and the book we'd quote against.
  - `<symbol>@trade`        gives individual executed trade prints --
    the raw material for backtesting fill/execution behavior (Phase 3+),
    since it tells us what actually traded, at what price, and which
    side was the taker.

Both streams are opened concurrently via asyncio and each incoming
message is timestamped on receipt (`local_ts`, a `time.time()` UTC
epoch float) in addition to whatever timestamp Binance embeds in the
payload itself (`E`/`T`), so the two streams can be aligned/synchronized
downstream even though they arrive on independent connections.

Output: one newline-delimited JSON (.jsonl) file per stream per run,
written to --out-dir (default data/raw/, gitignored -- this is scratch
capture data, not something to commit).

Usage:
    python src/data/collect_market_data.py --duration 60
    python src/data/collect_market_data.py --symbol ethusdt --duration 120
    python src/data/collect_market_data.py --duration 30 --out-dir data/raw --port 443

See data/README.md for more detail and sample captured-stats output.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone

import websockets

# Binance documents both port 9443 and the default HTTPS port 443 as
# valid for the public market-data websocket streams. Some networks
# (firewalled sandboxes, some corporate networks) block 9443 outbound
# while allowing 443. We try the candidates in order and fall back
# automatically, per connection, rather than hard-failing on the first
# one that doesn't work.
CANDIDATE_PORTS = [9443, 443]
WS_HOST = "stream.binance.com"


def build_url(port: int, symbol: str, stream: str) -> str:
    if port == 443:
        # Default HTTPS port: omit it from the URL entirely (Binance
        # accepts wss://stream.binance.com/ws/... on the implicit 443).
        return f"wss://{WS_HOST}/ws/{symbol}@{stream}"
    return f"wss://{WS_HOST}:{port}/ws/{symbol}@{stream}"


async def connect_with_fallback(symbol: str, stream: str, ports, connect_timeout=10.0):
    """Try each candidate port in order; return the first live connection."""
    last_err = None
    for port in ports:
        url = build_url(port, symbol, stream)
        try:
            ws = await asyncio.wait_for(websockets.connect(url), timeout=connect_timeout)
            print(f"[connect] {stream}: connected via {url}", file=sys.stderr)
            return ws, url
        except Exception as e:  # noqa: BLE001 -- want to try the next port on *any* failure
            last_err = e
            print(f"[connect] {stream}: failed on port {port} ({type(e).__name__}: {e}), trying next...",
                  file=sys.stderr)
    raise ConnectionError(f"Could not connect {stream} stream on any of {ports}: {last_err}")


async def capture_stream(symbol: str, stream_name: str, out_path: str, duration: float,
                          ports, stats: dict):
    """Connect to one raw Binance stream and append JSON-lines until duration elapses."""
    ws, url = await connect_with_fallback(symbol, stream_name, ports)
    stats["url"] = url
    count = 0
    deadline = time.monotonic() + duration
    try:
        with open(out_path, "a", buffering=1) as f:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    raw_msg = await asyncio.wait_for(ws.recv(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                local_ts = time.time()
                try:
                    payload = json.loads(raw_msg)
                except json.JSONDecodeError:
                    continue
                record = {"local_ts": local_ts, "stream": stream_name, "raw": payload}
                f.write(json.dumps(record) + "\n")
                count += 1
    finally:
        await ws.close()
    stats["count"] = count
    stats["out_path"] = out_path
    return stats


def summarize_depth(path: str) -> dict:
    """Top-of-book / mid-price stats from a captured diff-depth stream.

    `@depth@100ms` messages are incremental diffs (changed price levels
    only, qty "0" meaning "remove this level"), not full-book snapshots.
    Binance's documented procedure for an exact reconstruction is to seed
    from a REST `/api/v3/depth` snapshot and then apply diffs on top of
    it. This capture script only opens the websocket (by design -- it's
    a raw capture tool, not a book-builder), so instead we seed the local
    book from the *first* diff message's own levels and apply every
    subsequent diff on top of that running state. Any price level that
    never appears in a diff during the capture window is invisible to
    this reconstruction -- in practice, for a liquid pair like BTCUSDT,
    levels near the touch update many times per second, so top-of-book
    converges to the true book almost immediately and the mid-price
    series below is accurate; only the deep book (far from touch) may be
    incomplete. Full snapshot-seeded reconstruction belongs to the
    baseline implementation in Part 2.
    """
    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    times, mids, spreads = [], [], []
    n = 0
    if not os.path.exists(path):
        return {"count": 0}
    with open(path) as f:
        for line in f:
            n += 1
            rec = json.loads(line)
            raw = rec["raw"]
            for p, qty in raw.get("b") or []:
                p, qty = float(p), float(qty)
                if qty == 0.0:
                    bids.pop(p, None)
                else:
                    bids[p] = qty
            for p, qty in raw.get("a") or []:
                p, qty = float(p), float(qty)
                if qty == 0.0:
                    asks.pop(p, None)
                else:
                    asks[p] = qty
            if bids and asks:
                best_bid, best_ask = max(bids), min(asks)
                if best_bid < best_ask:  # sanity: book must not be crossed
                    times.append(rec["local_ts"])
                    mids.append((best_bid + best_ask) / 2.0)
                    spreads.append(best_ask - best_bid)
    out = {"count": n}
    if times:
        out["first_ts"] = min(times)
        out["last_ts"] = max(times)
        out["span_sec"] = max(times) - min(times)
    if mids:
        out["mid_min"] = min(mids)
        out["mid_max"] = max(mids)
        out["mid_last"] = mids[-1]
        out["spread_min"] = min(spreads)
        out["spread_max"] = max(spreads)
        out["spread_mean"] = sum(spreads) / len(spreads)
        out["book_states"] = len(mids)
    return out


def summarize_trades(path: str) -> dict:
    prices, times = [], []
    n = 0
    if not os.path.exists(path):
        return {"count": 0}
    with open(path) as f:
        for line in f:
            n += 1
            rec = json.loads(line)
            times.append(rec["local_ts"])
            p = rec["raw"].get("p")
            if p is not None:
                prices.append(float(p))
    out = {"count": n}
    if times:
        out["first_ts"] = min(times)
        out["last_ts"] = max(times)
        out["span_sec"] = max(times) - min(times)
    if prices:
        out["price_min"] = min(prices)
        out["price_max"] = max(prices)
        out["price_last"] = prices[-1]
    return out


def fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + "Z"


async def main_async(args: argparse.Namespace) -> int:
    os.makedirs(args.out_dir, exist_ok=True)
    run_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    depth_path = os.path.join(args.out_dir, f"depth_{args.symbol}_{run_tag}.jsonl")
    trades_path = os.path.join(args.out_dir, f"trades_{args.symbol}_{run_tag}.jsonl")

    ports = [args.port] if args.port else CANDIDATE_PORTS

    print(f"[run] symbol={args.symbol} duration={args.duration}s "
          f"depth_stream={args.symbol}@{args.depth_stream} trade_stream={args.symbol}@trade",
          file=sys.stderr)
    print(f"[run] writing depth -> {depth_path}", file=sys.stderr)
    print(f"[run] writing trades -> {trades_path}", file=sys.stderr)

    depth_stats: dict = {}
    trade_stats: dict = {}

    results = await asyncio.gather(
        capture_stream(args.symbol, f"{args.depth_stream}", depth_path, args.duration, ports, depth_stats),
        capture_stream(args.symbol, "trade", trades_path, args.duration, ports, trade_stats),
        return_exceptions=True,
    )

    for label, res in zip(["depth", "trade"], results):
        if isinstance(res, Exception):
            print(f"[error] {label} stream failed: {type(res).__name__}: {res}", file=sys.stderr)

    depth_summary = summarize_depth(depth_path)
    trade_summary = summarize_trades(trades_path)

    print("\n===== Capture summary =====")
    print(f"Symbol: {args.symbol.upper()}   Requested duration: {args.duration}s")
    print(f"\nDepth stream ({args.symbol}@{args.depth_stream}):")
    print(f"  events captured : {depth_summary.get('count', 0)}")
    if depth_summary.get("count"):
        print(f"  time span       : {depth_summary['span_sec']:.2f}s "
              f"({fmt_ts(depth_summary['first_ts'])} -> {fmt_ts(depth_summary['last_ts'])})")
        if "mid_min" in depth_summary:
            print(f"  reconstructed book states: {depth_summary['book_states']}")
            print(f"  mid-price range : {depth_summary['mid_min']:.2f} - {depth_summary['mid_max']:.2f}"
                  f"  (last: {depth_summary['mid_last']:.2f})")
            print(f"  spread range    : {depth_summary['spread_min']:.2f} - {depth_summary['spread_max']:.2f}"
                  f"  (mean: {depth_summary['spread_mean']:.3f})")
    print(f"\nTrade stream ({args.symbol}@trade):")
    print(f"  trades captured : {trade_summary.get('count', 0)}")
    if trade_summary.get("count"):
        print(f"  time span       : {trade_summary['span_sec']:.2f}s "
              f"({fmt_ts(trade_summary['first_ts'])} -> {fmt_ts(trade_summary['last_ts'])})")
        print(f"  trade price range: {trade_summary['price_min']:.2f} - {trade_summary['price_max']:.2f}"
              f"  (last: {trade_summary['price_last']:.2f})")
    print(f"\nRaw files:\n  {depth_path}\n  {trades_path}")
    print("============================\n")

    ok = depth_summary.get("count", 0) > 0 and trade_summary.get("count", 0) > 0
    return 0 if ok else 1


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Capture live Binance L2 depth + trade streams to data/raw/ (jsonl).",
    )
    p.add_argument("--symbol", default="btcusdt",
                    help="Trading pair, lowercase, Binance format (default: btcusdt)")
    p.add_argument("--duration", type=float, default=60.0,
                    help="Capture window in seconds, applied to both streams (default: 60)")
    p.add_argument("--out-dir", default="data/raw",
                    help="Output directory for captured .jsonl files (default: data/raw)")
    p.add_argument("--depth-stream", default="depth@100ms",
                    help="Depth stream suffix after '<symbol>@' (default: depth@100ms). "
                         "Use 'depth' for the (slower, unbounded-cadence) full-diff variant.")
    p.add_argument("--port", type=int, default=None,
                    help="Force a specific websocket port (9443 or 443). "
                         "Default: try 9443 first, fall back to 443 automatically.")
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    try:
        exit_code = asyncio.run(main_async(args))
    except KeyboardInterrupt:
        exit_code = 130
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

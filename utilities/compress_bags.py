"""Losslessly compress or decompress ROS1 bags for transport.

Rewrites each bag with ROS1's built-in chunk compression (BZ2) or without
compression, using the pure-Python ``rosbags`` library (no ROS install needed).
Message payloads, topics, types, and timestamps are preserved exactly; the bag
file bytes themselves differ because chunk layout and indexes are rebuilt.

Integrity is checked with a digest over the full message stream:

* ``compress`` hashes the source while writing, reads the output back, and
  requires the digests to match. It writes a sidecar ``<output>.json`` holding
  the digest.
* ``decompress`` hashes the compressed input (and checks it against the
  sidecar when present, which catches corruption in transit), then verifies
  the decompressed output the same way.

Sources are deleted only with ``--delete-source`` and only after verification.

Usage (from the repo root, or copy this single file elsewhere; it needs only
``pip install rosbags``)::

    python -m utilities.compress_bags compress data/3_1_*/UMD_*chunk0[4-9].bag --jobs 4
    python -m utilities.compress_bags decompress /media/usb/*.bz2.bag --out-dir data/3_1_0822_mckeldin_north
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from rosbags.rosbag1 import Reader, Writer

COMPRESSED_SUFFIX = ".bz2.bag"


def stream_digest(path: Path) -> tuple[str, int]:
    """Return (sha256 hex digest, message count) over a bag's message stream."""
    h = hashlib.sha256()
    count = 0
    with Reader(path) as reader:
        for conn, timestamp, raw in reader.messages():
            _update(h, conn, timestamp, raw)
            count += 1
    return h.hexdigest(), count


def _update(h: hashlib._Hash, conn, timestamp: int, raw: bytes) -> None:
    # Separators keep field boundaries unambiguous within the digest.
    h.update(conn.topic.encode())
    h.update(b"\0")
    h.update(conn.msgtype.encode())
    h.update(b"\0")
    h.update(timestamp.to_bytes(8, "little"))
    h.update(len(raw).to_bytes(8, "little"))
    h.update(raw)


def rewrite(src: Path, dst: Path, compress: bool) -> tuple[str, int]:
    """Copy every message from src to dst; return the source stream digest and count."""
    tmp = dst.with_name(dst.name + ".part")
    tmp.unlink(missing_ok=True)
    writer = Writer(tmp)
    if compress:
        writer.set_compression(Writer.CompressionFormat.BZ2)
    h = hashlib.sha256()
    count = 0
    try:
        with Reader(src) as reader, writer:
            out_conns = {
                c.id: writer.add_connection(
                    c.topic,
                    c.msgtype,
                    msgdef=c.msgdef.data,
                    md5sum=c.digest,
                    callerid=c.ext.callerid,
                    latching=c.ext.latching,
                )
                for c in reader.connections
            }
            for conn, timestamp, raw in reader.messages():
                writer.write(out_conns[conn.id], timestamp, raw)
                _update(h, conn, timestamp, raw)
                count += 1
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(dst)
    return h.hexdigest(), count


def output_path(src: Path, mode: str, out_dir: Path | None) -> Path:
    name = src.name
    if mode == "compress":
        if name.endswith(COMPRESSED_SUFFIX):
            raise ValueError(f"{src} already looks compressed")
        name = name.removesuffix(".bag") + COMPRESSED_SUFFIX
    else:
        if not name.endswith(COMPRESSED_SUFFIX):
            raise ValueError(f"{src} does not end in {COMPRESSED_SUFFIX}")
        name = name.removesuffix(COMPRESSED_SUFFIX) + ".bag"
    return (out_dir or src.parent) / name


def sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".json")


def process(src: Path, mode: str, out_dir: Path | None, delete_source: bool, overwrite: bool) -> str:
    dst = output_path(src, mode, out_dir)
    if dst.exists() and not overwrite:
        raise FileExistsError(f"{dst} exists (use --overwrite)")
    dst.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    src_digest, count = rewrite(src, dst, compress=(mode == "compress"))

    if mode == "decompress" and sidecar(src).exists():
        expected = json.loads(sidecar(src).read_text())["stream_sha256"]
        if src_digest != expected:
            dst.unlink()
            raise RuntimeError(f"{src}: digest does not match its sidecar; file was corrupted in transit")

    dst_digest, dst_count = stream_digest(dst)
    if (dst_digest, dst_count) != (src_digest, count):
        dst.unlink()
        raise RuntimeError(f"{src}: verification failed ({count} vs {dst_count} messages); output removed")

    if mode == "compress":
        sidecar(dst).write_text(json.dumps({
            "source": src.name,
            "source_bytes": src.stat().st_size,
            "messages": count,
            "stream_sha256": src_digest,
        }, indent=2) + "\n")

    src_bytes, dst_bytes = src.stat().st_size, dst.stat().st_size
    if delete_source:
        src.unlink()
        sidecar(src).unlink(missing_ok=True)
    return (f"{src.name} -> {dst} | {count} msgs | {src_bytes / 1e9:.2f} GB -> {dst_bytes / 1e9:.2f} GB "
            f"| {time.time() - t0:.0f} s | verified{' | source deleted' if delete_source else ''}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("mode", choices=["compress", "decompress"])
    parser.add_argument("bags", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, help="write outputs here instead of next to each input")
    parser.add_argument("--jobs", type=int, default=1, help="bags to process in parallel (BZ2 is CPU-bound)")
    parser.add_argument("--delete-source", action="store_true", help="delete each input after its output verifies")
    parser.add_argument("--overwrite", action="store_true", help="replace existing outputs")
    args = parser.parse_args(argv)

    failures = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(process, bag, args.mode, args.out_dir, args.delete_source, args.overwrite): bag
            for bag in args.bags
        }
        for future in as_completed(futures):
            try:
                print(future.result(), flush=True)
            except Exception as exc:  # report every failure, keep processing the rest
                failures += 1
                print(f"FAILED {futures[future]}: {exc}", file=sys.stderr, flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

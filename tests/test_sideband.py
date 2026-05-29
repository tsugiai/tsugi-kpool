"""Sideband proof-of-life tests. Week 1-2 milestone: two-node heartbeat
on localhost demonstrates measured drift.
"""
import asyncio
import json
import time

import pytest

from tsugi_kpool.config import KPoolLoraConfig
from tsugi_kpool.sideband import Sideband, _MAX_FRAME_BYTES


def _heartbeat_frame(sender_id: str) -> bytes:
    payload = {
        "sender_id": sender_id,
        "ts_monotonic_ns": time.monotonic_ns(),
        "buffer_fill": {},
    }
    return json.dumps(payload).encode() + b"\n"


@pytest.mark.asyncio
async def test_two_node_drift_measured_on_localhost() -> None:
    cfg_a = KPoolLoraConfig(
        sideband_enabled=True,
        aggregation_mode="buffer_convergence",
        sideband_addr="tcp://127.0.0.1:51900",
        sideband_peers=("tcp://127.0.0.1:51901",),
        sideband_heartbeat_ms=20,
    )
    cfg_b = KPoolLoraConfig(
        sideband_enabled=True,
        aggregation_mode="buffer_convergence",
        sideband_addr="tcp://127.0.0.1:51901",
        sideband_peers=("tcp://127.0.0.1:51900",),
        sideband_heartbeat_ms=20,
    )
    a = Sideband(cfg_a, sender_id="node-a")
    b = Sideband(cfg_b, sender_id="node-b")
    await a.start()
    await b.start()
    try:
        await asyncio.sleep(0.2)
        drift_a_to_b = b.drift_ms("node-a")
        drift_b_to_a = a.drift_ms("node-b")
        assert drift_a_to_b != float("inf"), "node-b did not receive any node-a heartbeat"
        assert drift_b_to_a != float("inf"), "node-a did not receive any node-b heartbeat"
        # localhost loop should give sub-50ms drift in any sane environment
        assert abs(drift_a_to_b) < 50
        assert abs(drift_b_to_a) < 50
    finally:
        await a.stop()
        await b.stop()


@pytest.mark.asyncio
async def test_unknown_peer_source_rejected() -> None:
    # Allow-list contains only a non-loopback peer, so a localhost
    # connection's source (127.0.0.1) is outside it and must be dropped.
    cfg = KPoolLoraConfig(
        sideband_enabled=True,
        aggregation_mode="buffer_convergence",
        sideband_addr="tcp://127.0.0.1:51902",
        sideband_peers=("tcp://10.255.255.1:51999",),
    )
    sb = Sideband(cfg, sender_id="listener")
    await sb.start()
    try:
        _reader, writer = await asyncio.open_connection("127.0.0.1", 51902)
        writer.write(_heartbeat_frame("attacker"))
        await writer.drain()
        writer.close()
        await asyncio.sleep(0.1)
        assert sb.drift_ms("attacker") == float("inf")
        assert sb.snapshot() == {}
    finally:
        await sb.stop()


@pytest.mark.asyncio
async def test_allowed_peer_source_accepted() -> None:
    # Loopback IS in the allow-list here, so a localhost heartbeat is
    # accepted and recorded (positive control for the source check).
    cfg = KPoolLoraConfig(
        sideband_enabled=True,
        aggregation_mode="buffer_convergence",
        sideband_addr="tcp://127.0.0.1:51903",
        sideband_peers=("tcp://127.0.0.1:51999",),
    )
    sb = Sideband(cfg, sender_id="listener")
    await sb.start()
    try:
        _reader, writer = await asyncio.open_connection("127.0.0.1", 51903)
        writer.write(_heartbeat_frame("peer-1"))
        await writer.drain()
        writer.close()
        await asyncio.sleep(0.1)
        assert sb.drift_ms("peer-1") != float("inf")
    finally:
        await sb.stop()


@pytest.mark.asyncio
async def test_oversized_frame_rejected() -> None:
    cfg = KPoolLoraConfig(
        sideband_enabled=True,
        aggregation_mode="buffer_convergence",
        sideband_addr="tcp://127.0.0.1:51904",
        sideband_peers=("tcp://127.0.0.1:51999",),
    )
    sb = Sideband(cfg, sender_id="listener")
    await sb.start()
    try:
        _reader, writer = await asyncio.open_connection("127.0.0.1", 51904)
        # More than the frame limit with no separator -> server rejects.
        writer.write(b"x" * (_MAX_FRAME_BYTES + 1024))
        await writer.drain()
        writer.close()
        await asyncio.sleep(0.1)
        assert sb.snapshot() == {}
    finally:
        await sb.stop()


@pytest.mark.asyncio
async def test_malformed_payload_rejected() -> None:
    cfg = KPoolLoraConfig(
        sideband_enabled=True,
        aggregation_mode="buffer_convergence",
        sideband_addr="tcp://127.0.0.1:51905",
        sideband_peers=("tcp://127.0.0.1:51999",),
    )
    sb = Sideband(cfg, sender_id="listener")
    await sb.start()
    try:
        _reader, writer = await asyncio.open_connection("127.0.0.1", 51905)
        writer.write(b"this is not json\n")
        await writer.drain()
        writer.close()
        await asyncio.sleep(0.1)
        assert sb.snapshot() == {}
    finally:
        await sb.stop()

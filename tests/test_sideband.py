"""Sideband proof-of-life tests. Week 1-2 milestone: two-node heartbeat
on localhost demonstrates measured drift.
"""
import asyncio

import pytest

from tsugi_kpool.config import KPoolLoraConfig
from tsugi_kpool.sideband import Sideband


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

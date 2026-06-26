"""Phase-correction sideband communication channel.

A low-bandwidth TCP channel between training nodes carrying phase-drift
telemetry, deliberately separate from the NCCL gradient data plane.

Design notes:
    - Single-direction heartbeat: each node periodically emits its local
      monotonic-clock timestamp and per-adapter buffer-fill state.
    - Peers compute drift = abs(local_recv_ns - peer_send_ns) / 1e6 ms.
      The drift number is one-way wall-clock (clocks are not NTP-aligned
      across machines, so this is plesiochronous drift not absolute
      offset).
    - Bandwidth budget is sub-100KB/sec per peer; should never compete
      with NCCL for the InfiniBand fabric.
    - Implementation uses asyncio + plain TCP sockets. UDP is an option
      if heartbeat jitter dominates; tracked as Phase 2 work.
"""
from __future__ import annotations

import asyncio
import json
import socket
import time
from dataclasses import dataclass, asdict

from tsugi_kpool.config import KPoolLoraConfig, _split_tcp_addr

# Hard caps on inbound heartbeat frames. The sideband is a trusted-fabric
# control plane carrying tiny payloads (sender id + timestamp + a small
# per-adapter occupancy map); anything larger is treated as malformed or
# hostile and dropped before it can grow memory or poison drift state.
_MAX_FRAME_BYTES = 65536
_MAX_SENDER_ID_LEN = 256
_MAX_BUFFER_FILL_ENTRIES = 4096


@dataclass
class HeartbeatMessage:
    sender_id: str
    ts_monotonic_ns: int
    buffer_fill: dict[int, int]  # adapter_idx -> current buffer occupancy


# Low-bandwidth TCP control channel, logically distinct from the gradient
# data plane: heartbeat traffic occupies its own TCP port, separate from
# the NCCL gradient transport (RDMA/IB). It carries per-adapter
# buffer-fill telemetry and inter-node timing drift, parallel to (not
# displacing) the gradient-aggregation path.
class Sideband:
    """TCP sideband. Two-node is supported today; N-node is a planned
    extension."""

    def __init__(self, config: KPoolLoraConfig, sender_id: str) -> None:
        self.config = config
        self.sender_id = sender_id
        self._peer_last_ts: dict[str, int] = {}
        self._peer_drift_ms: dict[str, float] = {}
        self._allowed_hosts: set[str] = set()
        self._running = False
        self._tasks: list[asyncio.Task[None]] = []

    async def start(self) -> None:
        if not self.config.sideband_enabled:
            return
        self._running = True
        self._allowed_hosts = self._compute_allowed_hosts()
        host, port = self._parse_addr(self.config.sideband_addr)
        server = await asyncio.start_server(
            self._handle_peer, host, port, limit=_MAX_FRAME_BYTES
        )
        self._tasks.append(asyncio.create_task(server.serve_forever()))
        self._tasks.append(asyncio.create_task(self._heartbeat_loop()))

    async def stop(self) -> None:
        self._running = False
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass

    def drift_ms(self, peer_id: str) -> float:
        """Return last measured drift to a specific peer, or +inf if no
        heartbeat has been received from that peer yet."""
        return self._peer_drift_ms.get(peer_id, float("inf"))

    def max_drift_ms_across_peers(self) -> float:
        """Conservative bound: largest drift currently observed across
        all known peers. Used by the aggregator's HOLD/FIRE decision.

        Semantics:
            * No peers configured at all (single-node operation): returns
              0.0. With zero peers there is vacuously no drift; returning
              +inf would force the aggregator into permanent HOLD which is
              wrong for single-node + buffer-convergence operation.
            * Peers configured but no heartbeats received yet: returns
              +inf (conservative; we cannot prove phase lock).
            * Peers configured and at least one heartbeat seen: returns
              max(observed drifts).
        """
        if not self.config.sideband_peers:
            return 0.0
        if not self._peer_drift_ms:
            return float("inf")
        return max(self._peer_drift_ms.values())

    def peers_within_drift(self, threshold_ms: float) -> list[str]:
        """Return the subset of peer ids whose latest measured drift is
        at or below `threshold_ms`. Used by backend.all_reduce_subset
        when restricting NCCL all-reduce to in-phase peers (Phase 2)."""
        return [p for p, d in self._peer_drift_ms.items() if d <= threshold_ms]

    @staticmethod
    def _parse_addr(addr: str) -> tuple[str, int]:
        return _split_tcp_addr("sideband address", addr)

    def _compute_allowed_hosts(self) -> set[str]:
        """Source-address allow-list derived from `config.sideband_peers`.
        Inbound heartbeats whose source host is not in this set are dropped.
        The sideband is a trusted-fabric control plane, so this is a coarse
        network-level guard (peer identity by source address), not a
        cryptographic one; message authentication is planned for a later
        release. With no peers configured there is no legitimate inbound
        peer, so every connection is rejected."""
        allowed: set[str] = set()
        for peer in self.config.sideband_peers:
            try:
                host, _ = self._parse_addr(peer)
            except ValueError:
                continue
            allowed.add(host)
            try:
                # Best-effort: match a peer configured by hostname against
                # the source IP we actually observe on connect.
                _, _, ips = socket.gethostbyname_ex(host)
                allowed.update(ips)
            except OSError:
                pass
        return allowed

    def _peer_allowed(self, src_host: str | None) -> bool:
        if src_host is None:
            return False
        return src_host in self._allowed_hosts

    # The heartbeat payload carries `ts_monotonic_ns` (the source
    # node's monotonic-clock timestamp at send time), and the receiver
    # computes `drift = abs(local_recv_ns - peer_send_ns)` in ms. This
    # is phase-drift information (a wall-clock timestamp delta carried on
    # a separate low-bandwidth signaling channel), as opposed to the
    # step-count / token-count coordination metadata that Decoupled
    # DiLoCo (arXiv:2604.21428) carries on the data channel.
    async def _heartbeat_loop(self) -> None:
        while self._running:
            for peer in self.config.sideband_peers:
                await self._send_heartbeat(peer)
            await asyncio.sleep(self.config.sideband_heartbeat_ms / 1000.0)

    async def _send_heartbeat(self, peer: str) -> None:
        host, port = self._parse_addr(peer)
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=0.5
            )
        except (OSError, asyncio.TimeoutError):
            return
        msg = HeartbeatMessage(
            sender_id=self.sender_id,
            ts_monotonic_ns=time.monotonic_ns(),
            buffer_fill={},  # populated from aggregator state in runtime integration
        )
        try:
            writer.write(json.dumps(asdict(msg)).encode() + b"\n")
            await writer.drain()
        except (OSError, ConnectionError):
            return
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (OSError, ConnectionError):
                pass

    async def _handle_peer(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        local_recv_ns = time.monotonic_ns()
        # Source-address allow-list: drop connections from hosts that are
        # not configured peers before reading anything from the wire.
        peername = writer.get_extra_info("peername")
        src_host = peername[0] if peername else None
        if not self._peer_allowed(src_host):
            writer.close()
            return
        try:
            # `limit=_MAX_FRAME_BYTES` on the server bounds the read buffer;
            # an over-long line without a separator raises here.
            line = await reader.readline()
        except (OSError, ConnectionError, ValueError, asyncio.LimitOverrunError):
            writer.close()
            return
        if not line:
            writer.close()
            return
        try:
            payload = json.loads(line.decode())
            msg = HeartbeatMessage(**payload)
        except (json.JSONDecodeError, TypeError, ValueError):
            writer.close()
            return
        # Bound the fields we key state on so a malformed/hostile frame
        # cannot grow the drift table or its keys without limit.
        if (
            not isinstance(msg.sender_id, str)
            or len(msg.sender_id) > _MAX_SENDER_ID_LEN
            or not isinstance(msg.ts_monotonic_ns, int)
            or not isinstance(msg.buffer_fill, dict)
            or len(msg.buffer_fill) > _MAX_BUFFER_FILL_ENTRIES
        ):
            writer.close()
            return
        # Plesiochronous drift: one-way wall-clock difference. Clocks are
        # not NTP-aligned so the absolute value is what matters; we want
        # to know "how out-of-phase are we" not "what's the offset".
        drift_ns = abs(local_recv_ns - msg.ts_monotonic_ns)
        self._peer_last_ts[msg.sender_id] = msg.ts_monotonic_ns
        self._peer_drift_ms[msg.sender_id] = drift_ns / 1_000_000.0
        writer.close()

    def snapshot(self) -> dict[str, float]:
        """Return a defensive copy of the current per-peer drift table.
        Used by diagnostics."""
        return dict(self._peer_drift_ms)

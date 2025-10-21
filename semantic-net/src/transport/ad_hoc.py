"""Ad‑hoc UDP transport implementation.

This module implements a minimal UDP transport class for sending and
receiving byte payloads.  It is intended for experimentation and
demonstration rather than production use; features such as
retransmissions, congestion control and security are not provided.
Nevertheless, this provides a starting point for testing the semantic
communication pipeline over a custom network topology.
"""

from __future__ import annotations

import socket
from typing import Optional, Tuple


class AdHocTransport:
    """A simple UDP transport with send and receive capabilities."""

    def __init__(self, local_port: int = 5000, buffer_size: int = 4096) -> None:
        self.local_port = local_port
        self.buffer_size = buffer_size
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Bind to the local port so we can receive messages.
        self.sock.bind(("", self.local_port))

    def send(self, data: bytes, address: Tuple[str, int]) -> None:
        """Send a datagram to the given address.

        Parameters
        ----------
        data : bytes
            The payload to transmit.
        address : tuple
            A (host, port) tuple specifying the destination.
        """
        self.sock.sendto(data, address)

    def receive(self) -> Tuple[bytes, Tuple[str, int]]:
        """Receive a datagram from the socket.

        Returns
        -------
        tuple
            A tuple ``(data, addr)`` where ``data`` is the received
            payload and ``addr`` is the sender's address.
        """
        data, addr = self.sock.recvfrom(self.buffer_size)
        return data, addr

    def close(self) -> None:
        """Close the underlying socket."""
        self.sock.close()

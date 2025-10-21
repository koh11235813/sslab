"""Transport layer abstractions for the semantic network.

This package encapsulates communication mechanisms used for sending
model updates or semantic codes between federated clients and the
server.  The base implementation provided here uses UDP sockets in an
ad‑hoc manner; however, the API is intentionally generic so that
alternative transports (e.g. TCP, LoRa, Bluetooth) can be plugged in
without changing higher‑level code.
"""

__all__ = ["ad_hoc", "proto"]
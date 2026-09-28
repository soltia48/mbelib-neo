#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Small, synchronous AMBE-3000R packet client.

Only volatile codec settings are changed. Nothing is written to EEPROM.
Wire PCM is signed 16-bit big endian, 160 samples per 20 ms frame.
"""
import struct
import time


class ProtocolError(RuntimeError):
    """Unexpected, missing, or truncated device response."""


def packet(kind, payload):
    if kind not in (0, 1, 2) or not 0 < len(payload) <= 1024:
        raise ValueError("invalid packet kind or payload size")
    return b"\x61" + struct.pack(">HB", len(payload), kind) + payload


def unpack_raw(raw):
    """Ten rate-17 bytes -> 80 binary parameters, in priority order."""
    if len(raw) != 10:
        raise ValueError("expected 10 raw AMBE+ bytes")
    positions = [r + c for c in range(7)
                 for r in (range(0, 56, 7) if c == 6 else
                           (0, 7, 14, 21, 28, 35, 42, 49, 56, 62, 68, 74))]
    return [(raw[i // 8] >> (7 - i % 8)) & 1 for i in positions]


def pack_raw(bits):
    """80 binary parameters in priority order -> ten rate-17 bytes."""
    if len(bits) != 80 or any(b not in (0, 1) for b in bits):
        raise ValueError("expected 80 binary parameters")
    positions = [r + c for c in range(7)
                 for r in (range(0, 56, 7) if c == 6 else
                           (0, 7, 14, 21, 28, 35, 42, 49, 56, 62, 68, 74))]
    raw = bytearray(10)
    for bit, i in zip(bits, positions):
        raw[i // 8] |= bit << (7 - i % 8)
    return bytes(raw)


class ThumbDV:
    def __init__(self, device, baud=460800, timeout=2.0):
        import serial  # Hardware-only optional dependency.
        self.port = serial.Serial(device, baudrate=baud, timeout=timeout,
                                  write_timeout=timeout, exclusive=True)
        self.timeout = timeout
        self.controls = []
        self.port.reset_input_buffer()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.port.close()

    def _read(self, count):
        chunks = bytearray()
        deadline = time.monotonic() + self.timeout
        while len(chunks) < count:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProtocolError("device response timed out")
            self.port.timeout = remaining
            chunk = self.port.read(count - len(chunks))
            if not chunk:
                raise ProtocolError("missing or truncated device response")
            chunks.extend(chunk)
        return bytes(chunks)

    def transact(self, kind, payload, expected_kind):
        data = packet(kind, payload)
        if self.port.write(data) != len(data):
            raise ProtocolError("incomplete device write")
        header = self._read(4)
        size = int.from_bytes(header[1:3], "big")
        if header[0] != 0x61 or header[3] != expected_kind or not 0 < size <= 1024:
            raise ProtocolError("invalid response header: " + header.hex())
        return self._read(size)

    def control(self, payload):
        response = self.transact(0, payload, 0)
        self.controls.append({"tx": payload.hex(), "rx": response.hex()})
        if response != payload[:1] + b"\x00":
            raise ProtocolError("control rejected: " + response.hex())

    def identity(self):
        info = {}
        for key, field in (("product", 0x30), ("version", 0x31)):
            response = self.transact(0, bytes([field]), 0)
            if response[:1] != bytes([field]) or not response.endswith(b"\x00"):
                raise ProtocolError("invalid identity response")
            info[key] = response[1:-1].decode("ascii")
        return info

    def configure(self, rate):
        if rate not in (17, 26):
            raise ValueError("only AMBE+ raw 4000 (17) and 4000/2400 (26) supported")
        self.control(bytes([0x09, rate]))
        self.control(b"\x0b\x03")
        self.control(b"\x32\x00")
        self.control(b"\x4b\x00\x00")  # Explicit unity packet PCM input/output gain.
        self.control(b"\x05\x00\x00")
        self.control(b"\x06\x00\x00")
        self.control(b"\x15\x00\x00")  # Channel data only.
        self.control(b"\x16\x00\x05")  # PCM, decoder mode, sample count.
        self.bits = 80 if rate == 17 else 128

    def reset_encoder(self):
        self.control(b"\x0b\x01")
        self.control(b"\x05\x00\x00")

    def reset_decoder(self):
        self.control(b"\x0b\x02")
        self.control(b"\x06\x00\x00")

    def encode(self, pcm_be):
        if len(pcm_be) != 320:
            raise ValueError("expected 160 PCM16 samples")
        payload = self.transact(2, b"\x00\xa0" + pcm_be, 1)
        if payload[:1] == b"\x40":
            payload = payload[1:]
        if len(payload) != 2 + self.bits // 8 or payload[:2] != bytes([1, self.bits]):
            raise ProtocolError("invalid channel response: " + payload.hex())
        return payload[2:]

    def decode(self, channel):
        if len(channel) != self.bits // 8:
            raise ValueError("incorrect channel frame size")
        payload = self.transact(1, bytes([1, self.bits]) + channel, 2)
        pos = 0
        pcm = mode = None
        seen_count = False
        while pos < len(payload):
            field = payload[pos]
            pos += 1
            if field == 0x40:
                continue
            if field == 0:
                if pcm is not None or payload[pos:pos + 1] != b"\xa0" or pos + 321 > len(payload):
                    raise ProtocolError("invalid PCM field")
                pcm = payload[pos + 1:pos + 321]
                pos += 321
            elif field == 2:
                if mode is not None or pos + 2 > len(payload):
                    raise ProtocolError("invalid decoder mode field")
                mode = int.from_bytes(payload[pos:pos + 2], "big")
                pos += 2
            elif field == 3:
                if seen_count or payload[pos:pos + 1] != b"\xa0":
                    raise ProtocolError("invalid sample count field")
                pos += 1
                seen_count = True
            else:
                raise ProtocolError("unexpected speech field: " + hex(field))
        if pcm is None or mode is None:
            raise ProtocolError("missing speech fields")
        return pcm, mode

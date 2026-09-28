#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Offline CLI framing and WAV tests; no hardware or external Python packages."""
import io
from pathlib import Path
import re
import struct
import subprocess
import sys
import unittest
import wave

DECODER = sys.argv[1]
sys.argv[1:] = []


class DecodeCLI(unittest.TestCase):
    def run_decoder(self, data):
        return subprocess.run([DECODER], input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def test_empty(self):
        result = self.run_decoder(b"")
        self.assertEqual(result.returncode, 0, result.stderr)
        with wave.open(io.BytesIO(result.stdout), "rb") as wav:
            self.assertEqual((wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()),
                             (1, 2, 8000, 0))

    def test_all_partial_lengths(self):
        for size in range(1, 16):
            result = self.run_decoder(bytes(size))
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, b"")
            self.assertIn(b"partial", result.stderr)

    def test_synthetic_reference_vectors(self):
        text = (Path(__file__).parent / "data/ambe4000_vectors.h").read_text()
        groups = re.findall(r"\{([^{}]+)\}", text)
        channels = []
        for group in groups:
            values = re.findall(r"0x([0-9a-f]{2})", group)
            if len(values) == 16:
                channels.append(bytes(int(v, 16) for v in values))
        self.assertEqual(len(channels), 27)
        data = b"".join(channels)
        result = self.run_decoder(data)
        self.assertEqual(result.returncode, 0, result.stderr)
        with wave.open(io.BytesIO(result.stdout), "rb") as wav:
            self.assertEqual(wav.getnframes(), 27*160)
            pcm = struct.unpack("<" + "h" * wav.getnframes(), wav.readframes(wav.getnframes()))
        self.assertGreater(max(abs(v) for v in pcm), 100)
        self.assertLessEqual(max(abs(v) for v in pcm), 31128)
        self.assertEqual(self.run_decoder(data).stdout, result.stdout)
        result = self.run_decoder(data + b"\0")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b"")


if __name__ == "__main__":
    unittest.main()

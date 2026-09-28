#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Hardware-free protocol and experiment tests; pyserial is not required."""
import io
import importlib.util
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import wave
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "ambe4000"))
from thumbdv import ProtocolError, ThumbDV, pack_raw, packet, unpack_raw
import codec
import fields


class Port:
    def __init__(self, response, chunk=1024):
        self.response = io.BytesIO(response)
        self.chunk = chunk
        self.writes = []

    def read(self, count):
        return self.response.read(min(count, self.chunk))

    def write(self, data):
        self.writes.append(data)
        return len(data)


def client(response, chunk=1024):
    dv = ThumbDV.__new__(ThumbDV)
    dv.port = Port(response, chunk)
    dv.timeout = 1.0
    dv.bits = 128
    dv.controls = []
    return dv


class ProtocolTests(unittest.TestCase):
    def test_raw_permutation(self):
        raw = bytes.fromhex("dcac2beb853b6edd04bc")
        logical = bytes.fromhex("8b9d3f1e1d99bfad3860")
        expected = [(b >> shift) & 1 for b in logical for shift in range(7, -1, -1)]
        self.assertEqual(unpack_raw(raw), expected)
        self.assertEqual(pack_raw(expected), raw)
        for i in range(80):
            bits = [int(j == i) for j in range(80)]
            self.assertEqual(unpack_raw(pack_raw(bits)), bits)
        for raw in (bytes(9), bytes(11)):
            with self.assertRaises(ValueError):
                unpack_raw(raw)
        for bits in ([0] * 79, [0] * 79 + [2]):
            with self.assertRaises(ValueError):
                pack_raw(bits)

    def test_fragmented_pcm_with_mode(self):
        pcm = struct.pack(">160h", *range(-80, 80))
        for prefix in (b"", b"\x40"):
            for count in (b"", b"\x03\xa0"):
                dv = client(packet(2, prefix + b"\x00\xa0" + pcm + b"\x02\x00\x22" + count), 3)
                self.assertEqual(dv.decode(bytes(16)), (pcm, 0x22))
                self.assertEqual(dv.port.writes, [packet(1, b"\x01\x80" + bytes(16))])

    def test_invalid_packet(self):
        for response in (b"", b"\x61", b"\x60\x00\x01\x02x",
                         b"\x61\x04\x01\x02", packet(1, b"x"), b"\x61\x00\x02\x02x"):
            with self.subTest(response=response), self.assertRaises(ProtocolError):
                client(response).decode(bytes(16))

    def test_invalid_speech_fields(self):
        pcm = b"\x00\xa0" + bytes(320)
        for payload in (pcm, pcm + b"\x02", pcm + b"\x02\x00\x00\x02\x00\x00",
                        pcm + b"\x02\x00\x00\x03\x9f", pcm + pcm,
                        b"\x00\x9f" + bytes(318) + b"\x02\x00\x00", b"\x88"):
            with self.subTest(payload=payload[:5]), self.assertRaises(ProtocolError):
                client(packet(2, payload)).decode(bytes(16))

    def test_encoded_count_and_control_ack(self):
        dv = client(packet(1, b"\x01\x80" + bytes(range(16))))
        self.assertEqual(dv.encode(bytes(320)), bytes(range(16)))
        self.assertEqual(dv.port.writes, [packet(2, b"\x00\xa0" + bytes(320))])
        with self.assertRaises(ProtocolError):
            client(packet(1, b"\x01\x50" + bytes(10))).encode(bytes(320))
        client(packet(0, b"\x09\x00")).control(b"\x09\x1a")
        with self.assertRaises(ProtocolError):
            client(packet(0, b"\x09\x01")).control(b"\x09\x1a")


class ExperimentTests(unittest.TestCase):
    def test_field_partition_and_known_frame(self):
        positions = [bit for field in fields.FIELDS.values() for bit in field]
        self.assertEqual(sorted(positions), list(range(80)))
        values = fields.indices(unpack_raw(bytes.fromhex("dcac2beb853b6edd04bc")))
        self.assertEqual(values["pitch"], 68)
        bits = fields.reference(80, 40)
        self.assertEqual(fields.indices(bits)["pitch"], 80)
        self.assertEqual(fields.indices(bits)["gain"], 40)
        for name, positions in fields.FIELDS.items():
            for value in (0, (1 << len(positions)) - 1):
                before = bits.copy()
                fields.set_field(bits, name, value)
                self.assertEqual(fields.indices(bits)[name], value)
                for index in set(range(80)) - set(positions):
                    self.assertEqual(bits[index], before[index])
            before = bits.copy()
            for value in (-1, 1 << len(positions)):
                with self.assertRaises(ValueError):
                    fields.set_field(bits, name, value)
                self.assertEqual(bits, before)

    def test_impulse_has_exactly_one_changed_frame_and_two_controls(self):
        base = fields.reference()
        cases = list(fields.research_cases("impulse", base, 12))
        self.assertEqual(len(cases), 75)
        self.assertEqual(cases[0][0], ("control", 0))
        self.assertEqual(cases[-1][0], ("control", 1))
        for (name, bit), changed, stream in cases:
            self.assertEqual(len(stream), 24)
            self.assertEqual(stream[:12], [base] * 12)
            self.assertEqual(stream[13:], [base] * 11)
            if name == "bit":
                self.assertEqual([i for i in range(80) if changed[i] != base[i]], [bit])
                self.assertEqual(stream[12], changed)
            else:
                self.assertEqual(stream, [base] * 24)
        extra = list(fields.research_cases("extra", base, 2))
        self.assertEqual(len(extra), 274)
        hoc = list(fields.research_cases("hoc-impulse", base, 12))
        self.assertEqual(len(hoc), 130)
        self.assertEqual(hoc[1][0], ("hoc0", 0))
        self.assertEqual(hoc[128][0], ("hoc3", 31))
        self.assertEqual(hoc[1][2][13:], [base] * 11)
        self.assertEqual(fields.indices(base), fields.indices(fields.reference()))

    def test_configuration_explicitly_resets_pcm_gain(self):
        controls = [b"\x09\x11", b"\x0b\x03", b"\x32\x00", b"\x4b\x00\x00",
                    b"\x05\x00\x00", b"\x06\x00\x00", b"\x15\x00\x00", b"\x16\x00\x05"]
        dv = client(b"".join(packet(0, c[:1] + b"\x00") for c in controls))
        dv.configure(17)
        self.assertEqual(dv.bits, 80)
        self.assertEqual(dv.port.writes, [packet(0, c) for c in controls])

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "optional NumPy is unavailable")
    def test_harmonic_fit_and_known_decay(self):
        import numpy as np
        from analyze import harmonic_logs, impulse_report, fit_decay
        n = np.arange(160)
        tones = np.array([np.cos(2 * np.pi * 125 * h * n / 8000) for h in (1, 2)])
        base = fields.reference()
        cases = list(fields.research_cases("impulse", base, 12))
        amp = np.full((75, 24, 2), 32.)
        for frame in range(12, 24):
            delta = .5 * .8 ** (frame - 12)
            amp[5, frame] *= 2 ** np.array([delta, -delta])  # bit 10
        pcm = (amp @ tones).reshape(75, -1)
        data = dict(labels=np.array([case[0] for case in cases]),
                    bits=np.array([case[1] for case in cases]), base=np.array(base),
                    pcm=pcm, modes=np.full((75, 24), 2))
        logs, residual, condition = harmonic_logs(pcm, 125, 2)
        np.testing.assert_allclose(logs, np.log2(amp), atol=1e-12)
        self.assertLess(residual.max(), 1e-10)
        series = np.arange(1, 13).reshape(3, 4)[:, None, :] * .8 ** np.arange(8)[None, :, None]
        fit = fit_decay(series)
        self.assertAlmostEqual(fit["observed_decay"], .8, places=4)
        self.assertLess(fit["relative_fit_error"], 1e-6)
        self.assertFalse(fit["search_boundary"])
        with self.assertRaises(ValueError):
            fit_decay(np.zeros((3, 8, 4)))
        report = impulse_report(data, 125, 2)
        self.assertAlmostEqual(report["cases"][5]["observed_tail_ar1"], .8, places=10)
        self.assertEqual(report["control_drift_rms_log2"], 0)
        for frequency, harmonics in ((4000, 1), (0, 2), (125, 80)):
            with self.assertRaises(ValueError):
                harmonic_logs(pcm, frequency, harmonics)
        with self.assertRaises(ValueError):
            impulse_report(dict(data, labels=data["labels"][:-1]), 125, 2)
        data["modes"][0, 0] = 0x22
        with self.assertRaises(ValueError):
            impulse_report(data, 125, 2)
        from analyze import main
        from contextlib import redirect_stderr
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "device.json").write_text('{"status":"incomplete","experiment":"impulse"}')
            args = ["analyze.py", str(root / "research.npz"), "--fundamental-hz", "125",
                    "--harmonics", "2", "--out", str(root / "report.json")]
            with patch.object(sys, "argv", args), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main()
            self.assertEqual(error.exception.code, 2)
            self.assertFalse((root / "report.json").exists())

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "optional NumPy is unavailable")
    def test_voicing_analysis_recovers_independent_tones(self):
        import numpy as np
        from analyze import voicing_report
        base = fields.reference()
        cases = list(fields.research_cases("voicing", base, 16))
        n = np.arange(2560)
        signals = []
        for label, bits, stream in cases:
            code = fields.indices(bits)["voicing"]
            # Off-center tones provide a reproducible nonperiodic-band control;
            # this tests the estimator, not a claim about hardware excitation.
            frequencies = [500 * (h + 1) + (0 if code & (1 << (5 - h)) else 100)
                           for h in range(6)]
            signals.append(sum(100 * np.cos(2 * np.pi * f * n / 8000) for f in frequencies))
        data = dict(labels=np.array([case[0] for case in cases]),
                    bits=np.array([case[1] for case in cases]), base=np.array(base),
                    pcm=np.array(signals), modes=np.full((66, 16), 2))
        report = voicing_report(data, 500, 6)
        self.assertEqual(report["mismatches"], [])
        self.assertEqual(report["single_bit_masks"], {str(50 + h): [h + 1] for h in range(6)})
        data["bits"][1, 0] ^= 1
        with self.assertRaises(ValueError):
            voicing_report(data, 500, 6)


class AudioTests(unittest.TestCase):
    def test_wav_encode_padding_endian_and_flush(self):
        class Encoder:
            def __init__(self):
                self.frames = []

            def encode(self, pcm):
                self.frames.append(pcm)
                return bytes([len(self.frames)]) * 16

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.wav"
            with wave.open(str(source), "wb") as wav:
                wav.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
                wav.writeframes(struct.pack("<3h", -32768, 1234, 32767))
            device = Encoder()
            report = codec.encode(device, source, root, 3)
            self.assertEqual(report["input_samples"], 3)
            self.assertEqual(report["channel_frames"], 4)
            self.assertEqual(device.frames[0], struct.pack(">3h", -32768, 1234, 32767) + bytes(314))
            self.assertEqual(device.frames[1:], [bytes(320)] * 3)
            self.assertEqual((root / "audio.ambe").stat().st_size, 64)

    def test_wav_decode_length_endian_and_invalid_mode(self):
        pcm = struct.pack(">160h", *range(-80, 80))

        class Decoder:
            def decode(self, channel):
                return pcm, 0x22 if channel[0] else 2

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.ambe"
            source.write_bytes(bytes(16) + bytes([1]) * 16)
            report = codec.decode(Decoder(), source, root)
            self.assertEqual(report["output_samples"], 320)
            self.assertEqual(report["data_invalid_frames"], 1)
            with wave.open(str(root / "audio.wav"), "rb") as wav:
                self.assertEqual(wav.getnframes(), 320)
                self.assertEqual(wav.getframerate(), 8000)
                self.assertEqual(wav.readframes(320), struct.pack("<160h", *range(-80, 80)) * 2)
            self.assertEqual(len((root / "frames.jsonl").read_text().splitlines()), 2)

    def test_truncated_channel_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.ambe"
            source.write_bytes(bytes(17))
            with self.assertRaises(ValueError):
                codec.decode(None, source, root)
            self.assertFalse((root / "audio.wav").exists())


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""ThumbDV-backed AMBE+ 4000/2400 WAV encoder/decoder (not software synthesis)."""
import argparse
import json
import struct
import wave
from pathlib import Path
from thumbdv import ThumbDV


def encode(device, source, out, flush_frames):
    with wave.open(str(source), "rb") as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getcomptype()) != (8000, 1, 2, "NONE"):
            raise ValueError("input must be 8000 Hz, mono, signed PCM16 WAV")
        samples = wav.getnframes()
        frames = actual_samples = 0
        with (out / "audio.ambe").open("xb") as encoded:
            while True:
                pcm = wav.readframes(160)
                if not pcm:
                    break
                if len(pcm) % 2:
                    raise ValueError("truncated PCM sample")
                actual_samples += len(pcm) // 2
                pcm = pcm.ljust(320, b"\x00")
                pcm_be = struct.pack(">160h", *struct.unpack("<160h", pcm))
                encoded.write(device.encode(pcm_be))
                frames += 1
            if actual_samples != samples:
                raise ValueError("truncated WAV data")
            for _ in range(flush_frames):
                encoded.write(device.encode(bytes(320)))
                frames += 1
    return {"input_samples": samples, "channel_frames": frames, "flush_frames": flush_frames,
            "format": "16 bytes per frame, 128 channel bits MSB first, rate index 26"}


def decode(device, source, out):
    if source.stat().st_size % 16:
        raise ValueError("channel input must contain whole 16-byte frames")
    frames = invalid = 0
    with source.open("rb") as channel, (out / "audio.wav").open("xb") as output:
        with wave.open(output, "wb") as wav, (out / "frames.jsonl").open("x") as log:
            wav.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
            while True:
                block = channel.read(16)
                if not block:
                    break
                pcm_be, mode = device.decode(block)
                wav.writeframesraw(struct.pack("<160h", *struct.unpack(">160h", pcm_be)))
                log.write(json.dumps({"frame": frames, "sample": frames * 160,
                                      "decoder_mode": mode, "data_invalid": bool(mode & 0x20)}) + "\n")
                frames += 1
                invalid += bool(mode & 0x20)
    return {"channel_frames": frames, "output_samples": frames * 160, "data_invalid_frames": invalid}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("encode", "decode"))
    parser.add_argument("--device", required=True)
    parser.add_argument("--baud", type=int, default=460800)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="new output directory")
    parser.add_argument("--flush-frames", type=int, default=3,
                        help="zero PCM frames appended when encoding (default 3); latency is not removed")
    args = parser.parse_args()
    if not 0 <= args.flush_frames <= 10:
        parser.error("--flush-frames must be between 0 and 10")
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"operation": args.operation, "rate_index": 26, "status": "incomplete"}
    with ThumbDV(args.device, args.baud) as device:
        report["device"] = device.identity()
        try:
            device.configure(26)
            if args.operation == "encode":
                report.update(encode(device, args.input, args.out, args.flush_frames))
            else:
                report.update(decode(device, args.input, args.out))
            report["status"] = "complete"
        finally:
            report["controls"] = device.controls
            (args.out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "controls"}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Evaluate archived AMBE speech using only a native decoder and local files.

Requires NumPy and pystoi; optional ASR requires faster-whisper and an already
cached tiny.en model. No serial port, audio input or network access is used.
Each corpus item needs NAME.ambe, NAME-reference.wav and NAME-hardware.wav.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import wave

import numpy as np
from pystoi import stoi


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def samples(path):
    with wave.open(str(path), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) != (1, 2, 8000, "NONE"):
            raise ValueError(f"not 8 kHz mono PCM16: {path}")
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(float)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path, help="directory containing corpus.json and archived audio")
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--library", type=Path, help="optional linked library to record in provenance")
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    parser.add_argument("--delay-samples", type=int, default=240, help="fixed source/output alignment, default 30 ms")
    parser.add_argument("--asr-cache", type=Path, help="optional local faster-whisper download_root")
    args = parser.parse_args()
    if args.delay_samples < 0:
        parser.error("delay must be nonnegative")
    corpus = json.loads((args.corpus / "corpus.json").read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    report = {
        "decoder_sha256": sha256(args.decoder), "corpus": corpus,
        "delay_samples": args.delay_samples, "extended_stoi": False,
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pystoi")},
        "status": "incomplete", "results": [],
    }
    if args.library:
        report["library_sha256"] = sha256(args.library)
    model = None
    if args.asr_cache:
        from faster_whisper import WhisperModel
        model = WhisperModel("tiny.en", device="cpu", compute_type="int8", cpu_threads=4,
                             download_root=str(args.asr_cache), local_files_only=True)
        report["versions"]["faster-whisper"] = importlib.metadata.version("faster-whisper")
        report["asr"] = {"model": "tiny.en", "compute_type": "int8", "beam_size": 5,
                         "language": "en", "condition_on_previous_text": False,
                         "weights": {str(p.relative_to(args.asr_cache)): sha256(p)
                                     for p in args.asr_cache.rglob("model.bin")}}
    try:
        for item in corpus:
            name = item["name"]
            # Corpus labels must be plain basenames, not paths.
            if not name or Path(name).name != name or name in (".", ".."):
                raise ValueError("invalid corpus label")
            raw = args.corpus / f"{name}.raw"
            if sha256(raw) != item["sha256"]:
                raise ValueError(f"corpus source hash mismatch: {name}")
            native = args.out / f"{name}-native.wav"
            encoded = args.corpus / f"{name}.ambe"
            with encoded.open("rb") as source, native.open("xb") as output:
                result = subprocess.run([str(args.decoder.resolve())], stdin=source, stdout=output,
                                        stderr=subprocess.PIPE, text=True, check=True)
            reference = args.corpus / f"{name}-reference.wav"
            x = samples(reference)
            if not np.array_equal(x, np.frombuffer(raw.read_bytes(), dtype="<i2")):
                raise ValueError(f"reference differs from corpus PCM: {name}")
            row = {"name": name, "channel_sha256": sha256(encoded), "decoder_log": result.stderr.strip(),
                   "reference_sha256": sha256(reference)}
            for label, path in (("native", native), ("hardware", args.corpus / f"{name}-hardware.wav")):
                y = samples(path)
                end = args.delay_samples + len(x)
                if len(y) < end:
                    raise ValueError(f"insufficient output for fixed alignment: {path}")
                row[label] = {"stoi": float(stoi(x, y[args.delay_samples:end], 8000, extended=False)),
                              "rms": float(np.sqrt(np.mean(y * y))), "sha256": sha256(path)}
                if model:
                    segments, _ = model.transcribe(str(path), language="en", beam_size=5,
                                                   condition_on_previous_text=False)
                    row[label]["transcript"] = " ".join(segment.text.strip() for segment in segments)
            if model:
                segments, _ = model.transcribe(str(reference), language="en", beam_size=5,
                                               condition_on_previous_text=False)
                row["reference_transcript"] = " ".join(segment.text.strip() for segment in segments)
            report["results"].append(row)
            print(json.dumps(row), flush=True)
        report["status"] = "complete"
    finally:
        (args.out / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

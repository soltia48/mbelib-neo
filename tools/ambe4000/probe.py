#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Collect synthetic AMBE+ stimuli and hardware responses; never uses a microphone."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from thumbdv import ThumbDV, pack_raw, unpack_raw
from fields import FIELDS, reference, research_cases

RESEARCH_EXPERIMENTS = ("impulse", "extra", "voicing", "gain", "hoc-impulse")


def stimulus():
    rng = np.random.default_rng(670026)
    signals, labels = [], []
    for frequency in (70, 85, 100, 125, 150, 200, 250, 300, 350):
        for shape in ("sine", "saw", "noise"):
            time = np.arange(160 * 12) / 8000
            if shape == "sine":
                signal = np.sin(2 * np.pi * frequency * time)
            elif shape == "saw":
                signal = sum(np.sin(2 * np.pi * frequency * k * time) / k
                             for k in range(1, int(3800 / frequency)))
            else:
                signal = rng.normal(size=len(time)) * .3
            signals.append(np.clip(signal * 4000, -32000, 32000).astype(np.int16))
            labels.append({"frequency": frequency, "shape": shape})
    return np.array(signals), labels


def channel_encode(bits):
    """Reference implementation of the K=7, rate-1/2 terminated code."""
    reg = 0
    coded = []
    for bit in list(bits[:36]) + [0] * 6:
        reg = ((reg << 1) | int(bit)) & 127
        coded.extend(bin(reg & polynomial).count("1") & 1 for polynomial in (0x4f, 0x6d))
    coded.extend(bits[36:])
    frame = np.empty(128, dtype=np.uint8)
    positions = [r + c for c in range(11)
                 for r in (range(0, 88, 11) if c == 10 else
                           (0, 11, 22, 33, 44, 55, 66, 77, 88, 98, 108, 118))]
    frame[positions] = coded
    return np.packbits(frame).tobytes()


def gf2_rank(bits):
    matrix = bits.copy()
    row = 0
    for col in range(matrix.shape[1]):
        candidates = np.flatnonzero(matrix[row:, col])
        if not len(candidates):
            continue
        pivot = row + candidates[0]
        matrix[[row, pivot]] = matrix[[pivot, row]]
        others = np.flatnonzero(matrix[:, col])
        matrix[others[others != row]] ^= matrix[row]
        row += 1
        if row == len(matrix):
            break
    return row


def collect_fec(device, out):
    pcm, labels = stimulus()
    np.save(out / "stimulus.npy", pcm)
    encoded = {}
    for rate in (17, 26):
        device.configure(rate)
        rows = []
        for signal in pcm:
            device.reset_encoder()
            rows.append([list(device.encode(frame.astype(">i2").tobytes()))
                         for frame in signal.reshape(-1, 160)])
        encoded[rate] = np.array(rows, dtype=np.uint8)
        np.save(out / ("encoded%d.npy" % rate), encoded[rate])
        print("Collected rate", rate, flush=True)
    raw = encoded[17].reshape(-1, 10)
    channel = encoded[26].reshape(-1, 16)
    source = np.array([unpack_raw(bytes(row)) for row in raw], dtype=np.uint8)
    generated = np.array([list(channel_encode(bits)) for bits in source], dtype=np.uint8)
    failures = int(np.unpackbits(generated ^ channel).sum())
    results = {"labels": labels, "frames": len(raw), "source_rank": gf2_rank(source),
               "compared_channel_bits": int(channel.size * 8), "mismatched_bits": failures,
               "pcm_sha256": hashlib.sha256(pcm.astype(">i2").tobytes()).hexdigest()}
    (out / "fec_results.json").write_text(json.dumps(results, indent=2) + "\n")
    if failures or results["source_rank"] != 80:
        raise RuntimeError("FEC verification failed or stimulus did not span all 80 bits")


def collect_parameters(device, out, base_hex, experiment, frames):
    bits = unpack_raw(bytes.fromhex(base_hex))
    device.configure(17)
    labels, inputs, pcm, modes = [], [], [], []
    values = range(128) if experiment == "pitch" else range(-1, 80)
    for value in values:
        test = bits.copy()
        if experiment == "pitch":
            for k, index in enumerate(list(range(6)) + [79]):
                test[index] = (value >> (6 - k)) & 1
        elif value >= 0:
            test[value] ^= 1
        device.reset_decoder()
        responses = [device.decode(pack_raw(test)) for _ in range(frames)]
        labels.append(value)
        inputs.append(test)
        pcm.append(np.frombuffer(b"".join(row[0] for row in responses), dtype=">i2"))
        modes.append([row[1] for row in responses])
        if value % 16 == 0:
            print("Collected", experiment, value, flush=True)
    np.savez(out / "parameters.npz", labels=labels, bits=inputs, pcm=pcm, modes=modes)


def set_index(bits, positions, value):
    for shift, index in enumerate(reversed(positions)):
        bits[index] = (value >> shift) & 1


def collect_models(device, out, frames, experiment):
    # Low gain and approximately flat coarse shape limit saturation during
    # artificial repeats. These indices are observations, not a decoder model.
    base = unpack_raw(bytes.fromhex("dcac2beb853b6edd04bc"))
    fields = [list(range(6)) + [79], list(range(6, 10)) + [56, 57],
              list(range(10, 17)) + [58, 59], list(range(17, 22)) + [60, 61]]
    fields += [list(range(22 + 4 * k, 26 + 4 * k)) + [62 + k] for k in range(4)]
    for positions, value in zip(fields, (100, 24, 169, 41, 18, 15, 16, 21)):
        set_index(base, positions, value)
    device.configure(17)
    cases = [(block, code) for block in range(4) for code in range(32)]
    if experiment == "transitions":
        cases = [(kind, code) for kind in ("pitch", "spectral") for code in range(4)]
    inputs, pcm, modes = [], [], []
    for group, code in cases:
        bits = base.copy()
        if experiment == "hoc":
            set_index(bits, fields[4 + group], code)
            stream = [bits] * frames
        else:
            if group == "pitch":
                set_index(bits, fields[0], 20)
            else:
                bits[11] ^= 1
            set_index(bits, [38, 39], code)
            stream = [base] * frames + [bits] * frames + [base] * frames
        device.reset_decoder()
        responses = [device.decode(pack_raw(frame)) for frame in stream]
        inputs.append(bits)
        pcm.append(np.frombuffer(b"".join(row[0] for row in responses), dtype=">i2"))
        modes.append([row[1] for row in responses])
        if code % 16 == 0:
            print("Collected", experiment, group, code, flush=True)
    np.savez(out / "models.npz", labels=cases, bits=inputs, base=base, pcm=pcm, modes=modes)



def checkpoint(out, name, **arrays):
    temporary = out / (name + ".tmp")
    with temporary.open("wb") as file:
        np.savez(file, **arrays)
    temporary.replace(out / (name + ".npz"))


def collect_special(device, out, frames):
    """Encoder-originated DTX/tone frames, then their actual decoder responses."""
    device.configure(17)
    labels, stimuli, channels, pcm, modes = [], [], [], [], []
    for frequency in (0, 350, 440, 1000, 2000, 3500):
        signal = (3000 * np.sin(2 * np.pi * frequency * np.arange(frames * 160) / 8000)).astype(np.int16)
        device.reset_encoder()
        device.control(b"\x05\x18\x00")  # Documented DTX_ENABLE + TD_ENABLE.
        encoded = [device.encode(row.astype(">i2").tobytes()) for row in signal.reshape(-1, 160)]
        device.reset_decoder()
        responses = [device.decode(row) for row in encoded]
        labels.append(frequency)
        stimuli.append(signal)
        channels.append(np.array([list(row) for row in encoded], dtype=np.uint8))
        pcm.append(np.frombuffer(b"".join(row[0] for row in responses), dtype=">i2"))
        modes.append([row[1] for row in responses])
        checkpoint(out, "special", frequency_hz=labels, stimulus_pcm=stimuli,
                   channel=channels, pcm=pcm, modes=modes)
        print("Collected special", frequency, flush=True)


def collect_research(device, out, frames, experiment, pitch, gain):
    base = reference(pitch, gain)
    device.configure(17)
    labels, inputs, pcm, modes = [], [], [], []
    for number, (label, bits, stream) in enumerate(research_cases(experiment, base, frames)):
        device.reset_decoder()
        responses = [device.decode(pack_raw(frame)) for frame in stream]
        labels.append(label)
        inputs.append(bits)
        pcm.append(np.frombuffer(b"".join(row[0] for row in responses), dtype=">i2"))
        modes.append([row[1] for row in responses])
        # Atomic checkpoints remain readable if a run is interrupted.
        checkpoint(out, "research", labels=labels, bits=inputs, base=base, pcm=pcm, modes=modes)
        if number % 16 == 0:
            print("Collected", experiment, *label, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("fec", "bits", "pitch", "hoc", "transitions", "special")
                        + RESEARCH_EXPERIMENTS)
    parser.add_argument("--device", required=True)
    parser.add_argument("--baud", type=int, default=460800)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--base-hex", default="dcac2beb853b6edd04bc",
                        help="10 rate-17 bytes; default is the 150 Hz harmonic reference")
    parser.add_argument("--frames", type=int, default=32, help="decoder repetitions per probe")
    parser.add_argument("--pitch-index", type=int, default=80, help="research reference pitch (0..123)")
    parser.add_argument("--gain-index", type=int, default=40, help="research reference gain (0..63)")
    args = parser.parse_args()
    if not 2 <= args.frames <= 200:
        parser.error("--frames must be between 2 and 200")
    if not 0 <= args.pitch_index <= 123 or not 0 <= args.gain_index <= 63:
        parser.error("reference pitch must be 0..123 and gain must be 0..63")
    args.out.mkdir(parents=True, exist_ok=False)
    with ThumbDV(args.device, args.baud) as device:
        identity = device.identity()
        print(identity, flush=True)
        status = "incomplete"
        try:
            if args.experiment in RESEARCH_EXPERIMENTS:
                collect_research(device, args.out, args.frames, args.experiment,
                                 args.pitch_index, args.gain_index)
            elif args.experiment == "special":
                collect_special(device, args.out, args.frames)
            elif args.experiment == "fec":
                collect_fec(device, args.out)
            elif args.experiment in ("hoc", "transitions"):
                collect_models(device, args.out, args.frames, args.experiment)
            else:
                collect_parameters(device, args.out, args.base_hex, args.experiment, args.frames)
            status = "complete"
        finally:
            metadata = {"status": status, "identity": identity, "experiment": args.experiment,
                        "candidate_fields": FIELDS,
                        "reference_pitch": args.pitch_index if args.experiment in RESEARCH_EXPERIMENTS else None,
                        "reference_gain": args.gain_index if args.experiment in RESEARCH_EXPERIMENTS else None,
                        "base_hex": args.base_hex if args.experiment in ("bits", "pitch") else None,
                        "frames_per_segment": args.frames if args.experiment != "fec" else None,
                        "baud": args.baud, "controls": device.controls}
            (args.out / "device.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()

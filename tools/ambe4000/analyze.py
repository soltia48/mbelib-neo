#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Quantify controlled AMBE+ responses; never exports decoder tables.

Requires NumPy. Frequency and harmonic count must be supplied from a separate
measurement. Results describe PCM responses, including hardware synthesis and
postfiltering; an estimated decay is not an exact internal predictor.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def harmonic_logs(pcm, frequency, harmonics, frame_size=160):
    """Fit a sinusoid pair per harmonic independently in each PCM window.

Reject underdetermined/ill-conditioned fits and the Nyquist boundary. Return
log2 amplitudes, the waveform fit residual, and the basis condition number.
The 1-LSB floor makes low-level measurements explicitly bounded, not exact.
"""
    pcm = np.asarray(pcm, dtype=float)
    if frame_size < 2 or pcm.ndim != 2 or pcm.shape[1] % frame_size:
        raise ValueError("PCM must contain whole analysis windows")
    if (not np.isfinite(pcm).all() or not np.isfinite(frequency) or frequency <= 0
            or harmonics < 1 or harmonics * frequency >= 4000
            or 2 * harmonics >= frame_size):
        raise ValueError("invalid frequency, harmonic count, or PCM")
    n = np.arange(frame_size)
    phase = 2 * np.pi * frequency / 8000 * n[:, None] * np.arange(1, harmonics + 1)
    basis = np.stack((np.cos(phase), np.sin(phase)), axis=2).reshape(frame_size, -1)
    condition = float(np.linalg.cond(basis))
    if condition > 100:
        raise ValueError("harmonic fit is ill-conditioned")
    windows = pcm.reshape(-1, frame_size).T
    coef = np.linalg.lstsq(basis, windows, rcond=None)[0]
    residual = np.sqrt(np.mean((windows - basis @ coef) ** 2, axis=0))
    amplitude = np.sqrt(np.sum(coef.reshape(harmonics, 2, -1) ** 2, axis=1)).T
    shape = (len(pcm), pcm.shape[1] // frame_size)
    return np.log2(np.maximum(amplitude, 1)).reshape(*shape, harmonics), residual.reshape(shape), condition


def impulse_report(data, frequency, harmonics):
    labels = [(str(name), int(code)) for name, code in data["labels"]]
    expected = [("control", 0)] + [("bit", i) for i in range(6, 79)] + [("control", 1)]
    if labels != expected:
        raise ValueError("expected a complete impulse run with both controls")
    pcm, modes = data["pcm"], data["modes"]
    if pcm.ndim != 2 or pcm.shape[0] != len(expected) or pcm.shape[1] % 320:
        raise ValueError("invalid impulse PCM shape")
    frames = pcm.shape[1] // 160
    if frames < 24:
        raise ValueError("use at least 12 warm-up frames for decay analysis")
    if modes.shape != (len(expected), frames) or np.any(modes & 0x20):
        raise ValueError("missing modes or hardware DATA_INVALID response")
    base, bits = data["base"], data["bits"]
    if base.shape != (80,) or bits.shape != (len(expected), 80):
        raise ValueError("invalid parameter dimensions")
    if np.any((bits != 0) & (bits != 1)) or np.any((base != 0) & (base != 1)):
        raise ValueError("nonbinary parameters")
    for row, (name, bit) in zip(bits, expected):
        reference = base.copy()
        if name == "bit":
            reference[bit] ^= 1
        if not np.array_equal(row, reference):
            raise ValueError("input parameters do not match the case label")
    logs, residual, condition = harmonic_logs(pcm, frequency, harmonics)
    delta = logs - logs[0]
    shape = delta - delta.mean(axis=2, keepdims=True)
    event = frames // 2
    # Exclude the immediate synthesis overlap from a late-tail AR(1) fit.
    start, end = event + 3, min(event + 11, frames - 1)
    noise_energy = float(np.sum(shape[-1, start:end] ** 2))
    cases = []
    for i, (name, bit) in enumerate(labels):
        before, after = shape[i, start:end], shape[i, start + 1:end + 1]
        energy = np.sum(before * before)
        decay = float(np.sum(before * after) / energy) if energy > max(1e-8, 10 * noise_energy) else None
        cases.append({"kind": name, "bit_or_control": bit,
                      "mean_log2_delta": delta[i, event:event + 12].mean(axis=1).tolist(),
                      "shape_rms_log2_delta": np.sqrt(np.mean(shape[i, event:event + 12] ** 2, axis=1)).tolist(),
                      "observed_tail_ar1": decay})
    return {"status": "measured_response_not_decoder_parameters", "frequency_hz": frequency,
            "harmonics": harmonics, "window_samples": 160, "amplitude_floor_lsb": 1,
            "basis_condition": condition, "cases": cases, "frames_per_case": frames,
            "impulse_frame": event, "pcm_peak": int(np.max(np.abs(pcm.astype(float)))),
            "pcm_rms": float(np.sqrt(np.mean(pcm.astype(float) ** 2))),
            "control_drift_rms_log2": float(np.sqrt(np.mean(delta[-1, event:] ** 2))),
            "median_waveform_fit_residual_lsb": float(np.median(residual)),
            "caution": "Decay includes enhancement, interpolation, phase motion and quantization; it is not an exact codec coefficient."}



def fit_decay(values):
    """Least-squares shared exponential decay, with one intercept per series.

Search resolution is 0.0001; the fitted range is 0.5..0.95. A boundary optimum
or a large residual contradicts this model rather than proving a predictor.
"""
    values = np.asarray(values, dtype=float)
    if values.ndim != 3 or values.shape[1] < 3 or not np.isfinite(values).all():
        raise ValueError("expected finite series x time x coefficients")
    energy = float(np.sum(values * values))
    if energy < 1e-12:
        raise ValueError("no measurable decay signal")
    rates = np.linspace(.5, .95, 4501)
    decay = rates[:, None] ** np.arange(values.shape[1])
    projection = np.einsum("ctk,rt->rck", values, decay)
    score = np.sum(projection ** 2, axis=(1, 2)) / np.sum(decay ** 2, axis=1)
    best = int(np.argmax(score))
    return {"observed_decay": float(rates[best]),
            "relative_fit_error": float(np.sqrt(max(0, 1 - score[best] / energy))),
            "search_boundary": best in (0, len(rates) - 1), "search_resolution": .0001}


def hoc_report(data, frequency, harmonics, blocks):
    from fields import set_field
    expected = [("control", 0)] + [("hoc" + str(b), i) for b in range(4) for i in range(32)] + [("control", 1)]
    labels = [(str(name), int(code)) for name, code in data["labels"]]
    if labels != expected:
        raise ValueError("expected a complete hoc-impulse run with both controls")
    if len(blocks) != 4 or min(blocks) < 6 or sum(blocks) != harmonics:
        raise ValueError("supply four measured block lengths >= 6 summing to the harmonic count")
    pcm, modes = data["pcm"], data["modes"]
    if pcm.ndim != 2 or pcm.shape[0] != 130 or pcm.shape[1] % 320 or pcm.shape[1] < 3840:
        raise ValueError("expected at least 12 warm-up frames, an impulse and recovery")
    frames = pcm.shape[1] // 160
    if modes.shape != (130, frames) or np.any(modes & 0x20):
        raise ValueError("missing modes or hardware DATA_INVALID response")
    if data["base"].shape != (80,) or data["bits"].shape != (130, 80):
        raise ValueError("invalid parameter dimensions")
    for row, (name, code) in zip(data["bits"], expected):
        reference = data["base"].tolist()
        if name != "control":
            set_field(reference, name, code)
        if not np.array_equal(row, reference):
            raise ValueError("input parameters do not match the case label")
    logs, residual, condition = harmonic_logs(pcm, frequency, harmonics)
    event = frames // 2
    delta = logs - logs[0]
    results, start = [], 0
    for b, length in enumerate(blocks):
        # DCT-II, normalized by block length, coefficients k=2..5.
        basis = np.cos(np.pi / length * (np.arange(length)[:, None] + .5) * np.arange(2, 6)) / length
        response = delta[1 + 32 * b:1 + 32 * (b + 1), event + 2:event + 10, start:start + length] @ basis
        result = fit_decay(response)
        result.update(block=b, harmonics=length, codes=32)
        results.append(result)
        start += length
    return {"status": "measured_decay_not_decoder_codebooks", "frequency_hz": frequency,
            "harmonics": harmonics, "basis_condition": condition, "frames_per_case": frames,
            "fit_frames_after_impulse": list(range(2, 10)), "blocks": results,
            "pcm_peak": int(np.max(np.abs(pcm.astype(float)))),
            "control_drift_rms_log2": float(np.sqrt(np.mean(delta[-1, event:] ** 2))),
            "caution": "The fitted exponent includes hardware synthesis; absolute HOC quantizers remain unresolved."}


def voicing_report(data, frequency, harmonics):
    from fields import set_field
    labels = [(str(name), int(code)) for name, code in data["labels"]]
    expected = [("control", 0)] + [("voicing", i) for i in range(64)] + [("control", 1)]
    pcm, modes = data["pcm"], data["modes"]
    if labels != expected or pcm.ndim != 2 or pcm.shape[0] != 66:
        raise ValueError("expected a complete 64-pattern voicing run with controls")
    if pcm.shape[1] < 2560 or pcm.shape[1] % 160 or modes.shape != (66, pcm.shape[1] // 160):
        raise ValueError("voicing analysis needs at least 16 complete frames")
    if np.any(modes & 0x20) or not np.isfinite(pcm).all():
        raise ValueError("invalid PCM or hardware DATA_INVALID response")
    if not np.isfinite(frequency) or frequency < 50 or harmonics < 1 or frequency * harmonics >= 4000:
        raise ValueError("invalid frequency or harmonic count")
    if data["base"].shape != (80,) or data["bits"].shape != (66, 80):
        raise ValueError("invalid parameter dimensions")
    for row, (name, code) in zip(data["bits"], expected):
        reference = data["base"].tolist()
        if name == "voicing":
            set_field(reference, name, code)
        if not np.array_equal(row, reference):
            raise ValueError("input parameters do not match the case label")
    # Welch power, periodic Hann window, 80 ms windows, 50% overlap.
    signal = pcm[:, -2560:].astype(float)
    window = np.hanning(641)[:-1]
    spectra = []
    for start in range(0, 1921, 320):
        segment = signal[:, start:start + 640]
        segment = (segment - segment.mean(axis=1, keepdims=True)) * window
        spectra.append(abs(np.fft.rfft(segment, n=8192, axis=1)) ** 2)
    power = np.mean(spectra, axis=0)
    grid = np.fft.rfftfreq(8192, 1 / 8000)
    ratio = []
    for harmonic in range(1, harmonics + 1):
        distance = abs(grid - harmonic * frequency)
        denominator = power[:, distance < .45 * frequency].sum(axis=1)
        if np.any(denominator <= 0):
            raise ValueError("zero-power voicing measurement")
        ratio.append(power[:, distance < 12].sum(axis=1) / denominator)
    ratio = np.array(ratio).T
    observed = ratio > .6
    masks = observed[[33, 17, 9, 5, 3, 2]]  # only six single-bit cases train the model
    predicted = np.array([np.any(masks[[bool(code & (1 << (5 - j))) for j in range(6)]], axis=0)
                          for code in range(64)])
    mismatch = np.argwhere(predicted != observed[1:65])
    return {"status": "measured_voicing_masks_not_normative_band_edges", "frequency_hz": frequency,
            "harmonics": harmonics, "window_samples": 640, "analysis_samples": 2560,
            "line_halfwidth_hz": 12, "band_halfwidth_hz": .45 * frequency, "threshold": .6,
            "single_bit_masks": {str(50 + i): (np.flatnonzero(mask) + 1).tolist()
                                 for i, mask in enumerate(masks)},
            "compared_decisions": 64 * harmonics,
            "mismatches": [{"code": int(c), "harmonic": int(h + 1)} for c, h in mismatch],
            "minimum_predicted_voiced_ratio": float(ratio[1:65][predicted].min()) if predicted.any() else None,
            "maximum_predicted_unvoiced_ratio": float(ratio[1:65][~predicted].max()) if (~predicted).any() else None,
            "control_max_ratio_drift": float(abs(ratio[0] - ratio[-1]).max()),
            "caution": "Masks apply to this measured pitch. Do not infer exact band edges from one harmonic grid."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="research.npz from probe.py")
    parser.add_argument("--experiment", choices=("impulse", "voicing", "hoc-impulse"), default="impulse")
    parser.add_argument("--fundamental-hz", type=float, required=True)
    parser.add_argument("--harmonics", type=int, required=True)
    parser.add_argument("--block-lengths", type=int, nargs=4, help="measured HOC block lengths")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = args.input.with_name("device.json")
    if not manifest.is_file():
        parser.error("the acquisition device.json manifest is required")
    metadata = json.loads(manifest.read_text())
    if metadata.get("status") != "complete" or metadata.get("experiment") != args.experiment:
        parser.error("acquisition is incomplete or its experiment does not match")
    with np.load(args.input, allow_pickle=False) as data:
        if args.experiment == "hoc-impulse":
            if args.block_lengths is None:
                parser.error("hoc-impulse requires --block-lengths")
            report = hoc_report(data, args.fundamental_hz, args.harmonics, args.block_lengths)
        else:
            analyze = impulse_report if args.experiment == "impulse" else voicing_report
            report = analyze(data, args.fundamental_hz, args.harmonics)
    report["input_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    with args.out.open("x") as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write("\n")
    print("Wrote", args.out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Generate approximate model v1 from saved numerical input arrays.

This is a fixed model-generation recipe.
Requires NumPy/SciPy; never opens hardware. Absolute HOC offsets, subframe
normalization and interpolation remain approximations. See the model notes.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.fft import dct, idct

from analyze import harmonic_logs
from fields import indices

HEADER = """// SPDX-License-Identifier: GPL-2.0-or-later
/** @file Approximate AMBE+ 4000 model, version 1.
 * Absolute HOC offsets, subframe tables and pitch values are approximate.
 * Model choices and limitations are documented in docs/ambe4000-research.md.
 * PRBA reconstruction reuses the existing code.
 */
#ifndef MBELIB_AMBE4000_MODEL_H
#define MBELIB_AMBE4000_MODEL_H
"""


def c_array(value, integer=False):
    if value.ndim == 0:
        return str(int(value)) if integer else f"{float(value):.8f}f"
    return "{" + ", ".join(c_array(row, integer) for row in value) + "}"


def write_header(path, model):
    with path.open("x") as output:
        output.write(HEADER)
        for key in ("hoc", "subgain", "shape123", "shape456", "shape789", "pitch", "gain", "blocks"):
            values = np.asarray(model[key])
            dimensions = "".join(f"[{n}]" for n in values.shape)
            kind = "int" if key == "blocks" else "float"
            output.write(f"static const {kind} ambe4000_{key}{dimensions} = {c_array(values, key == 'blocks')};\n")
        output.write("\n#endif\n")


def derive(root):
    paths = [root / name for name in ("tables.npz", "pitch_hz.npy", "phase2/reproduced-impulse/research.npz",
                                      "phase2/reproduced-hoc/research.npz", "phase2/extra.npz")]
    tables = dict(np.load(paths[0], allow_pickle=False))
    pitch = np.load(paths[1], allow_pickle=False)[:124]
    frequency, harmonics = float(pitch[80]), 30
    blocks = tables["AmbePlusLmprbl"][harmonics]
    if blocks.tolist() != [6, 7, 8, 9] or pitch.shape != (124,) or not np.isfinite(pitch).all():
        raise ValueError("model v1 requires the archived pitch-80 experiment")

    def coarse_shape(bits):
        b = indices(bits)
        g = np.r_[0, tables["AmbePlusPRBA24"][b["prba24"]], tables["AmbePlusPRBA58"][b["prba58"]]]
        r = idct(g, type=2, norm="forward")
        result = []
        for k, count in enumerate(blocks):
            coefficients = np.zeros(count)
            coefficients[0] = (r[2*k] + r[2*k+1]) / 2
            coefficients[1] = (r[2*k] - r[2*k+1]) / (2*np.sqrt(2))
            result.extend(idct(coefficients, type=2, norm="forward"))
        result = np.asarray(result)
        return result - result.mean()

    def acquisition(path, shape):
        data = dict(np.load(path, allow_pickle=False))
        if data["pcm"].shape != shape or np.any(data["modes"] & 32):
            raise ValueError(f"incomplete or invalid model-v1 acquisition: {path}")
        if not np.isfinite(data["pcm"]).all():
            raise ValueError("nonfinite PCM")
        return data

    # Late impulse responses isolate the main-frame predictor from the extra
    # subframe. Calibrate PCM log amplitudes against the existing coarse PRBA.
    data = acquisition(paths[2], (75, 64*160))
    expected_labels = [("control", 0)] + [("bit", i) for i in range(6, 79)] + [("control", 1)]
    if [(str(a), int(b)) for a, b in data["labels"]] != expected_labels:
        raise ValueError("incomplete impulse labels")
    logs = harmonic_logs(data["pcm"], frequency, harmonics)[0]
    weights = .8 ** np.arange(8)
    observed = ((logs - logs[0])[:, 34:42] * weights[None, :, None]).sum(1) / (weights @ weights)
    calibration = []
    for i, (name, bit) in enumerate(data["labels"]):
        if name != "bit" or int(bit) not in list(range(10, 22)) + [58, 59, 60, 61]:
            continue
        expected = coarse_shape(data["bits"][i]) - coarse_shape(data["base"])
        y = observed[i] - observed[i].mean()
        calibration.append(float(y @ expected / (expected @ expected)))
    scale = float(np.median(calibration))
    if not .5 < scale < 1.5:
        raise ValueError("unexpected amplitude calibration")

    data = acquisition(paths[3], (130, 48*160))
    expected_labels = [("control", 0)] + [(f"hoc{k}", i) for k in range(4) for i in range(32)] + [("control", 1)]
    if [(str(a), int(b)) for a, b in data["labels"]] != expected_labels:
        raise ValueError("incomplete HOC labels")
    logs = harmonic_logs(data["pcm"], frequency, harmonics)[0]
    observed = ((logs - logs[0])[:, 26:34] * weights[None, :, None]).sum(1) / (weights @ weights) / scale
    hoc, start = [], 0
    for k, count in enumerate(blocks):
        # A deliberately approximate absolute offset from the steady control.
        baseline = dct(logs[0, -10:].mean(0)[start:start+count], type=2, norm="forward")[2:6] * .2
        coefficients = dct(observed[1+k*32:1+(k+1)*32, start:start+count], type=2, norm="forward")[:, 2:6]
        hoc.append(coefficients + baseline)
        start += count

    data = acquisition(paths[4], (272, 32*160))
    expected_labels = [(k, i) for k, count in enumerate((16, 128, 64, 64)) for i in range(count)]
    if data["labels"].tolist() != [list(label) for label in expected_labels]:
        raise ValueError("model v1 expects the archived 272-case extra sweep")
    logs = harmonic_logs(data["pcm"][:, -2560:], frequency, harmonics)[0].mean(1)
    coefficients = dct(logs, type=2, norm="forward")
    shapes = []
    for block, selected in ((1, slice(1, 4)), (2, slice(4, 7)), (3, slice(7, 10))):
        rows = np.flatnonzero(data["labels"][:, 0] == block)
        values = (coefficients[rows, selected] - coefficients[9, selected]) * 2
        shapes.append(values - values.mean(axis=0))  # Unknown offsets: assume zero codebook means.
    subgain = (coefficients[:16, 0] - coefficients[9, 0]) * 2
    subgain -= np.median(subgain[7:9])
    model = dict(hoc=np.asarray(hoc), subgain=subgain, shape123=shapes[0], shape456=shapes[1],
                 shape789=shapes[2], pitch=pitch, gain=tables["AmbePlusDg"]*2,
                 prba24=tables["AmbePlusPRBA24"], prba58=tables["AmbePlusPRBA58"], blocks=tables["AmbePlusLmprbl"])
    if any(not np.isfinite(value).all() for value in model.values()):
        raise ValueError("nonfinite estimated table")
    report = {"model_version": 1, "approximate": True, "prba_calibration": calibration,
              "median_calibration": scale,
              "input_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    return model, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path, help="archived build/ambe4000 directory")
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    args = parser.parse_args()
    model, report = derive(args.artifacts)
    args.out.mkdir(parents=True, exist_ok=False)
    np.savez(args.out / "model.npz", **model)
    write_header(args.out / "ambe4000_model.h", model)
    (args.out / "provenance.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Wrote approximate model v1; calibration = {report['median_calibration']:.8f}")


if __name__ == "__main__":
    main()

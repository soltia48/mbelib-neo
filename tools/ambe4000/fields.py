# SPDX-License-Identifier: GPL-2.0-or-later
"""Candidate AMBE+ 4000 field layout for experiments, NOT a speech decoder.

Positions are zero-based, MSB first, in deinterleaved priority order. Field
names identify parameter groups; they do not establish exact quantizer values.
"""
FIELDS = {
    "pitch": tuple(range(6)) + (79,),
    "gain": tuple(range(6, 10)) + (56, 57),
    "prba24": tuple(range(10, 17)) + (58, 59),
    "prba58": tuple(range(17, 22)) + (60, 61),
    "hoc0": tuple(range(22, 26)) + (62,),
    "hoc1": tuple(range(26, 30)) + (63,),
    "hoc2": tuple(range(30, 34)) + (64,),
    "hoc3": tuple(range(34, 38)) + (65,),
    "interpolation": (38, 39),
    "subgain": (40, 41, 42, 66),
    "shape123": (43, 44, 45, 67, 68, 69, 70),
    "shape456": (46, 47, 71, 72, 73, 74),
    "shape789": (48, 49, 75, 76, 77, 78),
    "voicing": tuple(range(50, 56)),
}


def set_field(bits, name, value):
    """Change one candidate index; reject out-of-range values before mutation."""
    positions = FIELDS[name]
    if len(bits) != 80 or any(bit not in (0, 1) for bit in bits):
        raise ValueError("expected 80 binary parameters")
    if not isinstance(value, int) or not 0 <= value < 1 << len(positions):
        raise ValueError("index does not fit field " + name)
    for shift, index in enumerate(reversed(positions)):
        bits[index] = (value >> shift) & 1


def indices(bits):
    """Return candidate field indices, without interpreting their values."""
    if len(bits) != 80 or any(bit not in (0, 1) for bit in bits):
        raise ValueError("expected 80 binary parameters")
    return {name: sum(int(bits[index]) << shift
                      for shift, index in enumerate(reversed(positions)))
            for name, positions in FIELDS.items()}


def reference(pitch=80, gain=40):
    """Moderate-level reference vector for parameter tests.

This artificial all-voiced frame is not guaranteed to remain unsaturated when
arbitrary fields are changed. Measure output range for every experiment.
"""
    from thumbdv import unpack_raw
    bits = unpack_raw(bytes.fromhex("dcac2beb853b6edd04bc"))
    for name, value in zip(("pitch", "gain", "prba24", "prba58", "hoc0",
                            "hoc1", "hoc2", "hoc3", "voicing"),
                           (pitch, gain, 169, 41, 18, 15, 16, 21, 63)):
        set_field(bits, name, value)
    return bits


def research_cases(experiment, base, frames):
    """Yield (label, changed vector, full input stream); first case is a control.

An impulse follows `frames` reference frames and is followed by `frames-1`
reference frames. A repeated control at the end measures reset/repeat drift.
"""
    if frames < 2:
        raise ValueError("at least two frames are required")
    if experiment == "impulse":
        changes = [("bit", i) for i in range(6, 79)]
    else:
        names = {"extra": ("subgain", "shape123", "shape456", "shape789"),
                 "voicing": ("voicing",), "gain": ("gain",),
                 "hoc-impulse": ("hoc0", "hoc1", "hoc2", "hoc3")}[experiment]
        changes = [(name, code) for name in names for code in range(1 << len(FIELDS[name]))]
    for name, code in [("control", 0)] + changes + [("control", 1)]:
        bits = list(base)
        if name == "bit":
            bits[code] ^= 1
        elif name != "control":
            set_field(bits, name, code)
        stream = ([base] * frames + [bits] + [base] * (frames - 1)
                  if experiment in ("impulse", "hoc-impulse") else [bits] * frames)
        yield (name, code), bits, stream

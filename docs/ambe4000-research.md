# AMBE+ 4000 model and validation notes

The native C decoder produces 8 kHz PCM from AMBE+ 4000 speech frames using an
experimental, approximate model. The equations below describe this
implementation, not a normative codec specification. See [the API and
command-line usage](ambe4000.md).

## Parameter layout

Bit positions are zero-based in priority order. Multi-bit indices are listed
most-significant first.

| Field | Priority bits | Model use |
| --- | --- | --- |
| Fundamental index | 0–5, 79 | Ordinary voice codes 0–123; special codes 124–127 |
| Main gain index | 6–9, 56–57 | 64-entry gain table |
| Coarse spectral vector 1 | 10–16, 58–59 | 512-entry PRBA24 table |
| Coarse spectral vector 2 | 17–21, 60–61 | 128-entry PRBA58 table |
| Higher-order block 1 | 22–25, 62 | 32-entry HOC table |
| Higher-order block 2 | 26–29, 63 | 32-entry HOC table |
| Higher-order block 3 | 30–33, 64 | 32-entry HOC table |
| Higher-order block 4 | 34–37, 65 | 32-entry HOC table |
| Spectral interpolation | 38–39 | Four blend coefficients |
| Intermediate gain | 40–42, 66 | 16-entry subgain table |
| Shape components 1–3 | 43–45, 67–70 | 128-entry, three-dimensional table |
| Shape components 4–6 | 46–47, 71–74 | 64-entry, three-dimensional table |
| Shape components 7–9 | 48–49, 75–78 | 64-entry, three-dimensional table |
| Voicing controls | 50–55 | Six overlapping frequency masks |

## Software model v1

`src/ambe/ambe4000_decode.c` uses all 80 speech bits. Its equations and model
choices are:

- Fundamental frequency: 124 table entries for ordinary voice. Harmonic count
  is `floor(3800/f0)`, limited to 9–56; block lengths use the existing AMBE+ table.
- Coarse spectrum: existing PRBA24/58 reconstruction. Four 32×4 HOC tables
  supply block DCT coefficients 2–5.
- Main log2 spectral shape: interpolate the prior shape onto the new harmonic
  grid, remove its mean, multiply by 0.8 and add the coarse/HOC residual. Remove
  the resulting mean again; gain is handled separately.
- Main log2 amplitude: `2 * AmbePlusDg[index] + 1 - 0.5*log2(L)`.
  There is no main gain predictor.
- Intermediate subframe: blend prior/current shape with coefficients
  `[0.1, 0.3, 0.5, 0.9]`, add three global DCT corrections, and use the average
  old/new gain plus a 16-entry subgain correction. Interpolation and
  normalization remain approximate.
- Synthesis: two 80-sample steps per 20 ms frame, a minimum-phase approximation
  for voiced harmonics, inverse-FFT unvoiced bands, linear voiced overlap and
  equal-power noise overlap. Noise has a deterministic per-context xorshift32
  sequence. PCM is limited to ±31128; float samples use mbelib's PCM/7 scale.

The six voicing bits select overlapping frequency intervals. A harmonic is
voiced if at least one selected interval contains its frequency.

| Priority bit | Frequency interval in Hz |
| --- | --- |
| 50 | [0, 500) |
| 51 | [250, 1000) |
| 52 | [750, 1500) |
| 53 | [1250, 2000) |
| 54 | [1750, 3000) |
| 55 | [2750, 4000) |

Code 124 produces muted DTX (`SILENCE|MUTE`). Code 126 supports single-tone IDs
5–122 at `31.25 * ID` Hz, DTMF IDs 128–143 and call-progress IDs 144–146.
Tone amplitude is approximate. Codes 125/127 and unsupported tone IDs are
muted with `ERASURE`; `NOTONES=ON` also mutes recognized tones. Special frames
clear synthesis overlap but retain the previous voice model.

## Model tables

The ordinary build uses only `src/internal/ambe4000_model.h` and the existing
PRBA reconstruction code. No numerical Python packages or external model
files are needed at runtime.

`tools/ambe4000/estimate_tables.py` can regenerate the header from saved
numerical input arrays. It uses sinusoidal amplitude fits, DCT coefficients,
0.8 decay weights, and a median calibration factor of 0.9145158227. Absolute
HOC offsets and subframe normalization remain approximations. The generator
checks input dimensions, labels and finite values before writing its output.
[The model metadata](ambe4000-model-v1.json) records its calibration values and
input checksums.

## Independent native speech validation

Four public files were taken from Codec2 revision
[`310777b1c6f1af0bc7c72f5b32f80f6fd9136962`](https://github.com/drowe67/codec2/tree/310777b1c6f1af0bc7c72f5b32f80f6fd9136962/raw):
`hts1`, `hts2a`, `kristoff`, and `ve9qrp_10s`. The source PCM is 8 kHz mono,
signed 16-bit little endian. Native decoding reads only saved `.ambe` frames;
source PCM and ASR are used only for evaluation.

| Clip | Frames including flush | Native STOI |
| --- | ---: | ---: |
| hts1 | 303 | 0.8851 |
| hts2a | 153 | 0.8561 |
| kristoff | 253 | 0.7979 |
| ve9qrp_10s | 503 | 0.8625 |

Scores use non-extended STOI against the original PCM, with a fixed 240-sample
(30 ms) output delay and the original source length. That delay was selected on
`hts1` and retained for every other file. `hts1` was also used for global level
adjustment; the remaining clips were not used to fit the model tables or choose
parameters. These four clips are limited quality checks, not population-level
voice-quality measurements.

A local `tiny.en` faster-whisper model (CPU int8, beam size 5, English,
`condition_on_previous_text=False`) returned the same two sentences for native
and source `hts1`, apart from letter case. Other clips had recognition errors,
including a word substitution in `hts2a` and call-sign/name errors. ASR was used
only to assess independently produced PCM, never to synthesize or repair it.
No human listening score or sample-exact equivalence is claimed.

`tools/ambe4000/evaluate_speech.py` runs the native CLI on saved streams, checks
source hashes and WAV/source agreement, computes STOI, and optionally uses a
cached ASR model. It requires NumPy/SciPy/pystoi; optional ASR also needs
faster-whisper. It never opens a serial port or downloads a model. Corpus
files, audio, virtual environments and model weights are not shipped with the
library.

## Limitations

Exact codebook offsets and quantization, pitch and voicing boundary rules,
spectral interpolation and enhancement, phase regeneration, complete special-frame
semantics, comfort noise and state restoration after tones/DTX remain unresolved.
The decoder conservatively mutes DTX and unsupported special frames.
The 1000 Hz tone fixture checks frequency and RMS within 5%, not all combinations.

Frame errors in the protected part use Viterbi correction, while 44 speech bits
have no protection or error detection. Tests verify correction through PCM and
bounded output for arbitrary frames; no general packet-loss concealment or
uncorrectable-frame detector is claimed. Native PCM-to-AMBE+ 4000 encoding is
not implemented.

All 24 CTest cases passed in Debug, ASan/UBSan and NOTONES builds. The Clang frame
fuzz target completed 5,000 inputs under ASan/UBSan. Formatting checks and
byte-for-byte table regeneration passed.

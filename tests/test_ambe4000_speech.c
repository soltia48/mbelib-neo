// SPDX-License-Identifier: GPL-2.0-or-later
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include "data/ambe4000_vectors.h"
#include "mbelib-neo/mbelib.h"

static void
unpack(const unsigned char* bytes, char* bits, int count) {
    for (int i = 0; i < count; ++i) {
        bits[i] = (char)((bytes[i / 8] >> (7 - i % 8)) & 1);
    }
}

static void
check_pcm(const float* pcm) {
    for (int i = 0; i < 160; ++i) {
        assert(isfinite(pcm[i]) && fabsf(pcm[i]) <= 31128.0f / 7.0f);
    }
}

static void
paths_and_reset(void) {
    mbe_ambe4000_decoder* a = mbe_ambe4000DecoderAlloc();
    mbe_ambe4000_decoder* b = mbe_ambe4000DecoderAlloc();
    mbe_ambe4000_decoder* c = mbe_ambe4000DecoderAlloc();
    mbe_ambe4000_decoder* d = mbe_ambe4000DecoderAlloc();
    assert(a && b && c && d);
    float saved[27][160];
    double energy = 0;
    for (int pass = 0; pass < 2; ++pass) {
        mbe_ambe4000DecoderReset(a);
        mbe_ambe4000DecoderReset(b);
        mbe_ambe4000DecoderReset(c);
        mbe_ambe4000DecoderReset(d);
        for (int v = 0; v < 27; ++v) {
            char bits[80], frame[128], decoded[80];
            mbe_soft_bit soft[128];
            float x[160], y[160], z[160];
            short pcm[160], expected[160];
            mbe_process_result result;
            unpack(ambe4000_vectors[v].data, bits, 80);
            unpack(ambe4000_vectors[v].channel, frame, 128);
            for (int i = 0; i < 128; ++i) {
                soft[i].bit = (unsigned char)frame[i];
                soft[i].reliability = 255;
            }
            assert(mbe_processAmbe4000Dataf(x, &result, bits, a) == 0);
            assert(result.flags == 0 && result.total_errors == 0);
            /* A protected bit flip must not affect synthesized PCM. */
            frame[0] ^= 1;
            assert(mbe_processAmbe6400x4000Framef(y, &result, frame, decoded, b) == 1);
            assert(result.total_errors == 1 && result.protected_errors == 1);
            assert(memcmp(bits, decoded, 80) == 0);
            assert(mbe_processAmbe6400x4000SoftFramef(z, &result, soft, NULL, c) == 0);
            assert(result.flags == MBE_PROCESS_FLAG_SOFT_INPUT);
            assert(memcmp(x, y, sizeof(x)) == 0 && memcmp(x, z, sizeof(x)) == 0);
            check_pcm(x);
            mbe_floattoshort(x, expected);
            int ret;
            if (v % 3 == 0) {
                ret = mbe_processAmbe4000Data(pcm, NULL, bits, d);
            } else if (v % 3 == 1) {
                frame[0] ^= 1;
                ret = mbe_processAmbe6400x4000Frame(pcm, NULL, frame, NULL, d);
            } else {
                ret = mbe_processAmbe6400x4000SoftFrame(pcm, NULL, soft, NULL, d);
            }
            assert(ret == 0 && memcmp(expected, pcm, sizeof(pcm)) == 0);
            if (pass == 0) {
                memcpy(saved[v], x, sizeof(x));
                for (int i = 0; i < 160; ++i) {
                    energy += x[i] * x[i];
                }
            } else {
                assert(memcmp(saved[v], x, sizeof(x)) == 0);
            }
        }
    }
    assert(energy > 1000);
    mbe_ambe4000DecoderFree(a);
    mbe_ambe4000DecoderFree(b);
    mbe_ambe4000DecoderFree(c);
    mbe_ambe4000DecoderFree(d);
}

static void
invalid_input(void) {
    mbe_ambe4000_decoder* a = mbe_ambe4000DecoderAlloc();
    mbe_ambe4000_decoder* b = mbe_ambe4000DecoderAlloc();
    assert(a && b);
    char bits[80], frame[128] = {0}, decoded[80];
    mbe_soft_bit soft[128] = {{0, 0}};
    float x[160], y[160];
    short pcm[160];
    unpack(ambe4000_vectors[8].data, bits, 80);
    assert(mbe_processAmbe4000Dataf(x, NULL, bits, a) == 0);
    assert(mbe_processAmbe4000Dataf(y, NULL, bits, b) == 0);
    for (int i = 0; i < 160; ++i) {
        x[i] = 42;
        pcm[i] = 42;
    }
    memset(decoded, 42, sizeof(decoded));
    bits[79] = 2;
    frame[127] = -1;
    soft[127].bit = 2;
    assert(mbe_processAmbe4000Dataf(x, NULL, bits, a) == MBE_STATUS_INVALID_BITS);
    assert(mbe_processAmbe4000Data(pcm, NULL, bits, a) == MBE_STATUS_INVALID_BITS);
    assert(mbe_processAmbe6400x4000Framef(x, NULL, frame, decoded, a) == MBE_STATUS_INVALID_BITS);
    assert(mbe_processAmbe6400x4000SoftFramef(x, NULL, soft, decoded, a) == MBE_STATUS_INVALID_BITS);
    assert(mbe_processAmbe6400x4000Frame(pcm, NULL, frame, decoded, a) == MBE_STATUS_INVALID_BITS);
    assert(mbe_processAmbe6400x4000SoftFrame(pcm, NULL, soft, decoded, a) == MBE_STATUS_INVALID_BITS);
    assert(mbe_processAmbe4000Dataf(x, NULL, NULL, a) == MBE_STATUS_INVALID_ARGUMENT);
    assert(mbe_processAmbe4000Dataf(x, NULL, bits, NULL) == MBE_STATUS_INVALID_ARGUMENT);
    assert(mbe_processAmbe4000Data(NULL, NULL, bits, a) == MBE_STATUS_INVALID_ARGUMENT);
    assert(mbe_processAmbe6400x4000Framef(x, NULL, NULL, decoded, a) == MBE_STATUS_INVALID_ARGUMENT);
    assert(mbe_processAmbe6400x4000SoftFramef(x, NULL, NULL, decoded, a) == MBE_STATUS_INVALID_ARGUMENT);
    for (int i = 0; i < 160; ++i) {
        assert(x[i] == 42 && pcm[i] == 42);
    }
    for (int i = 0; i < 80; ++i) {
        assert(decoded[i] == 42);
    }
    bits[79] = 0;
    assert(mbe_processAmbe4000Dataf(x, NULL, bits, a) == 0);
    assert(mbe_processAmbe4000Dataf(y, NULL, bits, b) == 0);
    assert(memcmp(x, y, sizeof(x)) == 0); /* Invalid calls preserve history. */
    mbe_ambe4000DecoderFree(a);
    mbe_ambe4000DecoderFree(b);
    mbe_ambe4000DecoderFree(NULL);
    mbe_ambe4000DecoderReset(NULL);
}

static void
tone_and_silence(void) {
    /* 1000 Hz single-tone fixture; reference output RMS = 2213.98. */
    const unsigned char tone[10] = {0xfd, 0x0b, 0x40, 0, 0, 0, 0, 0x10, 0, 0};
    mbe_ambe4000_decoder* dec = mbe_ambe4000DecoderAlloc();
    assert(dec);
    char bits[80];
    float out[160];
    mbe_process_result result;
    unpack(tone, bits, 80);
    assert(mbe_processAmbe4000Dataf(out, &result, bits, dec) == 0);
#ifdef DISABLE_AMBE_TONES
    assert(result.flags == (MBE_PROCESS_FLAG_TONE | MBE_PROCESS_FLAG_MUTE));
    for (int i = 0; i < 160; ++i) {
        assert(out[i] == 0);
    }
#else
    assert(result.flags == MBE_PROCESS_FLAG_TONE);
    double energy = 0, real = 0, imag = 0;
    for (int i = 0; i < 160; ++i) {
        double pcm = 7.0 * out[i], phase = 2 * 3.14159265358979323846 * 1000 * i / 8000;
        energy += pcm * pcm;
        real += pcm * cos(phase);
        imag += pcm * sin(phase);
    }
    assert(fabs(sqrt(energy / 160) / 2213.98 - 1) < .05);
    assert(2 * (real * real + imag * imag) / (160 * energy) > .999);
#endif
    /* b0 = 124 denotes DTX/silence; comfort noise is deliberately muted. */
    bits[5] = 0;
    assert(mbe_processAmbe4000Dataf(out, &result, bits, dec) == 0);
    assert(result.flags == (MBE_PROCESS_FLAG_SILENCE | MBE_PROCESS_FLAG_MUTE));
    for (int i = 0; i < 160; ++i) {
        assert(out[i] == 0);
    }
    bits[79] = 1; /* b0 = 125 is reserved. */
    assert(mbe_processAmbe4000Dataf(out, &result, bits, dec) == 0);
    assert(result.flags == (MBE_PROCESS_FLAG_ERASURE | MBE_PROCESS_FLAG_MUTE));
    mbe_ambe4000DecoderFree(dec);
}

static void
arbitrary_frames(void) {
    mbe_ambe4000_decoder* dec = mbe_ambe4000DecoderAlloc();
    assert(dec);
    uint32_t rng = 42;
    for (unsigned frame = 0; frame < 1024; ++frame) {
        char bits[80];
        float out[160];
        for (int i = 0; i < 80; ++i) {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            bits[i] = (char)(rng & 1);
        }
        unsigned pitch = frame % 128;
        for (int i = 0; i < 6; ++i) {
            bits[i] = (char)((pitch >> (6 - i)) & 1);
        }
        bits[79] = (char)(pitch & 1);
        assert(mbe_processAmbe4000Dataf(out, NULL, bits, dec) == 0);
        check_pcm(out);
    }
    mbe_ambe4000DecoderFree(dec);
}

int
main(void) {
    paths_and_reset();
    invalid_input();
    tone_and_silence();
    arbitrary_frames();
    return 0;
}

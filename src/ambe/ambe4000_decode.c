// SPDX-License-Identifier: GPL-2.0-or-later
/** @file Experimental AMBE+ 4000 software speech decoder.
 * Two 80-sample synthesis steps per frame. See docs/ambe4000-research.md for
 * approximate model choices, validation and limitations.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "ambe3600x2400_internal.h"
#include "ambe4000_model.h"
#include "mbe_result.h"
#include "mbe_unvoiced_fft.h"
#include "mbelib-neo/mbelib.h"

#define PI4000    3.14159265358979323846f
#define LIMIT4000 (31128.0f / 7.0f)

struct mbe_ambe4000_decoder {
    mbe_fft_plan* fft;
    int L;
    float shape[56];
    float frequency, gain, pulse;
    float voiced_tail[80], noise_tail[80];
    float tone_phase[2];
    uint32_t rng;
};

static float
bound(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

void
mbe_ambe4000DecoderReset(mbe_ambe4000_decoder* dec) {
    if (!dec) {
        return;
    }
    mbe_fft_plan* fft = dec->fft;
    memset(dec, 0, sizeof(*dec));
    dec->fft = fft;
    dec->L = 30;
    dec->frequency = 125.0f;
    dec->rng = 0x4000u;
}

mbe_ambe4000_decoder*
mbe_ambe4000DecoderAlloc(void) {
    mbe_ambe4000_decoder* dec = (mbe_ambe4000_decoder*)calloc(1, sizeof(*dec));
    if (!dec) {
        return NULL;
    }
    dec->fft = mbe_fft_plan_alloc();
    if (!dec->fft) {
        free(dec);
        return NULL;
    }
    mbe_ambe4000DecoderReset(dec);
    return dec;
}

void
mbe_ambe4000DecoderFree(mbe_ambe4000_decoder* dec) {
    if (dec) {
        mbe_fft_plan_free(dec->fft);
        free(dec);
    }
}

static unsigned
index_bits(const char* bits, const unsigned char* positions, unsigned count) {
    unsigned result = 0;
    for (unsigned i = 0; i < count; ++i) {
        result = (result << 1) | (unsigned)bits[positions[i]];
    }
    return result;
}

static unsigned
contiguous(const char* bits, unsigned first, unsigned count) {
    unsigned result = 0;
    for (unsigned i = 0; i < count; ++i) {
        result = (result << 1) | (unsigned)bits[first + i];
    }
    return result;
}

static void
remove_mean(float* x, int count) {
    float mean = 0;
    for (int i = 0; i < count; ++i) {
        mean += x[i];
    }
    mean /= (float)count;
    for (int i = 0; i < count; ++i) {
        x[i] -= mean;
    }
}

static float
random_phase(mbe_ambe4000_decoder* dec) {
    uint32_t r = dec->rng;
    r ^= r << 13;
    r ^= r >> 17;
    r ^= r << 5;
    dec->rng = r;
    return (float)(r >> 8) * (2.0f * PI4000 / 16777216.0f);
}

/* Finite Hilbert kernel: approximate minimum-phase regeneration. */
static void
regenerate_phase(const float* logs, int L, float* phase) {
    float b[95] = {0};
    for (int h = 1; h <= L; ++h) {
        b[19 + h] = logs[h - 1];
    }
    for (int h = L + 1; h <= 75; ++h) {
        b[19 + h] = logs[L - 1] - .5f * (float)(h - L);
    }
    for (int h = 1; h <= 19; ++h) {
        b[19 - h] = b[19 + h];
    }
    for (int h = 1; h <= L; ++h) {
        float sum = 0;
        for (int k = 1; k < 20; k += 2) {
            sum += (b[19 + h + k] - b[19 + h - k]) / (float)k;
        }
        phase[h - 1] = (-0.6931471805599453f / PI4000) * sum;
    }
}

/* Voiced grains use linear overlap; independent noise grains use equal-power
 * overlap. All spectral amplitudes are physical PCM units before /7 scaling. */
static void
synthesize(float out[80], mbe_ambe4000_decoder* dec, float frequency, const float* logs, int L, unsigned voicing) {
    static const float low[6] = {0, 250, 750, 1250, 1750, 2750};
    static const float high[6] = {500, 1000, 1500, 2000, 3000, 4000};
    float voice[160] = {0}, spectrum[256] = {0}, noise[256], phase[56];
    dec->pulse = fmodf(dec->pulse + 2.0f * PI4000 * (dec->frequency + frequency) * .005f, 2.0f * PI4000);
    dec->frequency = frequency;
    regenerate_phase(logs, L, phase);
    for (int h = 1; h <= L; ++h) {
        float hz = frequency * (float)h;
        if (hz >= 4000.0f) {
            continue;
        }
        int voiced = 0;
        for (int j = 0; j < 6; ++j) {
            if ((voicing & (1u << (5 - j))) && hz >= low[j] && hz < high[j]) {
                voiced = 1;
            }
        }
        float amplitude = exp2f(bound(logs[h - 1], -12.0f, 14.0f)) / 7.0f;
        if (voiced) {
            float omega = 2.0f * PI4000 * hz / 8000.0f;
            float phi = (float)h * dec->pulse + phase[h - 1];
            for (int n = 0; n < 160; ++n) {
                voice[n] += amplitude * cosf(phi + omega * (float)(n - 80));
            }
        } else {
            int first = (int)ceilf(((float)h - .5f) * frequency / 31.25f);
            int end = (int)ceilf(((float)h + .5f) * frequency / 31.25f);
            if (first < 1) {
                first = 1;
            }
            if (end > 128) {
                end = 128;
            }
            if (end > first) {
                float scale = amplitude * 128.0f / sqrtf((float)(end - first));
                for (int k = first; k < end; ++k) {
                    float phi = random_phase(dec);
                    spectrum[2 * k] = scale * cosf(phi);
                    spectrum[2 * k + 1] = scale * sinf(phi);
                }
            }
        }
    }
    (void)mbe_fft_inverse_real(dec->fft, spectrum, noise);
    for (int n = 0; n < 80; ++n) {
        float w = (float)n / 80.0f;
        float y = dec->voiced_tail[n] * (1 - w) + voice[n] * w + dec->noise_tail[n] * cosf(w * PI4000 * .5f)
                  + noise[n] * sinf(w * PI4000 * .5f);
        out[n] = bound(y, -LIMIT4000, LIMIT4000);
        dec->voiced_tail[n] = voice[n + 80];
        dec->noise_tail[n] = noise[n + 80];
    }
}

static unsigned
special_frame(float out[160], const char* bits, unsigned pitch, mbe_ambe4000_decoder* dec) {
    memset(dec->voiced_tail, 0, sizeof(dec->voiced_tail));
    memset(dec->noise_tail, 0, sizeof(dec->noise_tail));
    memset(out, 0, 160 * sizeof(*out));
    if (pitch == 124) {
        return MBE_PROCESS_FLAG_SILENCE | MBE_PROCESS_FLAG_MUTE;
    }
    if (pitch == 125 || pitch == 127) {
        return MBE_PROCESS_FLAG_ERASURE | MBE_PROCESS_FLAG_MUTE;
    }
    static const unsigned char upper[8] = {4, 0, 1, 2, 3, 6, 5, 7};
    static const unsigned char id_low[5] = {9, 56, 57, 10, 11};
    static const unsigned char volume[8] = {12, 13, 14, 15, 16, 58, 59, 17};
    unsigned id = (unsigned)upper[contiguous(bits, 6, 3)] * 32u + index_bits(bits, id_low, 5);
    unsigned amplitude = index_bits(bits, volume, 8);
    float f1 = 0, f2 = 0;
    if (id >= 5 && id <= 122) {
        f1 = 31.25f * (float)id;
    } else if (id >= 128 && id <= 143) {
        static const float rows[4] = {697, 770, 852, 941}, cols[4] = {1209, 1336, 1477, 1633};
        f1 = rows[(id - 128) % 4];
        f2 = cols[(id - 128) / 4];
    } else if (id >= 144 && id <= 146) {
        static const float pairs[3][2] = {{440, 350}, {480, 440}, {620, 480}};
        f1 = pairs[id - 144][0];
        f2 = pairs[id - 144][1];
    }
    if (f1 == 0) {
        return MBE_PROCESS_FLAG_TONE | MBE_PROCESS_FLAG_ERASURE | MBE_PROCESS_FLAG_MUTE;
    }
#ifdef DISABLE_AMBE_TONES
    (void)amplitude;
    (void)f2;
    return MBE_PROCESS_FLAG_TONE | MBE_PROCESS_FLAG_MUTE;
#else
    float gain = exp2f((float)amplitude / 17.0f + 1.083f) / 7.0f;
    for (int n = 0; n < 160; ++n) {
        float y = gain * cosf(dec->tone_phase[0]);
        dec->tone_phase[0] = fmodf(dec->tone_phase[0] + 2.0f * PI4000 * f1 / 8000.0f, 2.0f * PI4000);
        if (f2 > 0) {
            y += gain * cosf(dec->tone_phase[1]);
            dec->tone_phase[1] = fmodf(dec->tone_phase[1] + 2.0f * PI4000 * f2 / 8000.0f, 2.0f * PI4000);
        }
        out[n] = bound(y, -LIMIT4000, LIMIT4000);
    }
    return MBE_PROCESS_FLAG_TONE;
#endif
}

static unsigned
speech_frame(float out[160], const char* bits, mbe_ambe4000_decoder* dec) {
    unsigned pitch = contiguous(bits, 0, 6) * 2u + (unsigned)bits[79];
    if (pitch >= 124) {
        return special_frame(out, bits, pitch, dec);
    }
    unsigned gain = contiguous(bits, 6, 4) * 4u + contiguous(bits, 56, 2);
    unsigned prba0 = contiguous(bits, 10, 7) * 4u + contiguous(bits, 58, 2);
    unsigned prba1 = contiguous(bits, 17, 5) * 4u + contiguous(bits, 60, 2);
    unsigned interpolation = contiguous(bits, 38, 2), voicing = contiguous(bits, 50, 6);
    static const unsigned char subgain_bits[4] = {40, 41, 42, 66};
    static const unsigned char shape0_bits[7] = {43, 44, 45, 67, 68, 69, 70};
    static const unsigned char shape1_bits[6] = {46, 47, 71, 72, 73, 74};
    static const unsigned char shape2_bits[6] = {48, 49, 75, 76, 77, 78};
    const float* shape0 = ambe4000_shape123[index_bits(bits, shape0_bits, 7)];
    const float* shape1 = ambe4000_shape456[index_bits(bits, shape1_bits, 6)];
    const float* shape2 = ambe4000_shape789[index_bits(bits, shape2_bits, 6)];
    float correction[9];
    for (int i = 0; i < 3; ++i) {
        correction[i] = shape0[i];
        correction[3 + i] = shape1[i];
        correction[6 + i] = shape2[i];
    }
    float f = ambe4000_pitch[pitch];
    int L = (int)(3800.0f / f);
    if (L < 9) {
        L = 9;
    }
    if (L > 56) {
        L = 56;
    }
    float prior[56], current[56], mid[56], extra[56], ri[9];
    mbe_ambe2400_reconstruct_prba((int)prba0, (int)prba1, ri);
    int offset = 0;
    for (int block = 0; block < 4; ++block) {
        int count = ambe4000_blocks[L][block];
        unsigned code = contiguous(bits, 22u + (unsigned)block * 4u, 4) * 2u + (unsigned)bits[62 + block];
        float coeff[6] = {(ri[1 + 2 * block] + ri[2 + 2 * block]) * .5f,
                          (ri[1 + 2 * block] - ri[2 + 2 * block]) * .3535533905932738f,
                          0,
                          0,
                          0,
                          0};
        for (int k = 2; k < 6; ++k) {
            coeff[k] = ambe4000_hoc[block][code][k - 2];
        }
        for (int j = 0; j < count; ++j) {
            float value = coeff[0];
            for (int k = 1; k < count && k < 6; ++k) {
                value += 2 * coeff[k] * cosf(PI4000 * (float)k * ((float)j + .5f) / (float)count);
            }
            current[offset++] = value;
        }
    }
    for (int h = 0; h < L; ++h) {
        float position = (float)(h + 1) * (float)dec->L / (float)L - 1;
        if (position <= 0) {
            prior[h] = dec->shape[0];
        } else if (position >= (float)(dec->L - 1)) {
            prior[h] = dec->shape[dec->L - 1];
        } else {
            int k = (int)position;
            float fraction = position - (float)k;
            prior[h] = (1 - fraction) * dec->shape[k] + fraction * dec->shape[k + 1];
        }
    }
    remove_mean(prior, L);
    for (int h = 0; h < L; ++h) {
        current[h] += .8f * prior[h];
    }
    remove_mean(current, L);
    float g = ambe4000_gain[gain] + 1.0f - .5f * log2f((float)L);
    float mg = (g + dec->gain) * .5f + ambe4000_subgain[index_bits(bits, subgain_bits, 4)];
    static const float beta[4] = {.1f, .3f, .5f, .9f};
    for (int h = 0; h < L; ++h) {
        extra[h] = 0;
        for (int k = 1; k <= 9; ++k) {
            extra[h] += 2 * correction[k - 1] * cosf(PI4000 * ((float)h + .5f) * (float)k / (float)L);
        }
    }
    remove_mean(extra, L);
    for (int h = 0; h < L; ++h) {
        mid[h] = (1 - beta[interpolation]) * prior[h] + beta[interpolation] * current[h] + extra[h] + mg;
        dec->shape[h] = current[h];
        current[h] += g;
    }
    float fmid = (f + dec->frequency) * .5f;
    synthesize(out, dec, fmid, mid, L, voicing);
    synthesize(out + 80, dec, f, current, L, voicing);
    dec->L = L;
    dec->gain = g;
    return 0;
}

int
mbe_processAmbe4000Dataf(float out[160], mbe_process_result* result, const char bits[80], mbe_ambe4000_decoder* dec) {
    mbe_initProcessResult(result);
    if (!out || !dec) {
        return MBE_STATUS_INVALID_ARGUMENT;
    }
    int ret = mbe_validate_bits(bits, 80);
    if (ret < 0) {
        return ret;
    }
    unsigned flags = speech_frame(out, bits, dec);
    if (result) {
        result->flags = flags;
    }
    return 0;
}

int
mbe_processAmbe4000Data(short out[160], mbe_process_result* result, const char bits[80], mbe_ambe4000_decoder* dec) {
    float samples[160];
    if (!out) {
        mbe_initProcessResult(result);
        return MBE_STATUS_INVALID_ARGUMENT;
    }
    int ret = mbe_processAmbe4000Dataf(samples, result, bits, dec);
    if (ret >= 0) {
        mbe_floattoshort(samples, out);
    }
    return ret;
}

int
mbe_processAmbe6400x4000Framef(float out[160], mbe_process_result* result, const char frame[128], char bits[80],
                               mbe_ambe4000_decoder* dec) {
    char decoded[80];
    mbe_initProcessResult(result);
    if (!out || !dec) {
        return MBE_STATUS_INVALID_ARGUMENT;
    }
    int ret = mbe_decodeAmbe6400x4000Frame(frame, decoded, result);
    if (ret < 0) {
        return ret;
    }
    unsigned flags = speech_frame(out, decoded, dec);
    if (result) {
        result->flags |= flags;
    }
    if (bits) {
        memcpy(bits, decoded, 80);
    }
    return ret;
}

int
mbe_processAmbe6400x4000SoftFramef(float out[160], mbe_process_result* result, const mbe_soft_bit frame[128],
                                   char bits[80], mbe_ambe4000_decoder* dec) {
    char decoded[80];
    mbe_initProcessResult(result);
    if (!out || !dec) {
        return MBE_STATUS_INVALID_ARGUMENT;
    }
    int ret = mbe_decodeAmbe6400x4000SoftFrame(frame, decoded, result);
    if (ret < 0) {
        return ret;
    }
    unsigned flags = speech_frame(out, decoded, dec);
    if (result) {
        result->flags |= flags;
    }
    if (bits) {
        memcpy(bits, decoded, 80);
    }
    return ret;
}

int
mbe_processAmbe6400x4000Frame(short out[160], mbe_process_result* result, const char frame[128], char bits[80],
                              mbe_ambe4000_decoder* dec) {
    float samples[160];
    if (!out) {
        mbe_initProcessResult(result);
        return MBE_STATUS_INVALID_ARGUMENT;
    }
    int ret = mbe_processAmbe6400x4000Framef(samples, result, frame, bits, dec);
    if (ret >= 0) {
        mbe_floattoshort(samples, out);
    }
    return ret;
}

int
mbe_processAmbe6400x4000SoftFrame(short out[160], mbe_process_result* result, const mbe_soft_bit frame[128],
                                  char bits[80], mbe_ambe4000_decoder* dec) {
    float samples[160];
    if (!out) {
        mbe_initProcessResult(result);
        return MBE_STATUS_INVALID_ARGUMENT;
    }
    int ret = mbe_processAmbe6400x4000SoftFramef(samples, result, frame, bits, dec);
    if (ret >= 0) {
        mbe_floattoshort(samples, out);
    }
    return ret;
}

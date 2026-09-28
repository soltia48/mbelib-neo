// SPDX-License-Identifier: GPL-2.0-or-later
/** @file AMBE+ 4000/2400 channel coding (rate index 26).
 * Parameter bits are in priority order, before interleaving.
 * Frame layout and API details are in docs/ambe4000.md.
 */
#include <limits.h>
#include <string.h>

#include "mbe_result.h"
#include "mbelib-neo/mbelib.h"

/* Read columns of a 12-row ragged matrix (eight long, four short rows). */
static unsigned
channel_position(unsigned index) {
    unsigned column = index / 12u;
    unsigned row = index % 12u;
    return row < 8u ? row * 11u + column : 88u + (row - 8u) * 10u + column;
}

static unsigned
parity(unsigned value) {
    value ^= value >> 4;
    value ^= value >> 2;
    value ^= value >> 1;
    return value & 1u;
}

int
mbe_encodeAmbe6400x4000Frame(const char ambe_d[80], char ambe_fr[128]) {
    char frame[128];
    unsigned state = 0;
    if (!ambe_fr) {
        return MBE_STATUS_INVALID_ARGUMENT;
    }
    int ret = mbe_validate_bits(ambe_d, 80u);
    if (ret < 0) {
        return ret;
    }
    for (unsigned t = 0; t < 42u; ++t) {
        state = ((state << 1) | (t < 36u ? (unsigned)ambe_d[t] : 0u)) & 127u;
        frame[channel_position(2u * t)] = (char)parity(state & 0x4fu);
        frame[channel_position(2u * t + 1u)] = (char)parity(state & 0x6du);
    }
    for (unsigned i = 36u; i < 80u; ++i) {
        frame[channel_position(i + 48u)] = ambe_d[i];
    }
    memcpy(ambe_fr, frame, sizeof(frame));
    return 0;
}

/* A full terminated trellis: 36 source bits followed by six zero tail bits.
 * Costs fit in int (84 * 255), including for zero-reliability erasures.
 * Traceback always starts at state zero. No heap or mutable global state. */
static int
ambe4000_viterbi(const mbe_soft_bit frame[128], char data[80], mbe_process_result* result, unsigned flags) {
    int metric[64], next[64];
    unsigned char predecessor[42][64];
    char decoded[80], encoded[128];
    const int infinity = INT_MAX;
    for (unsigned s = 0; s < 64u; ++s) {
        metric[s] = s == 0u ? 0 : infinity;
    }
    for (unsigned t = 0; t < 42u; ++t) {
        mbe_soft_bit a = frame[channel_position(2u * t)];
        mbe_soft_bit b = frame[channel_position(2u * t + 1u)];
        for (unsigned s = 0; s < 64u; ++s) {
            next[s] = infinity;
            predecessor[t][s] = 0;
        }
        for (unsigned s = 0; s < 64u; ++s) {
            if (metric[s] == infinity) {
                continue;
            }
            for (unsigned bit = 0; bit < (t < 36u ? 2u : 1u); ++bit) {
                unsigned reg = (s << 1) | bit;
                unsigned target = reg & 63u;
                int cost = metric[s];
                if (parity(reg & 0x4fu) != a.bit) {
                    cost += a.reliability;
                }
                if (parity(reg & 0x6du) != b.bit) {
                    cost += b.reliability;
                }
                if (cost < next[target]) {
                    next[target] = cost;
                    predecessor[t][target] = (unsigned char)s;
                }
            }
        }
        memcpy(metric, next, sizeof(metric));
    }
    unsigned state = 0;
    for (int t = 41; t >= 0; --t) {
        if (t < 36) {
            decoded[t] = (char)(state & 1u);
        }
        state = predecessor[t][state];
    }
    for (unsigned i = 36; i < 80u; ++i) {
        decoded[i] = (char)frame[channel_position(i + 48u)].bit;
    }
    (void)mbe_encodeAmbe6400x4000Frame(decoded, encoded);
    int errors = 0;
    for (unsigned i = 0; i < 84u; ++i) {
        unsigned j = channel_position(i);
        errors += encoded[j] != (char)frame[j].bit;
    }
    memcpy(data, decoded, sizeof(decoded));
    if (result) {
        result->protected_errors = errors;
        result->total_errors = errors;
        result->flags = flags;
    }
    return errors;
}

int
mbe_decodeAmbe6400x4000SoftFrame(const mbe_soft_bit ambe_fr[128], char ambe_d[80], mbe_process_result* result) {
    if (result) {
        mbe_initProcessResult(result);
    }
    if (!ambe_d) {
        return MBE_STATUS_INVALID_ARGUMENT;
    }
    int ret = mbe_validate_soft_bits(ambe_fr, 128u);
    if (ret < 0) {
        return ret;
    }
    return ambe4000_viterbi(ambe_fr, ambe_d, result, MBE_PROCESS_FLAG_SOFT_INPUT);
}

int
mbe_decodeAmbe6400x4000Frame(const char ambe_fr[128], char ambe_d[80], mbe_process_result* result) {
    mbe_soft_bit soft[128];
    if (result) {
        mbe_initProcessResult(result);
    }
    if (!ambe_d) {
        return MBE_STATUS_INVALID_ARGUMENT;
    }
    int ret = mbe_validate_bits(ambe_fr, 128u);
    if (ret < 0) {
        return ret;
    }
    for (unsigned i = 0; i < 128u; ++i) {
        soft[i].bit = (unsigned char)ambe_fr[i];
        soft[i].reliability = 1;
    }
    return ambe4000_viterbi(soft, ambe_d, result, 0u);
}

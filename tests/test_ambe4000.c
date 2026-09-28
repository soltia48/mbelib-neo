// SPDX-License-Identifier: GPL-2.0-or-later
#include <assert.h>
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
reference_vectors(void) {
    for (unsigned v = 0; v < sizeof(ambe4000_vectors) / sizeof(ambe4000_vectors[0]); ++v) {
        char data[80], frame[128], expected[128], decoded[80];
        mbe_process_result result;
        unpack(ambe4000_vectors[v].data, data, 80);
        unpack(ambe4000_vectors[v].channel, expected, 128);
        assert(mbe_encodeAmbe6400x4000Frame(data, frame) == 0);
        assert(memcmp(frame, expected, sizeof(frame)) == 0);
        assert(mbe_decodeAmbe6400x4000Frame(expected, decoded, &result) == 0);
        assert(memcmp(data, decoded, sizeof(data)) == 0);
        assert(result.total_errors == 0 && result.flags == 0);
        /* All 128 one-bit corruptions: exactly 84 protected, 44 unprotected. */
        int corrected = 0, unprotected = 0;
        for (int i = 0; i < 128; ++i) {
            frame[i] ^= 1;
            int distance = mbe_decodeAmbe6400x4000Frame(frame, decoded, &result);
            assert(result.total_errors == distance && result.protected_errors == distance);
            if (distance == 1) {
                assert(memcmp(data, decoded, sizeof(data)) == 0);
                ++corrected;
            } else {
                int changed = 0;
                assert(distance == 0);
                for (int j = 0; j < 80; ++j) {
                    if (data[j] != decoded[j]) {
                        assert(j >= 36);
                        ++changed;
                    }
                }
                assert(changed == 1);
                ++unprotected;
            }
            frame[i] ^= 1;
        }
        assert(corrected == 84 && unprotected == 44);
    }
}

static void
soft_decode(void) {
    char data[80], frame[128], decoded[80];
    mbe_soft_bit soft[128];
    mbe_process_result result;
    unpack(ambe4000_vectors[8].data, data, 80);
    unpack(ambe4000_vectors[8].channel, frame, 128);
    for (int i = 0; i < 128; ++i) {
        soft[i].bit = (unsigned char)frame[i];
        soft[i].reliability = 255;
    }
    /* Burst over protected columns, with weak confidence on the wrong bits. */
    for (int i = 0; i < 7; ++i) {
        soft[i].bit ^= 1;
        soft[i].reliability = 1;
    }
    assert(mbe_decodeAmbe6400x4000SoftFrame(soft, decoded, &result) == 7);
    assert(memcmp(data, decoded, sizeof(data)) == 0);
    assert(result.flags == MBE_PROCESS_FLAG_SOFT_INPUT);
    for (int i = 0; i < 128; ++i) {
        soft[i].reliability = 0;
    }
    assert(mbe_decodeAmbe6400x4000SoftFrame(soft, decoded, NULL) >= 0);
    for (int i = 0; i < 80; ++i) {
        assert(decoded[i] == 0 || decoded[i] == 1);
    }
}

static void
invalid_input(void) {
    char data[80] = {0}, frame[128] = {0}, sentinel[80];
    mbe_soft_bit soft[128] = {{0, 0}};
    memset(sentinel, 42, sizeof(sentinel));
    assert(mbe_encodeAmbe6400x4000Frame(NULL, frame) == MBE_STATUS_INVALID_ARGUMENT);
    assert(mbe_encodeAmbe6400x4000Frame(data, NULL) == MBE_STATUS_INVALID_ARGUMENT);
    data[79] = 2;
    assert(mbe_encodeAmbe6400x4000Frame(data, frame) == MBE_STATUS_INVALID_BITS);
    assert(mbe_decodeAmbe6400x4000Frame(NULL, data, NULL) == MBE_STATUS_INVALID_ARGUMENT);
    assert(mbe_decodeAmbe6400x4000Frame(frame, NULL, NULL) == MBE_STATUS_INVALID_ARGUMENT);
    frame[127] = -1;
    assert(mbe_decodeAmbe6400x4000Frame(frame, sentinel, NULL) == MBE_STATUS_INVALID_BITS);
    assert(mbe_decodeAmbe6400x4000SoftFrame(NULL, data, NULL) == MBE_STATUS_INVALID_ARGUMENT);
    assert(mbe_decodeAmbe6400x4000SoftFrame(soft, NULL, NULL) == MBE_STATUS_INVALID_ARGUMENT);
    soft[127].bit = 2;
    assert(mbe_decodeAmbe6400x4000SoftFrame(soft, sentinel, NULL) == MBE_STATUS_INVALID_BITS);
    for (int i = 0; i < 80; ++i) {
        assert(sentinel[i] == 42);
    }
}

int
main(void) {
    reference_vectors();
    soft_decode();
    invalid_input();
    return 0;
}

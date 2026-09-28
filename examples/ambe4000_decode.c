// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * Copyright (C) 2026 Rhizomatica
 * Author: Rafael Diniz <rafael@rhizomatica.org>
 */

/**
 * @file
 * @brief Decode packed AMBE+ 6400/4000 frames to 8 kHz mono PCM16 WAV.
 * Usage: ambe4000_decode < input.ambe > output.wav
 * Input is consecutive 16-byte channel frames, MSB first, without headers.
 * The speech model is approximate; see docs/ambe4000.md.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mbelib-neo/mbelib.h>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#define FRAME_SAMPLES     160
#define DV_FRAME_BYTES    16
#define MAX_WAV_DATA_SIZE (UINT32_MAX - 36u)

struct pcm_buffer {
    unsigned char* data;
    size_t size;
    size_t capacity;
};

static void
write_le32(unsigned char* bytes, uint32_t value) {
    for (int i = 0; i < 4; i++) {
        bytes[i] = (unsigned char)(value >> (8 * i));
    }
}

static int
write_wav_header(FILE* fp, uint32_t data_size) {
    unsigned char header[44] = {'R', 'I', 'F', 'F', 0,  0, 0,   0,   'W', 'A',  'V',  'E', 'f', 'm',  't',
                                ' ', 16,  0,   0,   0,  1, 0,   1,   0,   0x40, 0x1f, 0,   0,   0x80, 0x3e,
                                0,   0,   2,   0,   16, 0, 'd', 'a', 't', 'a',  0,    0,   0,   0};
    write_le32(header + 4, 36 + data_size);
    write_le32(header + 40, data_size);
    return fwrite(header, 1, sizeof(header), fp) == sizeof(header) ? 0 : -1;
}

static void
store_samples(unsigned char* bytes, short pcm[FRAME_SAMPLES]) {
    for (int i = 0; i < FRAME_SAMPLES; i++) {
        int value = pcm[i];
        if (value > 31128) {
            value = 31128;
        }
        if (value < -31128) {
            value = -31128;
        }
        uint16_t sample = (uint16_t)value;
        bytes[2 * (size_t)i] = (unsigned char)sample;
        bytes[2 * (size_t)i + 1] = (unsigned char)(sample >> 8);
    }
}

static int
append_samples(struct pcm_buffer* buffer, short pcm[FRAME_SAMPLES]) {
    if (buffer->size > MAX_WAV_DATA_SIZE - FRAME_SAMPLES * 2u) {
        (void)fprintf(stderr, "WAV size limit exceeded\n");
        return -1;
    }
    size_t needed = buffer->size + (size_t)FRAME_SAMPLES * 2;
    if (needed > buffer->capacity) {
        size_t capacity = buffer->capacity;
        if (capacity > MAX_WAV_DATA_SIZE / 2u) {
            capacity = MAX_WAV_DATA_SIZE;
        } else {
            capacity *= 2;
        }
        unsigned char* data = (unsigned char*)realloc(buffer->data, capacity);
        if (data == NULL) {
            (void)fprintf(stderr, "out of memory buffering decoded PCM\n");
            return -1;
        }
        buffer->data = data;
        buffer->capacity = capacity;
    }
    store_samples(buffer->data + buffer->size, pcm);
    buffer->size = needed;
    return 0;
}

static int
decode_frames(FILE* fin, struct pcm_buffer* buffer) {
    mbe_ambe4000_decoder* decoder = mbe_ambe4000DecoderAlloc();
    if (!decoder) {
        (void)fprintf(stderr, "out of memory allocating decoder\n");
        return -1;
    }
    unsigned char dv[DV_FRAME_BYTES];
    uint32_t frames = 0, corrections = 0, muted = 0;
    size_t n;
    int status = -1;
    while ((n = fread(dv, 1, sizeof(dv), fin)) == sizeof(dv)) {
        short pcm[FRAME_SAMPLES];
        char frame[128];
        mbe_process_result result;
        for (int i = 0; i < 128; ++i) {
            frame[i] = (char)((dv[i / 8] >> (7 - i % 8)) & 1);
        }
        int distance = mbe_processAmbe6400x4000Frame(pcm, &result, frame, NULL, decoder);
        if (distance < 0 || append_samples(buffer, pcm) < 0) {
            (void)fprintf(stderr, "decode error\n");
            goto done;
        }
        corrections += (uint32_t)distance;
        muted += (result.flags & MBE_PROCESS_FLAG_MUTE) != 0;
        frames++;
    }
    if (ferror(fin) != 0 || n != 0) {
        (void)fprintf(stderr, "read error or trailing partial 16-byte AMBE frame\n");
        goto done;
    }
    (void)fprintf(stderr, "decoded %u frames; %u corrected channel bits; %u muted frames\n", frames, corrections,
                  muted);
    status = 0;
done:
    mbe_ambe4000DecoderFree(decoder);
    return status;
}

int
main(int argc, char** argv) {
    if (argc != 1) {
        (void)fprintf(stderr, "usage: %s < input.ambe > output.wav\n", argv[0]);
        return 1;
    }
#ifdef _WIN32
    if (_setmode(_fileno(stdin), _O_BINARY) == -1 || _setmode(_fileno(stdout), _O_BINARY) == -1) {
        perror("cannot set binary stdin/stdout");
        return 1;
    }
#endif
    struct pcm_buffer buffer = {NULL, 0, (size_t)FRAME_SAMPLES * 2};
    buffer.data = (unsigned char*)malloc(buffer.capacity);
    if (buffer.data == NULL) {
        (void)fprintf(stderr, "out of memory buffering decoded PCM\n");
        return 1;
    }
    int ret = 1;
    if (decode_frames(stdin, &buffer) == 0) {
        if (write_wav_header(stdout, (uint32_t)buffer.size) < 0
            || fwrite(buffer.data, 1, buffer.size, stdout) != buffer.size) {
            perror("WAV write error");
        } else {
            ret = 0;
        }
    }
    free(buffer.data);
    if (fflush(stdout) != 0 || ferror(stdout) != 0) {
        perror("write error");
        ret = 1;
    }
    return ret;
}

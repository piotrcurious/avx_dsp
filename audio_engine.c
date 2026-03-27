#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <alsa/asoundlib.h>
#include "avx_dsp.h"

#define SAMPLE_RATE 44100
#define FFT_SIZE 1024

typedef struct {
    snd_pcm_t *handle;
    float *buffer;
    float *fft_in;
    float *fft_out;
    float *mag;
    float *window_tmp; // Pre-allocated temporary buffer
} audio_ctx_t;

audio_ctx_t* audio_init(const char *device) {
    audio_ctx_t *ctx = malloc(sizeof(audio_ctx_t));
    int err;
    if ((err = snd_pcm_open(&ctx->handle, device, SND_PCM_STREAM_CAPTURE, 0)) < 0) {
        fprintf(stderr, "Cannot open audio device %s (%s)\n", device, snd_strerror(err));
        free(ctx);
        return NULL;
    }
    snd_pcm_set_params(ctx->handle, SND_PCM_FORMAT_S16_LE, SND_PCM_ACCESS_RW_INTERLEAVED, 1, SAMPLE_RATE, 1, 50000);
    ctx->buffer = avx_malloc(FFT_SIZE);
    ctx->fft_in = avx_malloc(2 * FFT_SIZE);
    ctx->fft_out = ctx->fft_in;
    ctx->mag = avx_malloc(FFT_SIZE);
    ctx->window_tmp = avx_malloc(FFT_SIZE);
    return ctx;
}

int audio_capture(audio_ctx_t *ctx) {
    short buf[FFT_SIZE];
    int err = snd_pcm_readi(ctx->handle, buf, FFT_SIZE);
    if (err < 0) {
        snd_pcm_prepare(ctx->handle);
        return err;
    }
    for (int i = 0; i < FFT_SIZE; i++) {
        ctx->buffer[i] = (float)buf[i] / 32768.0f;
    }
    return 0;
}

void audio_process(audio_ctx_t *ctx) {
    // Use pre-allocated buffer
    memcpy(ctx->window_tmp, ctx->buffer, FFT_SIZE * sizeof(float));
    avx_window_hann(ctx->window_tmp, FFT_SIZE);

    // Prepare complex input
    for (int i = 0; i < FFT_SIZE; i++) {
        ctx->fft_in[2 * i] = ctx->window_tmp[i];
        ctx->fft_in[2 * i + 1] = 0.0f;
    }

    // FFT
    avx_fft_array(ctx->fft_out, FFT_SIZE);

    // Magnitude
    avx_vector_magnitude(ctx->fft_out, FFT_SIZE, ctx->mag);
}

void audio_cleanup(audio_ctx_t *ctx) {
    snd_pcm_close(ctx->handle);
    avx_free(ctx->buffer);
    avx_free(ctx->fft_in);
    avx_free(ctx->mag);
    avx_free(ctx->window_tmp);
    free(ctx);
}

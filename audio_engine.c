#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <alsa/asoundlib.h>
#include "audio_engine.h"
#include "avx_dsp.h"

#define SAMPLE_RATE 44100

typedef struct {
    snd_pcm_t *handle;
    float *buffer;
    float *fft_in;
    float *fft_out;
    float *mag;
    float *phase;
    float *window_tmp;
    size_t fft_size;
    short *capture_buf; // Avoid allocation in loop
} internal_audio_ctx_t;

audio_ctx_t* audio_init(const char *device, size_t fft_size) {
    internal_audio_ctx_t *ctx = (internal_audio_ctx_t*)malloc(sizeof(internal_audio_ctx_t));
    int err;
    if ((err = snd_pcm_open(&ctx->handle, device, SND_PCM_STREAM_CAPTURE, 0)) < 0) {
        fprintf(stderr, "Cannot open audio device %s (%s)\n", device, snd_strerror(err));
        free(ctx);
        return NULL;
    }
    snd_pcm_set_params(ctx->handle, SND_PCM_FORMAT_S16_LE, SND_PCM_ACCESS_RW_INTERLEAVED, 1, SAMPLE_RATE, 1, 50000);

    ctx->fft_size = fft_size;
    ctx->buffer = avx_malloc(fft_size);
    ctx->fft_in = avx_malloc(2 * fft_size);
    ctx->fft_out = ctx->fft_in;
    ctx->mag = avx_malloc(fft_size);
    ctx->phase = avx_malloc(fft_size);
    ctx->window_tmp = avx_malloc(fft_size);
    ctx->capture_buf = (short*)malloc(fft_size * sizeof(short));

    memset(ctx->buffer, 0, fft_size * sizeof(float));
    return (audio_ctx_t*)ctx;
}

int audio_capture(audio_ctx_t *uctx) {
    internal_audio_ctx_t *ctx = (internal_audio_ctx_t*)uctx;
    int err = snd_pcm_readi(ctx->handle, ctx->capture_buf, ctx->fft_size);
    if (err < 0) {
        if (err == -EPIPE) {
            fprintf(stderr, "Overrun occurred\n");
            snd_pcm_prepare(ctx->handle);
        } else {
            fprintf(stderr, "Read error: %s\n", snd_strerror(err));
        }
    } else {
        for (int i = 0; i < err; i++) {
            ctx->buffer[i] = (float)ctx->capture_buf[i] / 32768.0f;
        }
    }
    return err;
}

void audio_process(audio_ctx_t *uctx) {
    internal_audio_ctx_t *ctx = (internal_audio_ctx_t*)uctx;
    memcpy(ctx->window_tmp, ctx->buffer, ctx->fft_size * sizeof(float));
    avx_window_hann(ctx->window_tmp, ctx->fft_size);

    for (size_t i = 0; i < ctx->fft_size; i++) {
        ctx->fft_in[2 * i] = ctx->window_tmp[i];
        ctx->fft_in[2 * i + 1] = 0.0f;
    }

    avx_fft_array(ctx->fft_out, ctx->fft_size);
    avx_vector_magnitude(ctx->fft_out, ctx->fft_size, ctx->mag);
    avx_vector_phase(ctx->fft_out, ctx->fft_size, ctx->phase);
}

void audio_cleanup(audio_ctx_t *uctx) {
    internal_audio_ctx_t *ctx = (internal_audio_ctx_t*)uctx;
    if (ctx->handle) snd_pcm_close(ctx->handle);
    if (ctx->buffer) avx_free(ctx->buffer);
    if (ctx->fft_in) avx_free(ctx->fft_in);
    if (ctx->mag) avx_free(ctx->mag);
    if (ctx->phase) avx_free(ctx->phase);
    if (ctx->window_tmp) avx_free(ctx->window_tmp);
    if (ctx->capture_buf) free(ctx->capture_buf);
    free(ctx);
}

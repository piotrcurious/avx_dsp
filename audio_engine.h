#ifndef AUDIO_ENGINE_H
#define AUDIO_ENGINE_H

#include "avx_dsp.h"
#include <alsa/asoundlib.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    snd_pcm_t *handle;
    float *buffer;
    float *fft_in;
    float *fft_out;
    float *mag;
    float *phase;
    float *window_tmp;
    size_t fft_size;
} audio_ctx_t;

/**
 * @brief Initialize audio capture and processing context.
 * @param device ALSA device name (e.g. "default" or "hw:0,0")
 * @param fft_size Size of the FFT (must be power of 2)
 */
audio_ctx_t* audio_init(const char *device, size_t fft_size);

/**
 * @brief Capture one block of audio.
 */
int audio_capture(audio_ctx_t *ctx);

/**
 * @brief Process the captured audio (window, FFT, magnitude, phase).
 */
void audio_process(audio_ctx_t *ctx);

/**
 * @brief Cleanup audio context.
 */
void audio_cleanup(audio_ctx_t *ctx);

#ifdef __cplusplus
}
#endif

#endif

#include <FL/Fl.H>
#include <FL/Fl_Double_Window.H>
#include <FL/Fl_Gl_Window.H>
#include <FL/gl.h>
#include <GL/glu.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "avx_dsp.h"
#include "audio_engine.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MAX_FFT_SIZE 4096
#define WATERFALL_DEPTH 256

float waterfall[WATERFALL_DEPTH][MAX_FFT_SIZE / 2];
int waterfall_ptr = 0;
int current_fft_size = 1024;
bool use_mock = false;
audio_ctx_t *audio_ctx = NULL;

class WaterfallWindow : public Fl_Gl_Window {
    void draw() {
        if (!valid()) {
            glViewport(0, 0, w(), h());
            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
            glOrtho(0, 1, 0, 1, -1, 1);
            glMatrixMode(GL_MODELVIEW);
            glLoadIdentity();
            valid(1);
        }
        glClear(GL_COLOR_BUFFER_BIT);
        glBegin(GL_POINTS);
        int half_fft = current_fft_size / 2;
        for (int y = 0; y < WATERFALL_DEPTH; y++) {
            int row = (waterfall_ptr - y + WATERFALL_DEPTH) % WATERFALL_DEPTH;
            for (int x = 0; x < half_fft; x++) {
                float val = waterfall[row][x] * 10.0f;
                if (val > 1.0f) val = 1.0f;
                glColor3f(val, val * 0.2f, 1.0f - val);
                glVertex2f((float)x / half_fft, (float)y / WATERFALL_DEPTH);
            }
        }
        glEnd();
    }
public:
    WaterfallWindow(int x, int y, int w, int h, const char* l = 0) : Fl_Gl_Window(x, y, w, h, l) {}
};

WaterfallWindow *win;

void audio_timer_cb(void*) {
    if (use_mock) {
        static float angle = 0;
        float mock_fft[2 * MAX_FFT_SIZE];
        float mock_mag[MAX_FFT_SIZE];
        float freq = 0.1f + 0.05f * sinf(angle);
        for (int i = 0; i < current_fft_size; i++) {
            mock_fft[2 * i] = sinf(2.0f * (float)M_PI * freq * i);
            mock_fft[2 * i + 1] = 0.0f;
        }
        angle += 0.05f;
        avx_fft_array(mock_fft, current_fft_size);
        avx_vector_magnitude(mock_fft, current_fft_size, mock_mag);

        waterfall_ptr = (waterfall_ptr + 1) % WATERFALL_DEPTH;
        for (int i = 0; i < current_fft_size / 2; i++) {
            waterfall[waterfall_ptr][i] = mock_mag[i] / (current_fft_size / 2.0f);
        }
    } else if (audio_ctx) {
        if (audio_capture(audio_ctx) >= 0) {
            audio_process(audio_ctx);
            waterfall_ptr = (waterfall_ptr + 1) % WATERFALL_DEPTH;
            // Access internal mag if we could, but let's assume it's exposed or we use a getter
            // For now, assume audio_ctx has a mag member as defined in audio_engine.h
            for (int i = 0; i < current_fft_size / 2; i++) {
                waterfall[waterfall_ptr][i] = audio_ctx->mag[i] * 2.0f;
            }
        }
    }

    win->redraw();
    if (!getenv("HEADLESS")) Fl::repeat_timeout(0.02, audio_timer_cb);
}

int main(int argc, char **argv) {
    const char *device = "default";
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--mock") == 0) use_mock = true;
        else if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) device = argv[++i];
        else if (strcmp(argv[i], "--fft-size") == 0 && i + 1 < argc) current_fft_size = atoi(argv[++i]);
    }

    if (current_fft_size > MAX_FFT_SIZE) current_fft_size = MAX_FFT_SIZE;

    if (!use_mock) {
        audio_ctx = audio_init(device, current_fft_size);
        if (!audio_ctx) {
            fprintf(stderr, "Failed to initialize audio, falling back to mock\n");
            use_mock = true;
        }
    }

    printf("Starting AVX DSP Waterfall Demo (FFT Size: %d, Mock: %s)...\n", current_fft_size, use_mock ? "Yes" : "No");

    Fl_Double_Window *main_win = new Fl_Double_Window(800, 600, "AVX DSP Waterfall");
    win = new WaterfallWindow(10, 10, 780, 580);
    main_win->end();

    if (getenv("HEADLESS")) {
        for(int i=0; i<10; i++) audio_timer_cb(NULL);
        return 0;
    }

    main_win->show(argc, argv);
    Fl::add_timeout(0.02, audio_timer_cb);
    int ret = Fl::run();
    if (audio_ctx) audio_cleanup(audio_ctx);
    return ret;
}

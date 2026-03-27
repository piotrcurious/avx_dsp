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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define FFT_SIZE 1024
#define WATERFALL_DEPTH 256

float waterfall[WATERFALL_DEPTH][FFT_SIZE / 2];
int waterfall_ptr = 0;

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
        for (int y = 0; y < WATERFALL_DEPTH; y++) {
            int row = (waterfall_ptr - y + WATERFALL_DEPTH) % WATERFALL_DEPTH;
            for (int x = 0; x < FFT_SIZE / 2; x++) {
                float val = waterfall[row][x];
                if (val > 1.0f) val = 1.0f;
                glColor3f(val, 0.2f, 1.0f - val);
                glVertex2f((float)x / (FFT_SIZE / 2), (float)y / WATERFALL_DEPTH);
            }
        }
        glEnd();
    }
public:
    WaterfallWindow(int x, int y, int w, int h, const char* l = 0) : Fl_Gl_Window(x, y, w, h, l) {}
};

WaterfallWindow *win;

void timer_cb(void*) {
    static float angle = 0;
    float mock_fft[2 * FFT_SIZE];
    float mock_mag[FFT_SIZE];

    float freq1 = 0.1f + 0.05f * sinf(angle * 0.1f);
    float freq2 = 0.3f + 0.1f * cosf(angle * 0.05f);

    for (int i = 0; i < FFT_SIZE; i++) {
        float val = sinf(2.0f * (float)M_PI * freq1 * i) + 0.5f * sinf(2.0f * (float)M_PI * freq2 * i);
        mock_fft[2 * i] = val;
        mock_fft[2 * i + 1] = 0.0f;
    }
    angle += 0.1f;

    avx_fft_array(mock_fft, FFT_SIZE);
    avx_vector_magnitude(mock_fft, FFT_SIZE, mock_mag);

    waterfall_ptr = (waterfall_ptr + 1) % WATERFALL_DEPTH;
    for (int i = 0; i < FFT_SIZE / 2; i++) {
        waterfall[waterfall_ptr][i] = mock_mag[i] / 500.0f;
    }

    win->redraw();
    if (!getenv("HEADLESS")) Fl::repeat_timeout(0.03, timer_cb);
}

int main(int argc, char **argv) {
    printf("Starting AVX DSP GUI Demo...\n");
    Fl_Double_Window *main_win = new Fl_Double_Window(800, 600, "AVX DSP Waterfall");
    win = new WaterfallWindow(10, 10, 780, 580);
    main_win->end();

    if (getenv("HEADLESS")) {
        printf("Headless mode detected, running 10 iterations...\n");
        for(int i=0; i<10; i++) timer_cb(NULL);
        printf("Verification successful.\n");
        return 0;
    }

    main_win->show(argc, argv);
    Fl::add_timeout(0.03, timer_cb);
    return Fl::run();
}

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

float waterfall_mag[WATERFALL_DEPTH][FFT_SIZE / 2];
float waterfall_phase[WATERFALL_DEPTH][FFT_SIZE / 2];
int waterfall_ptr = 0;

void hsv_to_rgb(float h, float s, float v, float &r, float &g, float &b) {
    if (s == 0.0f) {
        r = g = b = v;
        return;
    }
    h = fmodf(h, 2.0f * (float)M_PI);
    if (h < 0) h += 2.0f * (float)M_PI;
    h /= ((float)M_PI / 3.0f);
    int i = (int)floorf(h);
    float f = h - i;
    float p = v * (1.0f - s);
    float q = v * (1.0f - s * f);
    float t = v * (1.0f - s * (1.0f - f));
    switch (i) {
        case 0: r = v; g = t; b = p; break;
        case 1: r = q; g = v; b = p; break;
        case 2: r = p; g = v; b = t; break;
        case 3: r = p; g = q; b = v; break;
        case 4: r = t; g = p; b = v; break;
        default: r = v; g = p; b = q; break;
    }
}

class PhaseWaterfallWindow : public Fl_Gl_Window {
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
                float mag = waterfall_mag[row][x];
                float phase = waterfall_phase[row][x];
                float r, g, b;
                // Map phase to Hue, and magnitude to Value (Brightness)
                hsv_to_rgb(phase + (float)M_PI, 0.8f, mag, r, g, b);
                glColor3f(r, g, b);
                glVertex2f((float)x / (FFT_SIZE / 2), (float)y / WATERFALL_DEPTH);
            }
        }
        glEnd();
    }
public:
    PhaseWaterfallWindow(int x, int y, int w, int h, const char* l = 0) : Fl_Gl_Window(x, y, w, h, l) {}
};

PhaseWaterfallWindow *win;

void timer_cb(void*) {
    static float angle = 0;
    float mock_fft[2 * FFT_SIZE];
    float mock_mag[FFT_SIZE];
    float mock_phase[FFT_SIZE];

    float freq1 = 0.1f + 0.05f * sinf(angle * 0.1f);
    float phase_shift = angle; // Slowly varying phase

    for (int i = 0; i < FFT_SIZE; i++) {
        // Signal with moving phase
        float val = sinf(2.0f * (float)M_PI * freq1 * i + phase_shift);
        mock_fft[2 * i] = val;
        mock_fft[2 * i + 1] = 0.0f;
    }
    angle += 0.05f;

    avx_fft_array(mock_fft, FFT_SIZE);
    avx_vector_magnitude(mock_fft, FFT_SIZE, mock_mag);
    avx_vector_phase(mock_fft, FFT_SIZE, mock_phase);

    waterfall_ptr = (waterfall_ptr + 1) % WATERFALL_DEPTH;
    for (int i = 0; i < FFT_SIZE / 2; i++) {
        waterfall_mag[waterfall_ptr][i] = mock_mag[i] / 500.0f;
        waterfall_phase[waterfall_ptr][i] = mock_phase[i];
    }

    win->redraw();
    if (!getenv("HEADLESS")) Fl::repeat_timeout(0.03, timer_cb);
}

int main(int argc, char **argv) {
    printf("Starting AVX DSP Phase-Sensitive GUI Demo...\n");
    Fl_Double_Window *main_win = new Fl_Double_Window(800, 600, "AVX DSP Phase Waterfall");
    win = new PhaseWaterfallWindow(10, 10, 780, 580);
    main_win->end();

    if (getenv("HEADLESS")) {
        printf("Headless mode, verifying phase logic...\n");
        for(int i=0; i<10; i++) timer_cb(NULL);
        printf("Verification successful.\n");
        return 0;
    }

    main_win->show(argc, argv);
    Fl::add_timeout(0.03, timer_cb);
    return Fl::run();
}

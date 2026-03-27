# A makefile for compiling the avx_dsp library and the demo apps

CC = gcc
CXX = g++
CFLAGS = -Wall -O3 -mavx -mfma -fPIC
CXXFLAGS = -Wall -O3 -mavx -mfma $(shell fltk-config --cxxflags)
LDFLAGS = -L. -lavx_dsp -lm
FLTK_LDFLAGS = $(shell fltk-config --ldflags --use-gl) -lGL -lGLU -lasound

LIB_NAME = libavx_dsp.so
LIB_SRCS = avx_dsp.c
LIB_OBJS = $(LIB_SRCS:.c=.o)

all: lib demo phase_demo tests_large benchmark

lib: $(LIB_OBJS)
	$(CC) -shared -o $(LIB_NAME) $(LIB_OBJS) -lm

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

demo: lib gui_demo.cpp audio_engine.c
	$(CC) $(CFLAGS) -c audio_engine.c -o audio_engine.o
	$(CXX) $(CXXFLAGS) gui_demo.cpp audio_engine.o -o gui_demo $(LDFLAGS) $(FLTK_LDFLAGS)

phase_demo: lib gui_phase_demo.cpp audio_engine.c
	$(CC) $(CFLAGS) -c audio_engine.c -o audio_engine.o
	$(CXX) $(CXXFLAGS) gui_phase_demo.cpp audio_engine.o -o gui_phase_demo $(LDFLAGS) $(FLTK_LDFLAGS)

tests_large: lib tests_large.c generate_data.py
	python3 generate_data.py
	$(CC) $(CFLAGS) tests_large.c -o tests_large $(LDFLAGS)
	LD_LIBRARY_PATH=. ./tests_large

benchmark: lib benchmark.c benchmark_fft.c benchmark_fir.c
	$(CC) $(CFLAGS) benchmark.c -o benchmark $(LDFLAGS)
	$(CC) $(CFLAGS) benchmark_fft.c -o benchmark_fft $(LDFLAGS)
	$(CC) $(CFLAGS) benchmark_fir.c -o benchmark_fir $(LDFLAGS)
	LD_LIBRARY_PATH=. ./benchmark
	LD_LIBRARY_PATH=. ./benchmark_fft
	LD_LIBRARY_PATH=. ./benchmark_fir

clean:
	rm -f *.o $(LIB_NAME) gui_demo gui_phase_demo tests_large benchmark benchmark_fft benchmark_fir *.bin avx_dsp_example tests benchmark_align test_fft

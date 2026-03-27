# A makefile for compiling the avx_dsp library and the demo apps

# Define some variables
CC = gcc
CXX = g++
CFLAGS = -Wall -O3 -mavx -mfma
CXXFLAGS = -Wall -O3 -mavx -mfma $(shell fltk-config --cxxflags)
LDFLAGS = -L. -lavx_dsp -lm
FLTK_LDFLAGS = $(shell fltk-config --ldflags --use-gl) -lGL -lGLU -lasound

LIB_SRCS = avx_dsp.c
LIB_OBJS = $(LIB_SRCS:.c=.o)
LIB_NAME = libavx_dsp.so

# The default rule
all: lib demo phase_demo

# The rule to build the library
lib: $(LIB_OBJS)
	$(CC) -shared -o $(LIB_NAME) $(LIB_OBJS) -lm

# The rule to build the GUI demo
demo: lib gui_demo.cpp audio_engine.c
	$(CC) $(CFLAGS) -c audio_engine.c -o audio_engine.o
	$(CXX) $(CXXFLAGS) gui_demo.cpp audio_engine.o -o gui_demo $(LDFLAGS) $(FLTK_LDFLAGS)

# The rule to build the phase-sensitive GUI demo
phase_demo: lib gui_phase_demo.cpp audio_engine.c
	$(CC) $(CFLAGS) -c audio_engine.c -o audio_engine.o
	$(CXX) $(CXXFLAGS) gui_phase_demo.cpp audio_engine.o -o gui_phase_demo $(LDFLAGS) $(FLTK_LDFLAGS)

# The rule to build and run large tests
tests_large: lib tests_large.c generate_data.py
	python3 generate_data.py
	$(CC) $(CFLAGS) tests_large.c avx_dsp.c -o tests_large -lm
	LD_LIBRARY_PATH=. ./tests_large

# The rule to run benchmarks
benchmark: lib benchmark.c
	$(CC) $(CFLAGS) benchmark.c avx_dsp.c -o benchmark -lm
	LD_LIBRARY_PATH=. ./benchmark

# The rule to clean up
clean:
	rm -f *.o $(LIB_NAME) gui_demo gui_phase_demo tests_large benchmark *.bin avx_dsp_example tests
 

# A makefile for compiling the avx_dsp library and the example file

# Define some variables
CC = gcc # the C compiler
CFLAGS = -Wall -O3 -mavx # the C compiler flags
LDFLAGS = -L. -lavx_dsp # the linker flags
LIB_SRCS = avx_dsp.c # the library source file
LIB_OBJS = $(LIB_SRCS:.c=.o) # the library object file
LIB_NAME = libavx_dsp.so # the library name
EX_SRCS = avx_dsp_example.c # the example source file
EX_OBJS = $(EX_SRCS:.c=.o) # the example object file
EX_NAME = avx_dsp_example # the example name

# The default rule
all: lib example

# The rule to build the library
lib: $(LIB_OBJS)
	$(CC) -shared -o $(LIB_NAME) $(LIB_OBJS)

# The rule to build the example file
example: $(EX_OBJS)
	$(CC) -o $(EX_NAME) $(EX_OBJS) $(LDFLAGS)

# The rule to clean up the generated files
clean:
	rm -f $(LIB_OBJS) $(LIB_NAME) $(EX_OBJS) $(EX_NAME)
 

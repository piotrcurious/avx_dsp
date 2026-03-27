import math
import struct
import random

def generate_dot_product(size=1024):
    # Sine wave * Cosine wave
    a = [math.sin(2 * math.pi * 5 * i / size) for i in range(size)]
    b = [math.cos(2 * math.pi * 5 * i / size) for i in range(size)]
    expected = sum(x * y for x, y in zip(a, b))

    with open("dot_a.bin", "wb") as f:
        f.write(struct.pack(f"{size}f", *a))
    with open("dot_b.bin", "wb") as f:
        f.write(struct.pack(f"{size}f", *b))
    with open("dot_expected.bin", "wb") as f:
        f.write(struct.pack("f", expected))
    return size

def generate_convolution(size=1024, h_size=8):
    # Step function convolved with impulsive response
    x = [1.0 if i < size // 2 else 0.0 for i in range(size)]
    h = [1.0 / h_size] * h_size # Moving average filter

    res = [0.0] * (size + h_size - 1)
    for i in range(size):
        for j in range(h_size):
            res[i+j] += x[i] * h[j]

    with open("conv_x.bin", "wb") as f:
        f.write(struct.pack(f"{size}f", *x))
    with open("conv_h.bin", "wb") as f:
        f.write(struct.pack(f"{h_size}f", *h))
    with open("conv_expected.bin", "wb") as f:
        f.write(struct.pack(f"{len(res)}f", *res))
    return size, h_size, len(res)

def generate_fft(size=1024):
    # Sum of two sine waves
    x_64 = []
    for i in range(size):
        val = math.sin(2 * math.pi * 2 * i / size) + 0.5 * math.sin(2 * math.pi * 10 * i / size)
        x_64.extend([val, 0.0]) # Complex: real=val, imag=0.0

    def dft(x_comp):
        N = len(x_comp)
        X = []
        for k in range(N):
            re, im = 0.0, 0.0
            for n in range(N):
                angle = -2.0 * math.pi * k * n / N
                c, s = math.cos(angle), math.sin(angle)
                re += x_comp[n][0] * c - x_comp[n][1] * s
                im += x_comp[n][0] * s + x_comp[n][1] * c
            X.append((re, im))
        return X

    fft_size = size
    x_comp = [(x_64[2*i], x_64[2*i+1]) for i in range(fft_size)]
    X_comp = dft(x_comp)
    X = []
    for re, im in X_comp:
        X.extend([re, im])

    with open("fft_x.bin", "wb") as f:
        f.write(struct.pack(f"{2*fft_size}f", *x_64))
    with open("fft_expected.bin", "wb") as f:
        f.write(struct.pack(f"{2*fft_size}f", *X))
    return fft_size

if __name__ == "__main__":
    print(f"Dot product size: {generate_dot_product()}")
    print(f"Convolution sizes: {generate_convolution()}")
    print(f"FFT size: {generate_fft(128)}") # 128 is faster for DFT reference

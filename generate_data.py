import math
import struct
import random

def generate_dot_product(size=1024):
    a = [random.uniform(-1, 1) for _ in range(size)]
    b = [random.uniform(-1, 1) for _ in range(size)]
    expected = sum(x * y for x, y in zip(a, b))

    with open("dot_a.bin", "wb") as f:
        f.write(struct.pack(f"{size}f", *a))
    with open("dot_b.bin", "wb") as f:
        f.write(struct.pack(f"{size}f", *b))
    with open("dot_expected.bin", "wb") as f:
        f.write(struct.pack("f", expected))
    return size

def generate_convolution(size=1024, h_size=8):
    x = [random.uniform(-1, 1) for _ in range(size)]
    h = [random.uniform(-1, 1) for _ in range(h_size)]

    # Linear convolution
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
    # N-point complex FFT
    # Input is 2*size floats (real, imag)
    x = [random.uniform(-1, 1) for _ in range(2 * size)]

    # Simple DFT for expected result (since no numpy)
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

    # For large sizes, DFT is slow. Let's use 64 for testing FFT.
    fft_size = 64
    x_64 = [random.uniform(-1, 1) for _ in range(2 * fft_size)]
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
    print(f"FFT size: {generate_fft()}")

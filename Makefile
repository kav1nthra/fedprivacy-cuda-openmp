# Optional: build the kernel libraries without the notebook (the notebook does this itself).
CC ?= gcc
all: cpu cuda
cpu:
	mkdir -p src/build && $(CC) -O3 -march=native -ffast-math -shared -fPIC -o src/build/libfl_cpu.so src/fl_cpu.c -lm
cuda:
	mkdir -p src/build && nvcc -O3 -arch=native -shared -Xcompiler -fPIC -o src/build/libfl_cuda.so src/fl_cuda.cu
clean:
	rm -rf src/build

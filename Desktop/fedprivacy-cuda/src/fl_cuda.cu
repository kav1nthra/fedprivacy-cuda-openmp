/*
 * fl_cuda.cu -- CUDA versions of the two federated-learning kernels in fl_cpu.c
 *
 *   dp_grad  variant 0 : NAIVE      thread-per-row dot product (uncoalesced loads),
 *                                   thread-per-column accumulation (only D threads)
 *   dp_grad  variant 1 : OPTIMISED  warp-per-row dot product (coalesced + shuffle reduce),
 *                                   2-D tiled accumulation with shared memory + atomicAdd
 *   secagg                          thread-per-parameter masking + modular sum
 *
 * Build:  nvcc -O3 -arch=native -shared -Xcompiler -fPIC -o libfl_cuda.so fl_cuda.cu
 *
 * Timing: every entry point reports kernel-only time (cudaEvent) so the Python side
 * can separate "GPU compute" from "PCIe transfer" cost.
 */
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(e_), __FILE__, __LINE__); return -1; } } while (0)

/* ----------------------------------------------------------------- helpers */
__device__ __forceinline__ float sample_coef(float z, float y, float xnorm, float C) {
    float p  = 1.0f / (1.0f + expf(-z));
    float r  = p - y;
    float gn = fabsf(r) * xnorm;
    float sc = (gn > C) ? (C / gn) : 1.0f;
    return r * sc;
}

/* ------------------------------------------------- DP gradient: NAIVE kernels */
__global__ void coef_naive(const float *X, const float *y, const float *xn, const float *w,
                           int64_t N, int64_t D, float C, float *s) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const float *x = X + i * D;                 /* neighbouring threads are D floats apart -> uncoalesced */
    float z = 0.f;
    for (int64_t j = 0; j < D; j++) z += x[j] * w[j];
    s[i] = sample_coef(z, y[i], xn[i], C);
}

__global__ void accum_naive(const float *X, const float *s, int64_t N, int64_t D, float *g) {
    int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= D) return;                         /* only D threads, each walks all N rows */
    float a = 0.f;
    for (int64_t i = 0; i < N; i++) a += s[i] * X[i * D + j];
    g[j] = a;
}

/* ------------------------------------------- DP gradient: OPTIMISED kernels */
__global__ void coef_warp(const float *X, const float *y, const float *xn, const float *w,
                          int64_t N, int64_t D, float C, float *s) {
    int64_t row = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;   /* one warp per row */
    int lane = threadIdx.x & 31;
    if (row >= N) return;                                                   /* whole warp exits together */
    const float *x = X + row * D;
    float z = 0.f;
    for (int64_t j = lane; j < D; j += 32) z += x[j] * __ldg(&w[j]);        /* coalesced */
    for (int o = 16; o > 0; o >>= 1) z += __shfl_down_sync(0xffffffffu, z, o);
    if (lane == 0) s[row] = sample_coef(z, y[row], xn[row], C);
}

#define TILE 256
#define ROWS_PER_CHUNK 512
__global__ void accum_tiled(const float *X, const float *s, int64_t N, int64_t D, float *g) {
    __shared__ float ss[TILE];
    int64_t j  = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;            /* column */
    int64_t r0 = (int64_t)blockIdx.y * ROWS_PER_CHUNK;                       /* row chunk */
    int64_t r1 = (r0 + ROWS_PER_CHUNK < N) ? r0 + ROWS_PER_CHUNK : N;
    float a = 0.f;
    for (int64_t t = r0; t < r1; t += TILE) {
        int64_t idx = t + threadIdx.x;
        ss[threadIdx.x] = (idx < r1) ? s[idx] : 0.f;                        /* stage s[] in shared memory */
        __syncthreads();
        int64_t lim = (r1 - t < TILE) ? (r1 - t) : TILE;
        if (j < D) for (int k = 0; k < lim; k++) a += ss[k] * X[(t + k) * D + j];   /* coalesced over j */
        __syncthreads();
    }
    if (j < D) atomicAdd(&g[j], a);
}

/* --------------------------------------------------------- secure aggregation */
__device__ __forceinline__ uint32_t mix32(uint32_t h) {
    h ^= h >> 16; h *= 0x85ebca6bU; h ^= h >> 13; h *= 0xc2b2ae35U; h ^= h >> 16;
    return h;
}
__device__ __forceinline__ uint32_t mask_val(uint32_t seed, uint32_t k, uint32_t j) {
    return mix32(mix32(seed + k * 0x9E3779B1U) ^ (j * 0x85EBCA6BU + 0x27d4eb2fU));
}
__device__ __forceinline__ uint32_t quant(float u, float scale) {
    return (uint32_t)(int32_t)__float2int_rn(u * scale);
}
__global__ void secagg_kernel(const float *U, int K, int64_t D, uint32_t seed, float scale, float *out) {
    int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= D) return;
    uint32_t acc = 0, rs = 0;
    for (int k = 0; k < K - 1; k++) {
        uint32_t r = mask_val(seed, (uint32_t)k, (uint32_t)j);
        acc += quant(U[(int64_t)k * D + j], scale) + r;                    /* coalesced over j */
        rs  += r;
    }
    acc += quant(U[(int64_t)(K - 1) * D + j], scale) + (0u - rs);
    out[j] = (float)((int32_t)acc) / scale;
}

/* ================================================================ C interface */
static float *dX = 0, *dy = 0, *dxn = 0, *dw = 0, *dg = 0, *ds = 0;
static int64_t gN = 0, gD = 0;

extern "C" {

/* returns number of CUDA devices (0 if none / driver problem); fills name and memory */
int gpu_info(char *name, int n, double *mem_gb) {
    int cnt = 0;
    if (cudaGetDeviceCount(&cnt) != cudaSuccess || cnt == 0) return 0;
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return 0;
    strncpy(name, p.name, n - 1); name[n - 1] = 0;
    *mem_gb = (double)p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0);
    cudaFree(0);                                    /* force context creation now (not inside a timed region) */
    return cnt;
}

void gpu_free(void) {
    cudaFree(dX); cudaFree(dy); cudaFree(dxn); cudaFree(dw); cudaFree(dg); cudaFree(ds);
    dX = dy = dxn = dw = dg = ds = 0; gN = gD = 0;
}

/* one-time upload of a (client's) data matrix; stays resident on the GPU */
int gpu_upload(const float *X, const float *y, const float *xn, int64_t N, int64_t D) {
    gpu_free();
    CK(cudaMalloc(&dX,  (size_t)N * D * sizeof(float)));
    CK(cudaMalloc(&dy,  (size_t)N * sizeof(float)));
    CK(cudaMalloc(&dxn, (size_t)N * sizeof(float)));
    CK(cudaMalloc(&ds,  (size_t)N * sizeof(float)));
    CK(cudaMalloc(&dw,  (size_t)D * sizeof(float)));
    CK(cudaMalloc(&dg,  (size_t)D * sizeof(float)));
    CK(cudaMemcpy(dX,  X,  (size_t)N * D * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dy,  y,  (size_t)N * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dxn, xn, (size_t)N * sizeof(float), cudaMemcpyHostToDevice));
    gN = N; gD = D;
    return 0;
}

/* one DP-gradient evaluation. variant 0 = naive, 1 = optimised.
 * w is copied H2D and g copied D2H inside (small: D floats); kernel_ms = kernels only. */
int gpu_dp_grad(const float *w, float C, int variant, float *g_out, float *kernel_ms) {
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaMemcpy(dw, w, (size_t)gD * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaEventRecord(e0));
    if (variant == 0) {
        coef_naive<<<(unsigned)((gN + 127) / 128), 128>>>(dX, dy, dxn, dw, gN, gD, C, ds);
        accum_naive<<<(unsigned)((gD + 127) / 128), 128>>>(dX, ds, gN, gD, dg);
    } else {
        coef_warp<<<(unsigned)((gN + 7) / 8), 256>>>(dX, dy, dxn, dw, gN, gD, C, ds);
        CK(cudaMemsetAsync(dg, 0, (size_t)gD * sizeof(float)));
        dim3 grid((unsigned)((gD + TILE - 1) / TILE), (unsigned)((gN + ROWS_PER_CHUNK - 1) / ROWS_PER_CHUNK));
        accum_tiled<<<grid, TILE>>>(dX, ds, gN, gD, dg);
    }
    CK(cudaGetLastError());
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    CK(cudaEventElapsedTime(kernel_ms, e0, e1));
    CK(cudaMemcpy(g_out, dg, (size_t)gD * sizeof(float), cudaMemcpyDeviceToHost));
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return 0;
}

/* secure aggregation of K client updates (K x D). total_ms includes H2D + kernel + D2H. */
int gpu_secagg(const float *U, int K, int64_t D, uint32_t seed, float scale, float *out,
               float *kernel_ms, float *total_ms) {
    float *dU = 0, *dout = 0;
    cudaEvent_t t0, k0, k1, t1;
    CK(cudaMalloc(&dU,   (size_t)K * D * sizeof(float)));
    CK(cudaMalloc(&dout, (size_t)D * sizeof(float)));
    CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&k0)); CK(cudaEventCreate(&k1)); CK(cudaEventCreate(&t1));
    CK(cudaEventRecord(t0));
    CK(cudaMemcpy(dU, U, (size_t)K * D * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaEventRecord(k0));
    secagg_kernel<<<(unsigned)((D + 255) / 256), 256>>>(dU, K, D, seed, scale, dout);
    CK(cudaGetLastError());
    CK(cudaEventRecord(k1));
    CK(cudaMemcpy(out, dout, (size_t)D * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaEventRecord(t1));
    CK(cudaEventSynchronize(t1));
    CK(cudaEventElapsedTime(kernel_ms, k0, k1));
    CK(cudaEventElapsedTime(total_ms, t0, t1));
    cudaEventDestroy(t0); cudaEventDestroy(k0); cudaEventDestroy(k1); cudaEventDestroy(t1);
    cudaFree(dU); cudaFree(dout);
    return 0;
}

} /* extern "C" */

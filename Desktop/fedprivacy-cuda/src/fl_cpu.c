/*
 * fl_cpu.c  --  CPU kernels for privacy-preserving federated learning
 *               (serial baseline, single-threaded)
 *
 * Kernel 1  dp_grad_*  : per-sample gradient of logistic regression, L2 clipping
 *                        (the core of DP-SGD), summed over the batch.
 *                        Pass A: s_i = clipped residual coefficient of sample i
 *                        Pass B: g   = sum_i s_i * x_i
 * Kernel 2  secagg_*   : secure aggregation by additive masking. Every client's
 *                        (quantised) update is masked with a pseudo-random mask,
 *                        masks of all clients cancel in the server-side sum mod 2^32.
 *                        NOTE: the PRG here is a fast hash for benchmarking only,
 *                        NOT cryptographically secure.
 *
 * Build:  gcc -O3 -march=native -ffast-math -shared -fPIC -o libfl_cpu.so fl_cpu.c -lm
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ---------------------------------------------------------------- DP gradient */
static inline float dot_row(const float *restrict x, const float *restrict w, int64_t D) {
    float z = 0.f;
    for (int64_t j = 0; j < D; j++) z += x[j] * w[j];
    return z;
}

/* clipped residual:  r = sigmoid(z) - y ;  per-sample grad = r * x, norm = |r|*||x|| */
static inline float sample_coef(float z, float y, float xnorm, float C) {
    float p  = 1.0f / (1.0f + expf(-z));
    float r  = p - y;
    float gn = fabsf(r) * xnorm;
    float sc = (gn > C) ? (C / gn) : 1.0f;
    return r * sc;
}

/* X: N x D row-major, y: N, w: D, xn: N (row L2 norms), s: N scratch, g: D output (sum of clipped grads) */
void dp_grad_serial(const float *X, const float *y, const float *w, const float *xn,
                    int64_t N, int64_t D, float C, float *g, float *s) {
    for (int64_t i = 0; i < N; i++) s[i] = sample_coef(dot_row(X + i * D, w, D), y[i], xn[i], C);
    memset(g, 0, (size_t)D * sizeof(float));
    for (int64_t i = 0; i < N; i++) {
        const float si = s[i];
        const float *x = X + i * D;
        for (int64_t j = 0; j < D; j++) g[j] += si * x[j];
    }
}

/* ------------------------------------------------------ secure aggregation */
static inline uint32_t mix32(uint32_t h) {            /* murmur3 finaliser */
    h ^= h >> 16; h *= 0x85ebca6bU; h ^= h >> 13; h *= 0xc2b2ae35U; h ^= h >> 16;
    return h;
}
static inline uint32_t mask_val(uint32_t seed, uint32_t k, uint32_t j) {
    return mix32(mix32(seed + k * 0x9E3779B1U) ^ (j * 0x85EBCA6BU + 0x27d4eb2fU));
}
static inline uint32_t quant(float u, float scale) {  /* fixed point, two's complement */
    return (uint32_t)(int32_t)lrintf(u * scale);
}

/* U: K x D (client updates, already weighted). out[j] = sum_k U[k][j]  computed via masked shares.
 * Clients 0..K-2 add mask r_k; the last client adds -(sum of the others) so masks cancel mod 2^32. */
#define BLK 512
static void secagg_range(const float *U, int K, int64_t D, uint32_t seed, float scale,
                         int64_t j0, int64_t j1, float *out) {
    uint32_t acc[BLK], rs[BLK];
    int64_t n = j1 - j0;
    for (int64_t t = 0; t < n; t++) { acc[t] = 0; rs[t] = 0; }
    for (int k = 0; k < K - 1; k++) {
        const float *u = U + (int64_t)k * D + j0;
        for (int64_t t = 0; t < n; t++) {
            uint32_t r = mask_val(seed, (uint32_t)k, (uint32_t)(j0 + t));
            acc[t] += quant(u[t], scale) + r;       /* masked share seen by server */
            rs[t]  += r;
        }
    }
    const float *u = U + (int64_t)(K - 1) * D + j0;
    for (int64_t t = 0; t < n; t++) {
        acc[t] += quant(u[t], scale) + (0u - rs[t]);
        out[j0 + t] = (float)((int32_t)acc[t]) / scale;
    }
}

void secagg_serial(const float *U, int K, int64_t D, uint32_t seed, float scale, float *out) {
    for (int64_t j0 = 0; j0 < D; j0 += BLK) secagg_range(U, K, D, seed, scale, j0, (j0 + BLK < D) ? j0 + BLK : D, out);
}

/* masked share of a single client (used only to *show* that shares look random) */
void secagg_share(const float *U, int K, int64_t D, uint32_t seed, float scale, int client, uint32_t *share) {
    for (int64_t j = 0; j < D; j++) {
        uint32_t q = quant(U[(int64_t)client * D + j], scale);
        if (client < K - 1) share[j] = q + mask_val(seed, (uint32_t)client, (uint32_t)j);
        else {
            uint32_t rs = 0;
            for (int k = 0; k < K - 1; k++) rs += mask_val(seed, (uint32_t)k, (uint32_t)j);
            share[j] = q + (0u - rs);
        }
    }
}

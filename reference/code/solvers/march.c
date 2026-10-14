/* Block forward-marching solver for the local part K of the forward generator.
 *
 * Unknowns are ordered a-major: x[j*n + i], j = a-cell (0..Na-1), i = u-cell (0..n-1).
 * K is block lower bidiagonal in j:
 *     K_jj     = T_j  (tridiagonal in u: lo / dg[j] / up),
 *     K_j,j-1  = diag(ain[j])   (upwind transport inflow from a-cell j-1).
 * Solving K x = y is a forward march in j with one tridiagonal (Thomas) solve per block.
 * T_j - z I is column diagonally dominant for Re z >= 0 (sub-generator with killing),
 * so Thomas without pivoting is stable there.
 *
 * Build: gcc -O3 -march=native -shared -fPIC march.c -o march.so
 */
#include <complex.h>
#include <stddef.h>

#define DEFINE_KERNELS(T, PFX)                                                             \
void PFX##_factor(int Na, int n, const double* lo, const double* up, const T* dg,          \
                  T* mlt, T* winv) {                                                       \
    for (int j = 0; j < Na; j++) {                                                         \
        const T* d = dg + (size_t)j * n; T* m = mlt + (size_t)j * n;                       \
        T* wi = winv + (size_t)j * n;                                                      \
        T w = d[0]; wi[0] = 1.0 / w; m[0] = 0;                                             \
        for (int i = 1; i < n; i++) {                                                      \
            m[i] = lo[i] * wi[i - 1];                                                      \
            w = d[i] - m[i] * up[i - 1];                                                   \
            wi[i] = 1.0 / w;                                                               \
        }                                                                                  \
    }                                                                                      \
}                                                                                          \
void PFX##_march(int Na, int n, const double* up, const T* mlt, const T* winv,             \
                 const double* ain, const T* y, T* x) {                                    \
    for (int j = 0; j < Na; j++) {                                                         \
        const T* m = mlt + (size_t)j * n; const T* wi = winv + (size_t)j * n;              \
        const T* yj = y + (size_t)j * n; T* xj = x + (size_t)j * n;                        \
        if (j == 0) {                                                                      \
            xj[0] = yj[0];                                                                 \
            for (int i = 1; i < n; i++) xj[i] = yj[i] - m[i] * xj[i - 1];                  \
        } else {                                                                           \
            const T* xp = x + (size_t)(j - 1) * n; const double* a = ain + (size_t)j * n;  \
            xj[0] = yj[0] - a[0] * xp[0];                                                  \
            for (int i = 1; i < n; i++) xj[i] = (yj[i] - a[i] * xp[i]) - m[i] * xj[i - 1]; \
        }                                                                                  \
        xj[n - 1] *= wi[n - 1];                                                            \
        for (int i = n - 2; i >= 0; i--) xj[i] = (xj[i] - up[i] * xj[i + 1]) * wi[i];     \
    }                                                                                      \
}                                                                                          \
/* multi-RHS march for unit sources in block 0 at cells tgt[k]; returns                    \
   S[k*n + i] = sum_j H[j*n+i] * x_k[j*n+i]  (hazard-weighted a-integral of the response) */\
void PFX##_capmarch(int Na, int n, const double* up, const T* mlt, const T* winv,          \
                    const double* ain, const double* H, int nT, const int* tgt,            \
                    T* work, T* S) {                                                       \
    for (int k = 0; k < nT; k++) {                                                         \
        T* x = work;                                                                       \
        for (int j = 0; j < Na; j++) {                                                     \
            const T* m = mlt + (size_t)j * n; const T* wi = winv + (size_t)j * n;          \
            T* xj = x + (size_t)j * n;                                                     \
            if (j == 0) {                                                                  \
                for (int i = 0; i < n; i++) xj[i] = 0;                                     \
                xj[tgt[k]] = 1.0;                                                          \
                for (int i = 1; i < n; i++) xj[i] = xj[i] - m[i] * xj[i - 1];              \
            } else {                                                                       \
                const T* xp = x + (size_t)(j - 1) * n; const double* a = ain + (size_t)j * n; \
                xj[0] = -a[0] * xp[0];                                                     \
                for (int i = 1; i < n; i++) xj[i] = (-a[i] * xp[i]) - m[i] * xj[i - 1];    \
            }                                                                              \
            xj[n - 1] *= wi[n - 1];                                                        \
            for (int i = n - 2; i >= 0; i--) xj[i] = (xj[i] - up[i] * xj[i + 1]) * wi[i]; \
        }                                                                                  \
        T* Sk = S + (size_t)k * n;                                                         \
        for (int i = 0; i < n; i++) Sk[i] = 0;                                             \
        for (int j = 0; j < Na; j++) {                                                     \
            const double* h = H + (size_t)j * n; const T* xj = x + (size_t)j * n;          \
            for (int i = 0; i < n; i++) Sk[i] += h[i] * xj[i];                             \
        }                                                                                  \
    }                                                                                      \
}

DEFINE_KERNELS(double, d)
DEFINE_KERNELS(double complex, z)

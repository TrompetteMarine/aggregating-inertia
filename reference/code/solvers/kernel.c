/* c9 kernel: implicit sweep for block lower-triangular (in the clock variable b) systems whose
   diagonal blocks are tridiagonal (in u).  Used for
     - one backward-Euler step of the episode PDE of economy B (Nb slices),
     - the time-integrated (resolvent) episode solve of economy B (smat = 0),
     - one backward-Euler step of economy P (Nb = 1, R = 0, H = hbar(t_{n+1}, .)).

   Unknowns are cell MASSES X[j*Nu + i] (slice j in b, cell i in u).  For every slice j we solve
     (smat + kd[i] + R[j,i] + H[j,i]) X[j,i] - kl[i] X[j,i-1] - ku[i] X[j,i+1]
         = srhs*OLD[j,i] + SRC[j,i] + R[j-1,i] X[j-1,i]
   where K = tridiag(-kl, kd, -ku) is minus the finite-volume drift-diffusion generator (mass form,
   zero-flux ends), R[j,i] is the upwind outflow rate from b-slice j to j+1 and H the hazard.
   Up to three coupled fields share the matrix:
     field 0 : q        SRC = 0
     field 1 : m1       SRC = w1[i] * q[j,i]      (w1 = u)       first I-moment density
     field 2 : m2       SRC = w2[i] * m1[j,i]     (w2 = 2u)      second I-moment density
   OLD may alias the output (in-place update).  NULL OLD pointers are treated as zero.
   Slice sums are returned: p = sum_j q, c = sum_j H q, pm1 = sum_j m1, cm1 = sum_j H m1, pm2, cm2.
   The matrix is a nonsingular M-matrix, so Gaussian elimination without pivoting is stable and
   all quantities stay nonnegative for nonnegative data (positive OLD, w1 of either sign only
   enters m1, m2 linearly).  FTZ/DAZ is set during the call to avoid denormal slow-downs. */
#include <stddef.h>
#include <xmmintrin.h>

int sweep(int Nb, int Nu, double smat, double srhs,
          const double *kl, const double *kd, const double *ku,
          const double *R, const double *H, const double *w1, const double *w2,
          const double *q_old, const double *m1_old, const double *m2_old,
          double *q, double *m1, double *m2, int nf,
          double *p, double *c, double *pm1, double *cm1, double *pm2, double *cm2,
          double *work)
{
    unsigned int csr = _mm_getcsr();
    _mm_setcsr(csr | 0x8040);
    double *cp = work, *im = work + Nu, *d = work + 2 * Nu;
    int i, j;
    for (i = 0; i < Nu; i++) {
        p[i] = 0.0; c[i] = 0.0;
        if (nf > 1) { pm1[i] = 0.0; cm1[i] = 0.0; }
        if (nf > 2) { pm2[i] = 0.0; cm2[i] = 0.0; }
    }
    for (j = 0; j < Nb; j++) {
        const double *Rj = R + (size_t)j * Nu, *Hj = H + (size_t)j * Nu;
        const double *Rjm = (j > 0) ? R + (size_t)(j - 1) * Nu : NULL;
        size_t off = (size_t)j * Nu, offm = (j > 0) ? (size_t)(j - 1) * Nu : 0;
        /* factorization of the tridiagonal block of slice j */
        {
            double m = smat + kd[0] + Rj[0] + Hj[0];
            im[0] = 1.0 / m;
            cp[0] = -ku[0] * im[0];
            for (i = 1; i < Nu; i++) {
                m = smat + kd[i] + Rj[i] + Hj[i] + kl[i] * cp[i - 1];
                im[i] = 1.0 / m;
                cp[i] = -ku[i] * im[i];
            }
        }
        /* field 0 */
        {
            const double *o = q_old ? q_old + off : NULL;
            double *x = q + off;
            for (i = 0; i < Nu; i++) {
                double r = o ? srhs * o[i] : 0.0;
                if (j > 0) r += Rjm[i] * q[offm + i];
                d[i] = r;
            }
            d[0] = d[0] * im[0];
            for (i = 1; i < Nu; i++) d[i] = (d[i] + kl[i] * d[i - 1]) * im[i];
            x[Nu - 1] = d[Nu - 1];
            for (i = Nu - 2; i >= 0; i--) x[i] = d[i] - cp[i] * x[i + 1];
            for (i = 0; i < Nu; i++) { p[i] += x[i]; c[i] += Hj[i] * x[i]; }
        }
        if (nf > 1) {
            const double *o = m1_old ? m1_old + off : NULL;
            double *x = m1 + off;
            const double *xq = q + off;
            for (i = 0; i < Nu; i++) {
                double r = o ? srhs * o[i] : 0.0;
                r += w1[i] * xq[i];
                if (j > 0) r += Rjm[i] * m1[offm + i];
                d[i] = r;
            }
            d[0] = d[0] * im[0];
            for (i = 1; i < Nu; i++) d[i] = (d[i] + kl[i] * d[i - 1]) * im[i];
            x[Nu - 1] = d[Nu - 1];
            for (i = Nu - 2; i >= 0; i--) x[i] = d[i] - cp[i] * x[i + 1];
            for (i = 0; i < Nu; i++) { pm1[i] += x[i]; cm1[i] += Hj[i] * x[i]; }
        }
        if (nf > 2) {
            const double *o = m2_old ? m2_old + off : NULL;
            double *x = m2 + off;
            const double *xm = m1 + off;
            for (i = 0; i < Nu; i++) {
                double r = o ? srhs * o[i] : 0.0;
                r += w2[i] * xm[i];
                if (j > 0) r += Rjm[i] * m2[offm + i];
                d[i] = r;
            }
            d[0] = d[0] * im[0];
            for (i = 1; i < Nu; i++) d[i] = (d[i] + kl[i] * d[i - 1]) * im[i];
            x[Nu - 1] = d[Nu - 1];
            for (i = Nu - 2; i >= 0; i--) x[i] = d[i] - cp[i] * x[i + 1];
            for (i = 0; i < Nu; i++) { pm2[i] += x[i]; cm2[i] += Hj[i] * x[i]; }
        }
    }
    _mm_setcsr(csr);
    return 0;
}

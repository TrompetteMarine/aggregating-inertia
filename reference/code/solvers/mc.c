/* c9 Monte Carlo of economy B (true clock A) and economy P (tabulated hbar) with common random numbers.
   One episode per path, started at (U, A) = (0, 0), push f.  U is advanced with the exact OU transition;
   A and I = int U dt by the trapezoid rule; the killing is integrated out along each path
   (Rao-Blackwellisation): with survival weights S(t) = exp(-int_0^t hazard), per path
       tau   = int S dt,      U2 = int S U^2 dt,     IU = int h S I U dt,    I2 = int h S I^2 dt,
       I1    = int S U dt     (so that E[I_tau] = E[I1])
   are accumulated by the trapezoid rule for B (hazard H(U,A)) and P (hazard hbar(t,U), bilinear in (t,u)).
   Histograms (per batch): duration density (h S dt by t-bin), revision-size law (h S dt by U-bin),
   occupation (S dt by U-bin), for B and P. */
#include <stdint.h>
#include <math.h>
#include <stddef.h>

static inline uint64_t rotl(const uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
typedef struct { uint64_t s[4]; int has; double cache; } rng_t;
static uint64_t splitmix64(uint64_t *x) {
    uint64_t z = (*x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL; z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
static inline uint64_t next(rng_t *r) {
    uint64_t *s = r->s; const uint64_t result = rotl(s[1] * 5, 7) * 9; const uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 45);
    return result;
}
static inline double unif(rng_t *r) { return ((next(r) >> 11) + 0.5) * 0x1.0p-53; }
static inline double gauss(rng_t *r) {
    if (r->has) { r->has = 0; return r->cache; }
    double u1 = unif(r), u2 = unif(r);
    double rad = sqrt(-2.0 * log(u1)), ang = 6.283185307179586476925 * u2;
    r->cache = rad * sin(ang); r->has = 1;
    return rad * cos(ang);
}
static inline double smoothact(double u, double theta) {
    if (theta <= 0) return 1.0;
    double z = (fabs(u) - theta) / theta;
    if (z <= 0) return 0.0;
    if (z >= 1) return 1.0;
    double a = exp(-1.0 / z), b = exp(-1.0 / (1.0 - z));
    return a / (a + b);
}
static inline double hbar_at(const double *hb, int Nu, double u0, double du, int row, double w, double u) {
    double x = (u - u0) / du;   /* u0 = centre of cell 0 */
    int i = (int)floor(x);
    double a;
    if (i < 0) { i = 0; a = 0.0; } else if (i >= Nu - 1) { i = Nu - 2; a = 1.0; } else a = x - i;
    const double *r0 = hb + (size_t)row * Nu, *r1 = r0 + Nu;
    double h0 = (1 - a) * r0[i] + a * r0[i + 1];
    double h1 = (1 - a) * r1[i] + a * r1[i + 1];
    return (1 - w) * h0 + w * h1;
}

int mc(long npaths, uint64_t seed, double lam, double sig, double f, double theta, double eta, double nu,
       double gam, double dt, int kmax, double wtol,
       const double *hb, int Nrows, int Nu, double u0, double du,
       const int *krow, const double *kw,                 /* per MC step k: table row and weight in time */
       double *out,                                       /* npaths x 10 per-path results */
       int nb_t, double tbin, int nb_u, double ulo, double ubin, int nbatch,
       double *histB_dur, double *histP_dur, double *histB_rev, double *histP_rev,
       double *histB_occ, double *histP_occ)              /* nbatch x nb_* */
{
    double ea = exp(-lam * dt);
    double sd = (lam > 0) ? sig * sqrt(-expm1(-2.0 * lam * dt) / (2.0 * lam)) : sig * sqrt(dt);
    double mf = (lam > 0) ? f * (-expm1(-lam * dt)) / lam : f * dt;
    long n;
    for (n = 0; n < npaths; n++) {
        rng_t r; uint64_t sm = seed ^ (0x9E3779B97F4A7C15ULL * (uint64_t)(n + 1));
        r.s[0] = splitmix64(&sm); r.s[1] = splitmix64(&sm); r.s[2] = splitmix64(&sm); r.s[3] = splitmix64(&sm);
        r.has = 0;
        int bt = (int)((n * (long)nbatch) / npaths);
        double U = 0.0, A = 0.0, I = 0.0, al = smoothact(0.0, theta);
        double SB = 1.0, SP = 1.0;
        double HB = nu + eta * al * (-expm1(-gam * A));
        double HP = hbar_at(hb, Nu, u0, du, krow[0], kw[0], U);
        double tauB = 0, tauP = 0, U2B = 0, U2P = 0, IUB = 0, IUP = 0, I2B = 0, I2P = 0, I1B = 0, I1P = 0;
        int k;
        for (k = 0; k < kmax; k++) {
            double Un = U * ea + mf + sd * gauss(&r);
            double aln = smoothact(Un, theta);
            double An = A + 0.5 * (al + aln) * dt;
            double In = I + 0.5 * (U + Un) * dt;
            double HBn = nu + eta * aln * (-expm1(-gam * An));
            double HPn = hbar_at(hb, Nu, u0, du, krow[k + 1], kw[k + 1], Un);
            double SBn = SB * exp(-0.5 * (HB + HBn) * dt);
            double SPn = SP * exp(-0.5 * (HP + HPn) * dt);
            double h = 0.5 * dt;
            tauB += h * (SB + SBn);                tauP += h * (SP + SPn);
            U2B += h * (SB * U * U + SBn * Un * Un); U2P += h * (SP * U * U + SPn * Un * Un);
            I1B += h * (SB * U + SBn * Un);        I1P += h * (SP * U + SPn * Un);
            IUB += h * (HB * SB * I * U + HBn * SBn * In * Un);
            IUP += h * (HP * SP * I * U + HPn * SPn * In * Un);
            I2B += h * (HB * SB * I * I + HBn * SBn * In * In);
            I2P += h * (HP * SP * I * I + HPn * SPn * In * In);
            /* histograms: right-point rule */
            double tn = (k + 1) * dt;
            int it = (int)(tn / tbin);
            int iu = (int)floor((Un - ulo) / ubin);
            if (it < nb_t) {
                histB_dur[(size_t)bt * nb_t + it] += HBn * SBn * dt;
                histP_dur[(size_t)bt * nb_t + it] += HPn * SPn * dt;
            }
            if (iu >= 0 && iu < nb_u) {
                histB_rev[(size_t)bt * nb_u + iu] += HBn * SBn * dt;
                histP_rev[(size_t)bt * nb_u + iu] += HPn * SPn * dt;
                histB_occ[(size_t)bt * nb_u + iu] += SBn * dt;
                histP_occ[(size_t)bt * nb_u + iu] += SPn * dt;
            }
            U = Un; A = An; I = In; al = aln; HB = HBn; HP = HPn; SB = SBn; SP = SPn;
            if (SB < wtol && SP < wtol) break;
        }
        double *o = out + (size_t)n * 10;
        o[0] = tauB; o[1] = tauP; o[2] = U2B; o[3] = U2P; o[4] = IUB; o[5] = IUP; o[6] = I2B; o[7] = I2P;
        o[8] = I1B; o[9] = I1P;
    }
    return 0;
}

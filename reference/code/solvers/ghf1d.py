"""One-dimensional stationary solver for Markov completion rules H = Lam(u) with full resets.

Population generator on cell masses p (cells uniform on [-L, L], odd count, u = 0 a cell centre):
    L_f p = A_f p - Lam p + e_0 <Lam, p>,
A_f = drift-diffusion of dU = (-lam U + f) dt + sig dW with Scharfetter-Gummel fluxes, zero flux at +-L.
Occupation measure of one episode: q = (Lam - A_f)^{-1} e_0; stationary average M(f) = <u,q>/<1,q>.
Response kernel at the stationary state pi_f: k(t) = <u, e^{t L_f} b>, b = (dA_f/df) pi_f.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def bern(x):
    x = np.asarray(x, float); out = np.ones_like(x); m = np.abs(x) > 1e-12
    out[m] = x[m] / np.expm1(x[m]); return out


def dbern(x):
    x = np.asarray(x, float); out = np.empty_like(x); s = np.abs(x) < 1e-2; xs = x[s]
    out[s] = -0.5 + xs / 6.0 - xs**3 / 180.0 + xs**5 / 5040.0
    xl = x[~s]; em = np.expm1(xl); out[~s] = (em - xl * (em + 1.0)) / em**2; return out


class GHF:
    def __init__(self, Lam, lam=1.0, sig=1.0, L=8.0, h=0.01):
        n = int(np.ceil(2 * L / h)); n += (n % 2 == 0)
        self.n, self.h = n, 2 * L / n
        self.u = -L + (np.arange(n) + 0.5) * self.h; self.u[n // 2] = 0.0
        self.i0 = n // 2
        self.lam, self.sig = lam, sig
        self.Lam = np.asarray(Lam(self.u), float)

    def drift_parts(self, f):
        h, D = self.h, 0.5 * self.sig**2
        uf = -self.n * h / 2 + h * np.arange(1, self.n)
        v = -self.lam * uf + f; Pe = v * h / D
        rR = (D / h) * bern(-Pe) / h            # rate i -> i+1
        rL = (D / h) * bern(Pe) / h             # rate i+1 -> i
        drR = -dbern(-Pe) / h; drL = dbern(Pe) / h
        return rR, rL, drR, drL

    def A(self, f):
        rR, rL, _, _ = self.drift_parts(f)
        n = self.n
        main = np.zeros(n); main[:-1] -= rR; main[1:] -= rL
        return sp.diags([rR, main, rL], [-1, 0, 1], shape=(n, n), format='csc')

    def dA(self, f):
        _, _, drR, drL = self.drift_parts(f)
        n = self.n
        main = np.zeros(n); main[:-1] -= drR; main[1:] -= drL
        return sp.diags([drR, main, drL], [-1, 0, 1], shape=(n, n), format='csc')

    def occupation(self, f):
        K = sp.diags(self.Lam) - self.A(f)
        e0 = np.zeros(self.n); e0[self.i0] = 1.0
        return spla.spsolve(K.tocsc(), e0)

    def M(self, f):
        q = self.occupation(f)
        return float(self.u @ q / q.sum())

    def generator(self, f):
        n = self.n
        e0 = sp.csc_matrix((np.ones(n) * 0 + self.Lam, (np.full(n, self.i0), np.arange(n))), shape=(n, n))
        return (self.A(f) - sp.diags(self.Lam) + e0).tocsc()

    def kernel(self, f, ts):
        q = self.occupation(f); pi = q / q.sum()
        b = self.dA(f) @ pi
        Lg = self.generator(f)
        ks = spla.expm_multiply(Lg, b, start=ts[0], stop=ts[-1], num=len(ts), endpoint=True)
        return np.array([self.u @ v for v in ks])

    def S_tilted(self, f):
        """Integral of the kernel at pi_f, <u, x> with -L_f x = b and <1, x> = 0 (bordered solve)."""
        q = self.occupation(f); pi = q / q.sum()
        b = self.dA(f) @ pi
        n = self.n
        A = sp.bmat([[-self.generator(f), sp.csc_matrix(pi.reshape(n, 1))],
                     [sp.csc_matrix(np.ones((1, n))), None]], format='csc')
        x = spla.spsolve(A, np.concatenate([b, [0.0]]))[:n]
        return float(self.u @ x)

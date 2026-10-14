"""Finite-volume forward generator L* for the one-agent completion model on (u, a).

Model (push f frozen):
    dU = (-lam U + f) dt + sig dW,   dA = alpha(U) dt,
    completions at hazard H(u, a) = nu + eta chi(u) (1 - exp(-gam a)),  (U, A) -> (R(U), 0),
    alpha = chi = act_theta.

Discretisation (unknowns = cell masses m[j, i], j = a-cell, i = u-cell, a-major ordering):
  * u: uniform symmetric cells on [-Umax, Umax]; conservative Scharfetter-Gummel drift-diffusion
    flux, zero flux at +-Umax.
  * a: finite cells [a_j, a_{j+1}] (exponentially stretched, fine near a = 0) up to Amax plus one
    last cell lumping a >= Amax (no outflow, hazard evaluated at Amax). First-order upwind
    transport: mass leaves cell j at rate alpha(u_i)/da_j into cell j+1.
  * killing at rate H (cell average of 1 - exp(-gam a) over each finite a-cell);
  * reinjection of the killed mass of cell (i, j) at (R(u_i), a-cell 0), split between the two
    u-cells bracketing R(u_i) by linear interpolation (conserves mass and first moment).

Linear algebra: L = K + E_T Q_T where K (drift-diffusion + transport + killing) is block lower
bidiagonal in a with tridiagonal blocks (solved by forward marching, C kernel in march.c) and
E_T Q_T (reinjection) has rank n_T = number of u-cells that receive reinjected mass. Solves with
L - z use the Woodbury identity with an n_T x n_T capacitance matrix; all of this is a direct
method (no iteration). The model is symmetric under u -> -u, so L splits into an even sector
(contains the stationary density, eigenvalue 0) and an odd sector (contains b = -d_u pi, the
response kernel and all mean-field instabilities); each sector lives on the half grid u > 0.
"""
import os
import ctypes
from dataclasses import dataclass, asdict
import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla
from scipy.special import expit
from scipy.integrate import quad

# ----------------------------------------------------------------------------- C kernel
_here = os.path.dirname(os.path.abspath(__file__))
_lib = ctypes.CDLL(os.path.join(_here, 'march.so'))
_dp = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
_zp = np.ctypeslib.ndpointer(dtype=np.complex128, flags='C_CONTIGUOUS')
_ip = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
_ci = ctypes.c_int
for _pfx, _tp in (('d', _dp), ('z', _zp)):
    _f = getattr(_lib, _pfx + '_factor'); _f.restype = None
    _f.argtypes = [_ci, _ci, _dp, _dp, _tp, _tp, _tp]
    _f = getattr(_lib, _pfx + '_march'); _f.restype = None
    _f.argtypes = [_ci, _ci, _dp, _tp, _tp, _dp, _tp, _tp]
    _f = getattr(_lib, _pfx + '_capmarch'); _f.restype = None
    _f.argtypes = [_ci, _ci, _dp, _tp, _tp, _dp, _dp, _ci, _ip, _tp, _tp]


# ----------------------------------------------------------------------------- model pieces
def smoothstep(z):
    """s(z) = e^{-1/z} / (e^{-1/z} + e^{-1/(1-z)}) on (0,1), clipped: s = 0 (z<=0), 1 (z>=1)."""
    z = np.clip(np.asarray(z, dtype=float), 0.0, 1.0)
    out = np.where(z >= 1.0, 1.0, 0.0)
    m = (z > 0) & (z < 1)
    zm = z[m]
    out[m] = expit(1.0 / (1.0 - zm) - 1.0 / zm)
    return out


def activation(u, theta):
    u = np.asarray(u, dtype=float)
    if theta == 0:
        return np.ones_like(u)
    return smoothstep((np.abs(u) - theta) / theta)


_GLX, _GLW = np.polynomial.legendre.leggauss(80)


def _Phi(z):
    """int_0^z (1 - s(t)) dt, z in [0, 1] (C-infinity integrand: composite Gauss-Legendre)."""
    if z <= 0:
        return 0.0
    if z >= 1:
        return 0.5
    tot = 0.0
    edges = np.linspace(0.0, z, 5)
    for lo, hi in zip(edges[:-1], edges[1:]):
        t = 0.5 * (hi - lo) * _GLX + 0.5 * (hi + lo)
        tot += 0.5 * (hi - lo) * np.dot(_GLW, 1.0 - smoothstep(t))
    return tot


class Reset:
    """R_pl(u) = sgn(u) min(|u|, c); R_s(u) = sgn(u) int_0^{|u|} rho; R_0 = 0."""
    def __init__(self, kind, c):
        self.kind, self.c = kind, float(c)
        if kind == 's':
            self.b, self.s0 = 1.6 * c, 0.4 * c

    def __call__(self, u):
        u = np.asarray(u, dtype=float)
        if self.kind == '0' or self.c == 0.0:
            return np.zeros_like(u)
        x = np.abs(u)
        if self.kind == 'pl':
            return np.sign(u) * np.minimum(x, self.c)
        if self.kind == 's':
            b, s0 = self.b, self.s0
            out = np.where(x <= s0, x, self.c)
            mid = (x > s0) & (x < b)
            out[mid] = [s0 + (b - s0) * _Phi((xi - s0) / (b - s0)) for xi in x[mid]]
            return np.sign(u) * out
        raise ValueError(kind)

    def slope(self, u):
        u = np.asarray(u, dtype=float)
        if self.kind == '0' or self.c == 0.0:
            return np.zeros_like(u)
        x = np.abs(u)
        if self.kind == 'pl':
            return (x < self.c).astype(float)
        return 1.0 - smoothstep((x - self.s0) / (self.b - self.s0))


def bern(x):
    """Bernoulli function B(x) = x / (e^x - 1)."""
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    m = np.abs(x) > 1e-12
    out[m] = x[m] / np.expm1(x[m])
    return out


def dbern(x):
    """B'(x) = (e^x - 1 - x e^x) / (e^x - 1)^2."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    s = np.abs(x) < 1e-2
    xs = x[s]
    out[s] = -0.5 + xs / 6.0 - xs**3 / 180.0 + xs**5 / 5040.0
    xl = x[~s]
    em = np.expm1(xl)
    out[~s] = (em - xl * (em + 1.0)) / em**2
    return out


@dataclass
class Params:
    theta: float
    eta: float
    nu: float
    lam: float = 1.0
    gam: float = 1.0
    sig: float = 1.0
    reset: str = 'pl'          # 'pl', 's', '0'
    c: float = None            # default theta / 2

    def __post_init__(self):
        if self.c is None:
            self.c = self.theta / 2.0


@dataclass
class GridSpec:
    Umax: float
    Nu: int                    # even
    Na_f: int                  # number of finite a-cells (0 => single lumped cell; only if eta == 0)
    Amax: float = 20.0
    a_s: float = 0.5           # stretching scale: a(s) = a_s (exp(kappa s) - 1)

    def refined(self, fu=1, fa=1):
        return GridSpec(self.Umax, self.Nu * fu, self.Na_f * fa, self.Amax, self.a_s)


def default_grid(p: Params, h=0.04, Na_f=160, amax_tol=20.0, a_s=None):
    """Heuristic domain: Umax covers 2 theta + margin and the OU Gaussian tail; Amax = amax_tol/gam."""
    if p.lam > 0:
        tail = 5.0 * p.sig / np.sqrt(p.lam)          # exp(-lam u^2/sig^2) = e^-25
    else:
        tail = 25.0 * p.sig / np.sqrt(2.0 * p.nu)     # Laplace tail exp(-sqrt(2 nu) u / sig) = e^-25
    Umax = max(2 * p.theta + 3.0, tail)
    Nu = 2 * int(np.ceil(Umax / h))
    Umax = Nu * h / 2
    if p.eta == 0:
        return GridSpec(Umax, Nu, 0, 0.0, 1.0)
    Amax = amax_tol / p.gam
    if a_s is None:
        a_s = 0.5 * min(1.0 / p.gam, 1.0 / np.sqrt(p.eta * p.gam), 1.0 / (p.nu + p.eta))
        a_s = max(a_s, 0.02)
    return GridSpec(Umax, Nu, Na_f, Amax, a_s)


# ----------------------------------------------------------------------------- discretisation
class Disc:
    def __init__(self, p: Params, g: GridSpec, f=0.0):
        assert g.Nu % 2 == 0
        if g.Na_f == 0:
            assert p.eta == 0, "single lumped a-cell only valid for eta = 0"
        self.p, self.g, self.f = p, g, f
        half = np.linspace(0.0, g.Umax, g.Nu // 2 + 1)
        uf = np.concatenate([-half[::-1], half[1:]])
        self.uf = uf
        self.uc = 0.5 * (uf[1:] + uf[:-1])
        self.du = np.diff(uf)
        Nu = g.Nu
        # a grid
        if g.Na_f == 0:
            self.af = np.array([0.0])
            da = np.zeros(0)
        else:
            kappa = np.log1p(g.Amax / g.a_s)
            s = np.linspace(0.0, 1.0, g.Na_f + 1)
            af = g.a_s * np.expm1(kappa * s)
            af[0], af[-1] = 0.0, g.Amax
            self.af = af
            da = np.diff(af)
        self.da = da
        Na = len(da) + 1
        self.Nu, self.Na = Nu, Na
        gam = p.gam
        # cell average of 1 - exp(-gam a) on finite cells, value at Amax on the lump
        g1 = np.empty(Na)
        if len(da):
            a0 = self.af[:-1]
            g1[:-1] = 1.0 - np.exp(-gam * a0) * (-np.expm1(-gam * da)) / (gam * da)
            g1[-1] = -np.expm1(-gam * self.af[-1])
        else:
            g1[-1] = 1.0
        self.g1 = g1
        self.alpha = activation(self.uc, p.theta)
        self.chi = self.alpha
        self.H = p.nu + p.eta * self.chi[None, :] * g1[:, None]          # (Na, Nu)
        self.out = np.zeros((Na, Nu))
        self.ain = np.zeros((Na, Nu))
        if len(da):
            self.out[:-1] = self.alpha[None, :] / da[:, None]
            self.ain[1:] = self.alpha[None, :] / da[:, None]
        # Scharfetter-Gummel rates (mass generator): rR[i]: i -> i+1, rL[i]: i+1 -> i
        D = 0.5 * p.sig**2
        h = np.diff(self.uc)
        ufi = uf[1:-1]
        v = -p.lam * ufi + f
        Pe = v * h / D
        self.rR = (D / h) * bern(-Pe) / self.du[:-1]
        self.rL = (D / h) * bern(Pe) / self.du[1:]
        self.drR = -dbern(-Pe) / self.du[:-1]          # d rR / d f
        self.drL = dbern(Pe) / self.du[1:]             # d rL / d f
        self.lo = np.zeros(Nu); self.lo[1:] = self.rR          # Du[i, i-1]
        self.up = np.zeros(Nu); self.up[:-1] = self.rL         # Du[i, i+1]
        self.dgu = np.zeros(Nu); self.dgu[:-1] -= self.rR; self.dgu[1:] -= self.rL
        # reset map and interpolation matrix W (target cell <- source cell)
        self.Rfun = Reset(p.reset, p.c)
        self.Rv = self.Rfun(self.uc)
        self.W = self._interp_matrix(self.Rv)

    def _interp_matrix(self, t):
        uc = self.uc
        Nu = len(uc)
        k = np.searchsorted(uc, t, side='right') - 1
        k = np.clip(k, 0, Nu - 2)
        w0 = (uc[k + 1] - t) / (uc[k + 1] - uc[k])
        w0 = np.clip(w0, 0.0, 1.0)
        rows = np.concatenate([k, k + 1])
        cols = np.concatenate([np.arange(Nu), np.arange(Nu)])
        vals = np.concatenate([w0, 1.0 - w0])
        keep = vals != 0.0
        return sp.csr_matrix((vals[keep], (rows[keep], cols[keep])), shape=(Nu, Nu))

    def dDu_apply(self, m):
        """(dDu/df) m for m of shape (Na, Nu)."""
        out = np.zeros_like(m)
        out[:, 1:] += self.drR * m[:, :-1]
        out[:, :-1] += self.drL * m[:, 1:]
        out[:, :-1] -= self.drR * m[:, :-1]
        out[:, 1:] -= self.drL * m[:, 1:]
        return out


class Sector:
    """sign = None: full grid; +1: even sector; -1: odd sector (half grid u > 0)."""
    def __init__(self, d: Disc, sign=None):
        self.d, self.sign = d, sign
        Nu, Na = d.Nu, d.Na
        if sign is None:
            pos = np.arange(Nu)
            self.lo, self.up, dgu = d.lo.copy(), d.up.copy(), d.dgu.copy()
            W = d.W
            self.uw = d.uc.copy()
            self.mw = np.ones(Nu)
        else:
            hN = Nu // 2
            pos = np.arange(hN, Nu)
            mir = hN - 1 - np.arange(hN)
            self.lo = d.lo[pos].copy(); self.lo[0] = 0.0
            self.up = d.up[pos].copy()
            dgu = d.dgu[pos].copy()
            dgu[0] += sign * d.lo[hN]
            W = (d.W[pos][:, pos] + sign * d.W[pos][:, mir]).tocsr()
            W.eliminate_zeros()
            self.uw = (1 - sign) * d.uc[pos]
            self.mw = (1 + sign) * np.ones(hN)
        self.pos = pos
        self.n = len(pos)
        self.Na = Na
        self.uc = d.uc[pos]
        self.H = np.ascontiguousarray(d.H[:, pos])
        self.ain = np.ascontiguousarray(d.ain[:, pos])
        self.dgK = dgu[None, :] - d.H[:, pos] - d.out[:, pos]
        Wc = W.tocsr()
        Wc.data[np.abs(Wc.data) < 1e-15] = 0.0
        Wc.eliminate_zeros()
        self.W = Wc
        self.targets = np.unique(Wc.nonzero()[0]).astype(np.int32)
        self.nT = len(self.targets)
        self.Wt = Wc[self.targets].toarray() if self.nT else np.zeros((0, self.n))
        self.N = self.n * Na
        self._L = None

    # sparse matrix of the sector operator (for matvecs and checks)
    @property
    def L(self):
        if self._L is None:
            n, Na = self.n, self.Na
            N = n * Na
            idx = np.arange(N).reshape(Na, n)
            rows = [idx.ravel()]; cols = [idx.ravel()]; vals = [self.dgK.ravel()]
            rows.append(idx[:, 1:].ravel()); cols.append(idx[:, :-1].ravel())
            vals.append(np.tile(self.lo[1:], Na))
            rows.append(idx[:, :-1].ravel()); cols.append(idx[:, 1:].ravel())
            vals.append(np.tile(self.up[:-1], Na))
            if Na > 1:
                rows.append(idx[1:].ravel()); cols.append(idx[:-1].ravel()); vals.append(self.ain[1:].ravel())
            Wc = self.W.tocoo()
            for t, ip, w in zip(Wc.row, Wc.col, Wc.data):
                rows.append(np.full(Na, t)); cols.append(idx[:, ip]); vals.append(w * self.H[:, ip])
            self._L = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                    shape=(N, N))
        return self._L

    def inner_u(self, x):
        v = np.dot(np.tile(self.uw, self.Na), np.asarray(x).ravel())
        return v if np.iscomplexobj(v) else float(v)

    def mass(self, x):
        return np.dot(np.tile(self.mw, self.Na), x)


class Shifted:
    """Direct solver for (L_sector - z I) x = y via block marching + Woodbury (rank n_T)."""
    def __init__(self, sec: Sector, z=0.0, singular_ok=False):
        self.sec = sec
        self.z = z
        cplx = np.iscomplexobj(z) and np.imag(z) != 0.0
        self.dtype = np.complex128 if cplx else np.float64
        self.pfx = 'z' if cplx else 'd'
        zz = z if cplx else float(np.real(z))
        n, Na = sec.n, sec.Na
        dg = np.ascontiguousarray((sec.dgK - zz).astype(self.dtype))
        self.mlt = np.empty((Na, n), self.dtype)
        self.winv = np.empty((Na, n), self.dtype)
        getattr(_lib, self.pfx + '_factor')(Na, n, sec.lo, sec.up, dg, self.mlt, self.winv)
        self.QZ = None
        self.lu = None
        if sec.nT:
            S = np.empty((sec.nT, n), self.dtype)
            work = np.empty(Na * n, self.dtype)
            getattr(_lib, self.pfx + '_capmarch')(Na, n, sec.up, self.mlt, self.winv, sec.ain, sec.H,
                                                  sec.nT, sec.targets, work, S)
            self.QZ = sec.Wt @ S.T                       # Q_T K^{-1} E_T  (nT x nT)
            if not singular_ok:
                self.lu = sla.lu_factor(np.eye(sec.nT) + self.QZ)

    def march(self, y):
        y = np.ascontiguousarray(y, dtype=self.dtype).ravel()
        x = np.empty_like(y)
        getattr(_lib, self.pfx + '_march')(self.sec.Na, self.sec.n, self.sec.up, self.mlt, self.winv,
                                           self.sec.ain, y, x)
        return x

    def solve(self, y):
        sec = self.sec
        y = np.ascontiguousarray(y, dtype=np.result_type(self.dtype, np.asarray(y).dtype)).ravel()
        if y.dtype != self.dtype:      # complex rhs with real factorisation
            return self.solve(y.real) + 1j * self.solve(y.imag)
        x1 = self.march(y)
        if not sec.nT:
            return x1
        r = sec.Wt @ (sec.H * x1.reshape(sec.Na, sec.n)).sum(0)
        v = sla.lu_solve(self.lu, r)
        y2 = y.copy()
        y2[sec.targets] -= v
        return self.march(y2)


# ----------------------------------------------------------------------------- stationary problem
def stationary(sec: Sector):
    """Stationary masses on a sector containing the zero eigenvalue (full or even).
    Uses the reset-to-reset chain P = -Q_T K^{-1} E_T (column stochastic on the target cells)."""
    sh = Shifted(sec, 0.0, singular_ok=True)
    P = -sh.QZ
    nT = sec.nT
    colsum_err = np.max(np.abs(P.sum(0) - 1.0))
    A = np.eye(nT) - P
    A[-1, :] = 1.0
    rhs = np.zeros(nT); rhs[-1] = 1.0
    w = np.linalg.solve(A, rhs)
    y = np.zeros(sec.N); y[sec.targets] = w
    pi = -sh.march(y)
    pi /= sec.mass(pi)
    return pi.reshape(sec.Na, sec.n), dict(P_colsum_err=float(colsum_err), Pmin=float(P.min()))


class Model:
    """Everything for one parameter point on one grid."""
    def __init__(self, p: Params, g: GridSpec):
        self.p, self.g = p, g
        self.d = Disc(p, g, 0.0)
        self.even = Sector(self.d, +1)
        self.odd = Sector(self.d, -1)
        pi_h, info = stationary(self.even)
        self.stat_info = info
        self.pi_half = pi_h
        hN = self.d.Nu // 2
        pi = np.empty((self.d.Na, self.d.Nu))
        pi[:, hN:] = pi_h
        pi[:, :hN] = pi_h[:, ::-1]
        self.pi = pi
        b_full = self.d.dDu_apply(pi)
        self.b = np.ascontiguousarray(b_full[:, hN:]).ravel()      # odd sector representation
        self._sh0 = None

    # --- stationary statistics
    def stats(self):
        d, pi = self.d, self.pi
        u = d.uc
        mass = pi.sum()
        mean = (pi * u).sum()
        var = (pi * u**2).sum() - mean**2
        Hpi = d.H * pi
        F = Hpi.sum()
        Delta = d.Rv - u
        m2 = (Hpi * Delta**2).sum() / F
        m4 = (Hpi * Delta**4).sum() / F
        kurt = m4 / m2**2
        nonzero = (Hpi * (np.abs(Delta) > 1e-12)).sum() / F
        out = dict(mass=mass, mean=mean, var=var / self.p.sig**2, F=F, kurt=kurt,
                   kurt_over_6F=kurt / (6 * F), share_nonzero=nonzero,
                   lump_mass=pi[-1].sum() if d.Na > 1 else float('nan'),
                   edge_mass=pi[:, :3].sum() + pi[:, -3:].sum(),
                   a0_mass=pi[0].sum(), k0=self.odd.inner_u(self.b),
                   P_colsum_err=self.stat_info['P_colsum_err'])
        return out

    def S_resolvent(self):
        if self._sh0 is None:
            self._sh0 = Shifted(self.odd, 0.0)
        x = self._sh0.solve(-self.b)          # L x = -b  =>  x = (-L)^{-1} b
        return float(self.odd.inner_u(x))

    def S_fd(self, delta=1e-3):
        Ms = []
        for f in (delta, -delta):
            dfull = Disc(self.p, self.g, f)
            pif, _ = stationary(Sector(dfull, None))
            Ms.append((pif * dfull.uc[None, :]).sum())
        return (Ms[0] - Ms[1]) / (2 * delta)

    # --- Laplace transform of the kernel
    def khat(self, z, sh=None):
        if sh is None:
            sh = Shifted(self.odd, z)
        x = sh.solve(self.b)                   # (L - z) x = b  =>  (z - L)^{-1} b = -x
        return -self.odd.inner_u(x)

    # --- kernel in time: Crank-Nicolson with Rannacher start (4 implicit-Euler half steps)
    def kernel(self, T, dt1=0.01, T1=4.0, dt2=0.05, record_every=1):
        L = self.odd.L
        x = self.b.copy()
        uwt = np.tile(self.odd.uw, self.odd.Na)
        ts, ks = [0.0], [float(uwt @ x)]
        t = 0.0
        phases = [(dt1, min(T1, T))]
        if T > T1:
            phases.append((dt2, T))
        first = True
        for dt, tend in phases:
            z = 2.0 / dt
            sh = Shifted(self.odd, z)
            if first:
                for _ in range(4):     # implicit Euler with step dt/2: (L - z) x+ = -z x
                    x = sh.solve(-z * x)
                    t += dt / 2
                    ts.append(t); ks.append(float(uwt @ x))
                first = False
            nsteps = int(round((tend - t) / dt))
            for s in range(nsteps):
                x = sh.solve(-(z * x + L @ x))
                t += dt
                if (s + 1) % record_every == 0 or s == nsteps - 1:
                    ts.append(t); ks.append(float(uwt @ x))
        return np.array(ts), np.array(ks)


# ----------------------------------------------------------------------------- eigenvalues
def rank1_operator(model: Model, g, sigma):
    """Returns (OPinv as LinearOperator for (L_odd + g b u^T - sigma)^{-1}, matvec of the operator)."""
    from scipy.sparse.linalg import LinearOperator
    sec = model.odd
    sh = Shifted(sec, sigma)
    uwt = np.tile(sec.uw, sec.Na)
    b = model.b
    pb = sh.solve(b.astype(complex) if np.iscomplexobj(sigma) else b)
    denom = 1.0 + g * (uwt @ pb)

    def opinv(y):
        x = sh.solve(y)
        return x - g * pb * ((uwt @ x) / denom)

    N = sec.N
    dtype = np.complex128
    OP = LinearOperator((N, N), matvec=lambda y: opinv(np.asarray(y, dtype=dtype).ravel()).astype(dtype),
                        dtype=dtype)
    L = sec.L

    def A(x):
        return L @ x + g * b * (uwt @ x)
    return OP, A


def rightmost_eigs(model: Model, g, shifts, nev=8, tol=1e-10):
    """Eigenvalues of L_odd + g b u^T nearest to each shift (shift-invert Arnoldi); returns sorted by
    real part (descending) with residuals."""
    from scipy.sparse.linalg import eigs
    res = []
    for sigma in shifts:
        sig_c = complex(sigma)
        OP, A = rank1_operator(model, g, sig_c)
        vals, vecs = eigs(OP, k=nev, which='LM', tol=tol, ncv=max(2 * nev + 1, 24), maxiter=5000)
        for mu, v in zip(vals, vecs.T):
            lam = sig_c + 1.0 / mu
            r = np.linalg.norm(A(v) - lam * v) / np.linalg.norm(v)
            res.append((lam, r, sig_c))
    # deduplicate
    out = []
    for lam, r, s in sorted(res, key=lambda t: -t[0].real):
        if all(abs(lam - o[0]) > 1e-7 * max(1, abs(lam)) for o in out):
            out.append((lam, r, s))
    return out


# ----------------------------------------------------------------------------- renewal reference
def renewal_reference(eta, nu, lam, gam=1.0, sig=1.0):
    """theta = 0, R_0: exact quantities by quadrature (same formulas as albl_check.py)."""
    def surv(a):
        return np.exp(-nu * a - eta * (a + np.expm1(-gam * a) / gam))

    def haz(a):
        return nu + eta * (-np.expm1(-gam * a))

    def gfun(a, r):
        return a if r == 0 else -np.expm1(-r * a) / r

    def I(f):
        return quad(f, 0, np.inf, epsabs=1e-14, epsrel=1e-13, limit=500)[0]
    mu = I(surv)
    S = I(lambda a: surv(a) * gfun(a, lam)) / mu
    V = sig**2 * I(lambda a: surv(a) * gfun(a, 2 * lam)) / mu
    phi = lambda a: haz(a) * surv(a)
    m2 = sig**2 * I(lambda a: phi(a) * gfun(a, 2 * lam))
    m4 = 3 * sig**4 * I(lambda a: phi(a) * gfun(a, 2 * lam)**2)
    kurt = m4 / m2**2
    return dict(S=S, var=V / sig**2, F=1 / mu, kurt_over_6F=kurt * mu / 6)


def renewal_kernel(eta, nu, lam, t, gam=1.0):
    """theta = 0, R_0: k(t) = exp(-lam t) * Psi(t), Psi = stationary residual-life survival."""
    def surv(a):
        return np.exp(-nu * a - eta * (a + np.expm1(-gam * a) / gam))
    mu = quad(surv, 0, np.inf, epsabs=1e-14, epsrel=1e-13, limit=500)[0]
    out = []
    for ti in np.atleast_1d(t):
        Psi = quad(surv, ti, np.inf, epsabs=1e-14, epsrel=1e-13, limit=500)[0] / mu
        out.append(np.exp(-lam * ti) * Psi)
    return np.array(out)

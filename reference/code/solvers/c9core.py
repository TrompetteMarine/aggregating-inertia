"""c9 core: episode PDEs of economy B (true model, clock A) and economy P (Markovian projection).

Economy B (one agent, full resets, f frozen):
    dU = (-lam U + f) dt + sig dW,  dA = alpha(U) dt,  hazard H(u,a) = nu + eta chi(u) (1 - exp(-gam a)),
    alpha = chi = act_theta(u) = s(clip((|u|-theta)/theta, 0, 1)),  s = smooth step.
We use the clock variable b = 1 - exp(-gam a) in [0,1):  db = gam (1-b) alpha(u) dt,  H = nu + eta chi(u) b.

Discretisation (finite volumes, cell masses):
  u : uniform cells on [-L, L] (odd Nu so u = 0 is a cell centre), conservative central flux
      F = mu (x_k + x_{k+1})/2 - D (x_{k+1}-x_k)/du,  D = sig^2/2, zero-flux ends (second order).
  b : Nb uniform cells on [0,1], first-order upwind transport (velocity >= 0), no flux out at b = 1.
  time : backward Euler (fully implicit, unsplit) on a graded time grid.  The implicit system is
      block lower-triangular in b with tridiagonal blocks in u and is solved by a sweep (kernel.c).
Economy P: same u-discretisation, hazard hbar(t_n, u_i) = c_B / p_B tabulated on the same time grid.
Because sum_j over b-slices of the B scheme telescopes the b-fluxes, the P scheme with
hbar^{n+1} = c^{n+1}/p^{n+1} reproduces p_B exactly (the discrete version of claim 1).
"""
import ctypes, os, time, json
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_lib = ctypes.CDLL(os.path.join(_here, "libsweep.so"))
_dp = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
_dpn = ctypes.c_void_p
_lib.sweep.restype = ctypes.c_int
_lib.sweep.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double,
                       _dp, _dp, _dp, _dp, _dp, _dp, _dp,
                       _dpn, _dpn, _dpn,
                       _dpn, _dpn, _dpn, ctypes.c_int,
                       _dpn, _dpn, _dpn, _dpn, _dpn, _dpn, _dp]


def _ptr(a):
    return None if a is None else a.ctypes.data_as(ctypes.c_void_p)


def smoothstep(z):
    z = np.clip(np.asarray(z, dtype=float), 0.0, 1.0)
    out = (z >= 1.0).astype(float)
    m = (z > 0) & (z < 1)
    a = np.exp(-1.0 / z[m]); b = np.exp(-1.0 / (1.0 - z[m]))
    out[m] = a / (a + b)
    return out


def act(u, theta):
    if theta == 0:
        return np.ones_like(np.asarray(u, dtype=float))
    return smoothstep((np.abs(u) - theta) / theta)


class Model:
    def __init__(self, lam, theta, eta, nu, L, Nu, Nb, gam=1.0, sig=1.0, bvel="face"):
        assert Nu % 2 == 1
        self.lam, self.theta, self.eta, self.nu, self.gam, self.sig = lam, theta, eta, nu, gam, sig
        self.L, self.Nu, self.Nb = L, Nu, Nb
        self.du = 2.0 * L / Nu
        self.u = -L + (np.arange(Nu) + 0.5) * self.du
        self.u[Nu // 2] = 0.0
        self.i0 = Nu // 2
        self.alpha = act(self.u, theta)
        self.chi = self.alpha.copy()
        self.db = 1.0 / Nb
        self.b = (np.arange(Nb) + 0.5) * self.db
        self.H = np.ascontiguousarray(nu + eta * self.b[:, None] * self.chi[None, :])
        if bvel == "face":
            vb = gam * (1.0 - (np.arange(Nb) + 1.0) * self.db)          # velocity at upper face
        else:
            vb = gam * (1.0 - self.b)                                   # donor-cell velocity
        vb[-1] = 0.0
        self.R = np.ascontiguousarray(vb[:, None] * self.alpha[None, :] / self.db)
        self.w1 = np.ascontiguousarray(self.u.copy())
        self.w2 = np.ascontiguousarray(2.0 * self.u)
        self._work = np.zeros(3 * Nu)
        self.Hmax = nu + eta * self.chi            # bounds for hbar
        self.coef = {}

    def ucoef(self, f):
        """K = -L_u (mass form) tridiagonal coefficients (kl, kd, ku) for push f."""
        key = float(f)
        if key in self.coef:
            return self.coef[key]
        du, D, Nu = self.du, 0.5 * self.sig ** 2, self.Nu
        uf = -self.L + np.arange(1, Nu) * du                      # interior faces
        mu = -self.lam * uf + f
        A = (0.5 * mu + D / du) / du                              # flux = A x_k + B x_{k+1}
        B = (0.5 * mu - D / du) / du
        if np.any(A < 0) or np.any(B > 0):
            raise ValueError("cell Peclet number too large: refine du")
        kl = np.zeros(Nu); kd = np.zeros(Nu); ku = np.zeros(Nu)
        kd[:-1] += A; ku[:-1] = -B; kl[1:] = A; kd[1:] += -B
        out = tuple(np.ascontiguousarray(v) for v in (kl, kd, ku))
        self.coef[key] = out
        return out

    def sweep(self, f, smat, srhs, q_old, m1_old, m2_old, q, m1, m2, nf, Nb=None, R=None, H=None, outs=None):
        kl, kd, ku = self.ucoef(f)
        Nb = self.Nb if Nb is None else Nb
        R = self.R if R is None else R
        H = self.H if H is None else H
        Nu = self.Nu
        if outs is None:
            outs = [np.zeros(Nu) for _ in range(6)]
        _lib.sweep(Nb, Nu, smat, srhs, kl, kd, ku, R, H, self.w1, self.w2,
                   _ptr(q_old), _ptr(m1_old), _ptr(m2_old), _ptr(q), _ptr(m1), _ptr(m2), nf,
                   *[_ptr(o) for o in outs], self._work)
        return outs


# ----------------------------------------------------------------------------------------------
# Economy B: time-integrated (resolvent) episode solve  -> exact in time
# ----------------------------------------------------------------------------------------------
def resolvent_B(m, f=0.0, nf=3):
    Nb, Nu = m.Nb, m.Nu
    delta = np.zeros((Nb, Nu)); delta[0, m.i0] = 1.0
    Q = np.zeros((Nb, Nu)); M1 = np.zeros((Nb, Nu)) if nf > 1 else None
    M2 = np.zeros((Nb, Nu)) if nf > 2 else None
    p, c, pm1, cm1, pm2, cm2 = m.sweep(f, 0.0, 1.0, delta, None, None, Q, M1, M2, nf)
    u = m.u
    r = dict(Etau=p.sum(), EI=(u * p).sum(), EU2int=(u * u * p).sum(), mass_out=c.sum(),
             occ=p.copy(), rev=c.copy())
    if nf > 1:
        r["EIU"] = (u * cm1).sum()
        r["EI_alt"] = cm1.sum()
    if nf > 2:
        r["EI2"] = cm2.sum()
        r["EI2_alt"] = (2 * u * pm1).sum()
    return r


def S_from_moments(EIU, EI2, Etau, lam, sig=1.0):
    return (EIU + lam * EI2) / (sig ** 2 * Etau)


def SB_fd(m, delta=0.01):
    rp = resolvent_B(m, +delta, nf=1); rm = resolvent_B(m, -delta, nf=1)
    Mp = rp["EI"] / rp["Etau"]; Mm = rm["EI"] / rm["Etau"]
    return (Mp - Mm) / (2 * delta), Mp, Mm


# ----------------------------------------------------------------------------------------------
# Time grids
# ----------------------------------------------------------------------------------------------
def time_grid(T, dt_max, dt_min=1e-4, growth=1.06, refine=1):
    ts = [0.0]; dt = dt_min
    while ts[-1] < T:
        ts.append(ts[-1] + dt)
        dt = min(dt_max, dt * growth)
    ts = np.array(ts)
    if refine > 1:
        fine = [0.0]
        for k in range(len(ts) - 1):
            h = (ts[k + 1] - ts[k]) / refine
            fine.extend(ts[k] + h * np.arange(1, refine + 1))
        ts = np.array(fine)
    return ts


# ----------------------------------------------------------------------------------------------
# Economy B: backward-Euler time stepping of q (and optionally m1, m2)
# ----------------------------------------------------------------------------------------------
def ts_B(m, ts, f=0.0, nf=1, store=("p", "c"), mass_tol=0.0, verbose=False, snaps=()):
    """snaps: step indices n (output time ts[n+1]) at which copies of q (and m1, m2) are kept."""
    Nb, Nu = m.Nb, m.Nu
    q = np.zeros((Nb, Nu)); q[0, m.i0] = 1.0
    m1 = np.zeros((Nb, Nu)) if nf > 1 else None
    m2 = np.zeros((Nb, Nu)) if nf > 2 else None
    N = len(ts) - 1
    st = {k: np.zeros((N, Nu)) for k in store}
    outs = [np.zeros(Nu) for _ in range(6)]
    u = m.u
    acc = dict(Etau=0.0, EI=0.0, EU2int=0.0, EIU=0.0, EI2=0.0, EI2_alt=0.0)
    occ = np.zeros(Nu); rev = np.zeros(Nu)
    surv = np.zeros(N); dens = np.zeros(N)
    snapd = {}
    snaps = set(int(s) for s in snaps)
    t0 = time.time(); nlast = N
    for n in range(N):
        dt = ts[n + 1] - ts[n]; s = 1.0 / dt
        p, c, pm1, cm1, pm2, cm2 = m.sweep(f, s, s, q, m1, m2, q, m1, m2, nf, outs=outs)
        if n in snaps:
            snapd[n] = dict(q=q.copy(), m1=None if m1 is None else m1.copy(), m2=None if m2 is None else m2.copy())
        for k in store:
            st[k][n] = {"p": p, "c": c, "pm1": pm1, "cm1": cm1, "pm2": pm2, "cm2": cm2}[k]
        sp = p.sum()
        surv[n] = sp; dens[n] = c.sum()
        acc["Etau"] += dt * sp; acc["EI"] += dt * (u * p).sum(); acc["EU2int"] += dt * (u * u * p).sum()
        occ += dt * p; rev += dt * c
        if nf > 1:
            acc["EIU"] += dt * (u * cm1).sum()
        if nf > 2:
            acc["EI2"] += dt * cm2.sum(); acc["EI2_alt"] += dt * (2 * u * pm1).sum()
        if mass_tol > 0 and sp < mass_tol:
            nlast = n + 1
            break
    if verbose:
        print(f"   ts_B: {nlast} steps, T={ts[nlast]:.2f}, remaining mass {surv[nlast-1]:.2e}, {time.time()-t0:.1f}s")
    for k in st:
        st[k] = st[k][:nlast]
    acc.update(occ=occ, rev=rev, surv=surv[:nlast], dens=dens[:nlast], nsteps=nlast, ts=ts[:nlast + 1],
               mass_left=surv[nlast - 1], store=st, snaps=snapd)
    return acc


def make_hbar(m, p, c, pfloor=1e-280):
    """hbar = c/p where p is representable; elsewhere nearest valid value in u (outward extension),
    then clipped to the a-priori bounds [nu, nu + eta chi(u)].  Inside the band hbar = nu exactly."""
    N, Nu = p.shape
    hb = np.full((N, Nu), np.nan)
    valid = p > pfloor
    with np.errstate(invalid="ignore", divide="ignore"):
        hb[valid] = c[valid] / p[valid]
    idx = np.where(valid, np.arange(Nu)[None, :], -1)
    i0 = m.i0
    # right side: nearest valid index to the left (toward centre)
    left_fill = np.maximum.accumulate(idx, axis=1)
    # left side: nearest valid index to the right (toward centre)
    idx2 = np.where(valid, np.arange(Nu)[None, :], Nu)
    right_fill = np.minimum.accumulate(idx2[:, ::-1], axis=1)[:, ::-1]
    jj = np.arange(Nu)[None, :].repeat(N, 0)
    src = np.where(jj >= i0, left_fill, right_fill)
    bad = ~valid
    nbad = int(bad.sum())
    if nbad:
        rows = np.nonzero(bad)[0]
        s = src[bad]
        s = np.where((s < 0) | (s >= Nu), i0, s)
        hb[bad] = hb[rows, s]
        hb[np.isnan(hb)] = m.nu
    hb = np.clip(hb, m.nu, m.Hmax[None, :])
    hb[:, m.chi == 0] = m.nu
    return np.ascontiguousarray(hb), nbad


# ----------------------------------------------------------------------------------------------
# Economy P: 1D backward-Euler time stepping with tabulated hbar(t_n, u_i)
# ----------------------------------------------------------------------------------------------
def ts_P(m, ts, hbar, f=0.0, nf=1, store=(), compare=None):
    Nu = m.Nu
    N = hbar.shape[0]
    q = np.zeros(Nu); q[m.i0] = 1.0
    m1 = np.zeros(Nu) if nf > 1 else None
    m2 = np.zeros(Nu) if nf > 2 else None
    R0 = np.zeros(Nu)
    outs = [np.zeros(Nu) for _ in range(6)]
    st = {k: np.zeros((N, Nu)) for k in store}
    u = m.u
    acc = dict(Etau=0.0, EI=0.0, EU2int=0.0, EIU=0.0, EI2=0.0, EI2_alt=0.0)
    occ = np.zeros(Nu); rev = np.zeros(Nu)
    surv = np.zeros(N); dens = np.zeros(N)
    maxdiff = dict(p=0.0, c=0.0)
    for n in range(N):
        dt = ts[n + 1] - ts[n]; s = 1.0 / dt
        H = hbar[n]
        p, c, pm1, cm1, pm2, cm2 = m.sweep(f, s, s, q, m1, m2, q, m1, m2, nf, Nb=1, R=R0, H=H, outs=outs)
        for k in store:
            st[k][n] = {"p": p, "c": c, "pm1": pm1, "cm1": cm1, "pm2": pm2, "cm2": cm2}[k]
        sp = p.sum(); surv[n] = sp; dens[n] = c.sum()
        acc["Etau"] += dt * sp; acc["EI"] += dt * (u * p).sum(); acc["EU2int"] += dt * (u * u * p).sum()
        occ += dt * p; rev += dt * c
        if nf > 1:
            acc["EIU"] += dt * (u * cm1).sum()
        if nf > 2:
            acc["EI2"] += dt * cm2.sum(); acc["EI2_alt"] += dt * (2 * u * pm1).sum()
        if compare is not None:
            maxdiff["p"] = max(maxdiff["p"], np.abs(p - compare["p"][n]).max())
            maxdiff["c"] = max(maxdiff["c"], np.abs(c - compare["c"][n]).max())
    acc.update(occ=occ, rev=rev, surv=surv, dens=dens, store=st, maxdiff=maxdiff, mass_left=surv[-1])
    return acc


def SP_fd(m, ts, hbar, delta=0.01):
    rp = ts_P(m, ts, hbar, +delta, nf=1); rm = ts_P(m, ts, hbar, -delta, nf=1)
    Mp = rp["EI"] / rp["Etau"]; Mm = rm["EI"] / rm["Etau"]
    return (Mp - Mm) / (2 * delta), Mp, Mm


# ----------------------------------------------------------------------------------------------
# Summary statistics from the occupation measure and the revision (completion) measure
# ----------------------------------------------------------------------------------------------
def stats_from(m, r):
    u = m.u
    Etau = r["Etau"]; F = 1.0 / Etau
    var = r["EU2int"] / Etau - (r["EI"] / Etau) ** 2
    rev = r["rev"]; tot = rev.sum()
    d2 = (u ** 2 * rev).sum() / tot; d4 = (u ** 4 * rev).sum() / tot
    kurt = d4 / d2 ** 2
    out = dict(Etau=Etau, F=F, VarU=var / m.sig ** 2, Ed2=d2, Ed4=d4, Kurt=kurt, ALO=kurt / (6 * F),
               VarU_formula=(m.sig ** 2 - F * d2) / (2 * m.lam) / m.sig ** 2 if m.lam > 0 else np.nan,
               rev_mass=tot)
    if "EIU" in r and "EI2" in r:
        out["S_formula"] = S_from_moments(r["EIU"], r["EI2"], Etau, m.lam, m.sig)
        out["EIU"] = r["EIU"]; out["EI2"] = r["EI2"]
        out["EI2_alt"] = r.get("EI2_alt", np.nan)
        out["S_girsanov_split"] = var / m.sig ** 2 + m.lam * r["EI2"] / (2 * m.sig ** 2 * Etau)
    return out


def default_L(lam):
    return {0.0: 18.0, 0.25: 12.0, 1.0: 6.5, 2.0: 5.5}.get(float(lam), 8.0)


def odd(n):
    n = int(round(n))
    return n if n % 2 == 1 else n + 1

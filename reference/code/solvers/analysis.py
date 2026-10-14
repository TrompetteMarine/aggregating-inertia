"""Kernel, Laplace transform, first-crossing detection, eigenvalue tracking and Nyquist certificate.

Conventions (odd sector, where everything relevant to the mean-field feedback lives):
  k(t)    = <u, e^{L t} b>,           k(0) = <u,b> ~ 1,
  khat(z) = <u, (z - L)^{-1} b>,       khat(0) = S,
  characteristic equation 1 = g khat(z)  <=>  z eigenvalue of L_odd + g b u^T.
Since khat has no poles in Re z >= 0 (L_odd generates a strictly decaying semigroup: the zero
eigenvalue of L is even) and khat(z) -> 0 as |z| -> inf, roots can enter Re z > 0 only through the
imaginary axis, at g = 1/khat(i w) for w with khat(i w) real and positive. Hence the first
instability is at g* = 1 / max{ khat(i w) : w >= 0, khat(i w) in (0, inf) }; the w = 0 candidate is S.
"""
import numpy as np
from scipy.optimize import brentq
from fvsolver import Shifted, rightmost_eigs


# ----------------------------------------------------------------------------- kernel in time
def kernel_stats(model, T=None, dt1=0.01, T1=4.0, dt2=0.05, gap=None):
    if T is None:
        if gap is None:
            gap = slowest_rate(model)
        T = float(min(600.0, max(40.0, 36.0 / gap)))
    t, k = model.kernel(T, dt1=dt1, T1=T1, dt2=dt2)
    i = int(np.argmin(k))
    dtv = np.diff(t)
    integ = np.sum(0.5 * (k[1:] + k[:-1]) * dtv)
    absmom1 = np.sum(0.5 * (t[1:] * np.abs(k[1:]) + t[:-1] * np.abs(k[:-1])) * dtv)
    absint = np.sum(0.5 * (np.abs(k[1:]) + np.abs(k[:-1])) * dtv)
    tv = np.sum(np.abs(np.diff(k)))
    neg = k < 0
    negmass = np.sum(0.5 * (np.minimum(k[1:], 0) + np.minimum(k[:-1], 0)) * dtv)
    return dict(T=T, kmin=float(k[i]), tmin=float(t[i]), int_k=float(integ), int_t_absk=float(absmom1),
                int_absk=float(absint), TV=float(tv), k_T=float(k[-1]), k0=float(k[0]),
                first_neg_t=float(t[np.argmax(neg)]) if neg.any() else None,
                neg_mass=float(negmass)), (t, k)


def slowest_rate(model):
    """Spectral gap of L_odd: minus the real part of its rightmost eigenvalue (shift-invert at 0)."""
    ev = rightmost_eigs(model, 0.0, [0.0], nev=6, tol=1e-9)
    return -max(e[0].real for e in ev)


# ----------------------------------------------------------------------------- Nyquist curve
class KhatEvaluator:
    def __init__(self, model):
        self.m = model
        self.cache = {}
        self.nevals = 0

    def __call__(self, w):
        w = float(w)
        if w not in self.cache:
            z = 1j * w if w != 0 else 0.0
            self.cache[w] = complex(self.m.khat(z))
            self.nevals += 1
        return self.cache[w]


def adaptive_curve(kh, S, w_max=60.0, n0=90, rel_step=0.03, max_pts=1500):
    """Samples w -> khat(i w) on [0, w_max], refining until consecutive points are within
    rel_step * max(S, |khat|) of each other and the arg of khat changes by < 0.25 rad."""
    ws = np.concatenate([[0.0], np.geomspace(1e-3, w_max, n0)])
    vals = [kh(w) for w in ws]
    ws = list(ws)
    i = 0
    while i < len(ws) - 1 and len(ws) < max_pts:
        a, b = vals[i], vals[i + 1]
        scale = max(min(abs(a), abs(b)), 0.05 * S)
        dang = abs(np.angle(b / a)) if a != 0 and b != 0 else np.pi
        if abs(b - a) > rel_step * max(S, scale) or dang > 0.25:
            wm = 0.5 * (ws[i] + ws[i + 1])
            if ws[i + 1] - ws[i] < 1e-7:
                i += 1
                continue
            ws.insert(i + 1, wm)
            vals.insert(i + 1, kh(wm))
        else:
            i += 1
    return np.array(ws), np.array(vals)


def positive_real_crossings(kh, ws, vals):
    """w > 0 where Im khat(i w) changes sign; refined by Brent. Returns list of (w, Re khat)."""
    out = []
    im = vals.imag
    for i in range(1, len(ws) - 1):
        if ws[i] == 0:
            continue
        if im[i] == 0.0 or np.sign(im[i]) != np.sign(im[i + 1]):
            if vals[i].real <= 0 and vals[i + 1].real <= 0:
                continue          # crossing of the negative real axis: never an instability
            try:
                wc = brentq(lambda w: kh(w).imag, ws[i], ws[i + 1], xtol=1e-12, rtol=1e-12, maxiter=100)
            except ValueError:
                continue
            out.append((wc, kh(wc).real))
    return out


def first_crossing(model, w_max=60.0, n0=90):
    kh = KhatEvaluator(model)
    S = kh(0.0).real
    ws, vals = adaptive_curve(kh, S, w_max=w_max, n0=n0)
    cr = positive_real_crossings(kh, ws, vals)
    pos = [(w, x) for (w, x) in cr if x > 0]
    best = max(pos, key=lambda t: t[1]) if pos else None
    if best is not None and best[1] > S:
        res = dict(type='oscillatory', g=1.0 / best[1], omega=best[0])
    else:
        res = dict(type='real', g=1.0 / S, omega=0.0)
    res.update(S=S, invS=1.0 / S, crossings=cr, nevals=kh.nevals,
               max_abs_khat_w_pos=float(np.max(np.abs(vals[1:]))),
               min_re_khat=float(np.min(vals.real)), curve=(ws, vals))
    return res


# ----------------------------------------------------------------------------- eigenvalues
def eig_check(model, g, shifts, nev=6):
    ev = rightmost_eigs(model, g, shifts, nev=nev, tol=1e-10)
    lam, res, sh = ev[0]
    return dict(g=g, rightmost=lam, residual=res, shift=sh,
                top=[(complex(e[0]), float(e[1])) for e in ev[:4]])


# ----------------------------------------------------------------------------- certificate
def certify_count(model, g, kstats, w_cap=None, safety=2.0, max_evals=20000, verbose=False):
    """Number of zeros of F(z) = 1 - g khat(z) in Re z > 0 by the argument principle on the
    imaginary axis, with rigorous step control for the discrete operator:
       |d khat(i w)/dw| <= M1 := int t |k(t)| dt   and   |khat(i w)| <= (k(0) + TV(k)) / w.
    The bounds are taken from the time-stepped kernel times a safety factor. Between samples
    w_n < w_{n+1} with g M1 (w_{n+1} - w_n) < |F(i w_n)|, F stays in a disc that excludes 0, so the
    continuous arg change equals the principal one. For w >= Omega := safety*g*(k0+TV) we have
    |g khat| <= 1/safety < 1, so F stays in Re F > 0 up to F(i inf) = 1.
    N = (theta(0) - theta(inf)) / pi, theta = continuous arg F(i w) (conjugate symmetry)."""
    M1 = safety * kstats['int_t_absk']
    B = safety * (abs(kstats['k0']) + kstats['TV'])
    Omega = safety * g * B if w_cap is None else w_cap
    kh = KhatEvaluator(model)
    F = lambda w: 1.0 - g * kh(w)
    w = 0.0
    f0 = F(0.0)
    assert abs(f0.imag) < 1e-12
    theta0 = 0.0 if f0.real > 0 else np.pi
    theta = theta0
    fprev = f0
    minabs = abs(f0)
    nsteps = 0
    while w < Omega:
        step = 0.5 * abs(fprev) / (g * M1)
        if step < 1e-9:
            raise RuntimeError(f"step underflow at w={w}: |F|={abs(fprev)} (g too close to a crossing)")
        wn = min(w + step, Omega)
        fn = F(wn)
        theta += np.angle(fn / fprev)
        minabs = min(minabs, abs(fn))
        w, fprev = wn, fn
        nsteps += 1
        if nsteps > max_evals:
            raise RuntimeError("too many steps")
    # tail: |g khat(i w)| <= g B / w <= 1/safety for w >= Omega; F in disc around 1 of radius < 1
    tail_bound = g * B / Omega
    assert tail_bound < 1.0
    theta_inf = theta - np.angle(fprev)          # arg 1 = 0 reached without leaving Re F > 0
    N = (theta0 - theta_inf) / np.pi
    return dict(g=g, N=float(N), N_round=int(round(N)), Omega=Omega, M1=M1, B=B, nsteps=nsteps,
                min_absF=float(minabs), tail_bound=tail_bound, F0=f0.real)

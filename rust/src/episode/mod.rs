//! Episode solver: economy B (true model with a processing clock) and economy P (its Markovian
//! projection). Port of the reference `c9core.py` and `kernel.c`.
//!
//! Economy B, one agent, full resets, push f frozen:
//!   dU = (-lam U + f) dt + sig dW,  dA = alpha(U) dt,  H(u, a) = nu + eta chi(u) (1 - exp(-gam a)).
//! With the clock variable b = 1 - exp(-gam a) in [0, 1): db = gam (1 - b) alpha(u) dt and
//! H = nu + eta chi(u) b.
//!
//! Discretisation (finite volumes on cell masses):
//! * u: uniform cells on [-L, L], odd count so that u = 0 is a cell centre; central flux
//!   F = mu (x_k + x_{k+1}) / 2 - D (x_{k+1} - x_k) / du, D = sig^2 / 2, zero flux at the ends.
//! * b: Nb uniform cells on [0, 1], first-order upwind transport, no flux out at b = 1.
//! * time: backward Euler on a graded grid. The implicit system is block lower triangular in b with
//!   tridiagonal blocks in u and is solved by one sweep.
//! Economy P: the same u-discretisation, with the hazard hbar(t_n, u_i) = c_B / p_B tabulated on the
//! time grid. Because summing the B scheme over b-slices telescopes the b-fluxes, P with
//! hbar^{n+1} = c^{n+1} / p^{n+1} reproduces p_B exactly (the discrete form of Theorem 3).

pub mod claim1;
pub mod run;
pub mod speed;

use crate::util::{self, fp::FtzGuard, np_sum};
use serde::{Deserialize, Serialize};

/// Completion technology. `Pausing` is Example 1 of the paper; the others are the history-dependent
/// variants of Section 5 (processing accumulating at rate |U| or U^2, hazards smooth or at a threshold).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Clock {
    /// alpha = chi = s((|u| - theta) / theta), H = nu + eta chi(u) (1 - e^{-gam a})
    Pausing,
    /// dA = |U| dt, H = nu + eta (1 - e^{-gam A})
    Dist,
    /// dA = U^2 dt, H = nu + eta (1 - e^{-gam A})
    Sq,
    /// dA = |U| dt, H = nu + eta 1{A > kappa}
    DistThr(f64),
}

impl Clock {
    pub fn label(&self) -> String {
        match self {
            Clock::Pausing => "clock".into(),
            Clock::Dist => "dist".into(),
            Clock::Sq => "sq".into(),
            Clock::DistThr(k) => format!("distthr_{:?}", k),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Model {
    pub clock: Clock,
    pub lam: f64,
    pub theta: f64,
    pub eta: f64,
    pub nu: f64,
    pub gam: f64,
    pub sig: f64,
    pub l: f64,
    /// number of assessment cells (odd)
    pub n_u: usize,
    /// number of clock cells
    pub nb: usize,
    pub du: f64,
    pub u: Vec<f64>,
    pub i0: usize,
    pub alpha: Vec<f64>,
    pub chi: Vec<f64>,
    pub db: f64,
    pub b: Vec<f64>,
    /// hazard on (b, u), row-major Nb x Nu
    pub h: Vec<f64>,
    /// upwind outflow rate from b-slice j to j + 1, Nb x Nu
    pub r: Vec<f64>,
    pub w1: Vec<f64>,
    pub w2: Vec<f64>,
    /// a priori bounds of the projected hazard
    pub hmax: Vec<f64>,
}

/// Tridiagonal coefficients of K = -L_u (mass form) for a given push.
#[derive(Clone, Debug)]
pub struct UCoef {
    pub kl: Vec<f64>,
    pub kd: Vec<f64>,
    pub ku: Vec<f64>,
}

/// Slice sums returned by a sweep: p = sum_j q, c = sum_j H q, and the same for the two moment fields.
#[derive(Clone, Debug)]
pub struct SweepOut {
    pub p: Vec<f64>,
    pub c: Vec<f64>,
    pub pm1: Vec<f64>,
    pub cm1: Vec<f64>,
    pub pm2: Vec<f64>,
    pub cm2: Vec<f64>,
}

impl SweepOut {
    pub fn new(n: usize) -> Self {
        SweepOut { p: vec![0.0; n], c: vec![0.0; n], pm1: vec![0.0; n], cm1: vec![0.0; n], pm2: vec![0.0; n], cm2: vec![0.0; n] }
    }
}

impl Model {
    pub fn new(lam: f64, theta: f64, eta: f64, nu: f64, l: f64, n_u: usize, nb: usize) -> Self {
        Self::with_clock(Clock::Pausing, lam, theta, eta, nu, l, n_u, nb, 1.0, 1.0)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_clock(clock: Clock, lam: f64, theta: f64, eta: f64, nu: f64, l: f64, n_u: usize, nb: usize, gam: f64, sig: f64) -> Self {
        assert!(n_u % 2 == 1, "the number of assessment cells must be odd");
        let du = 2.0 * l / n_u as f64;
        let mut u: Vec<f64> = (0..n_u).map(|i| -l + (i as f64 + 0.5) * du).collect();
        let i0 = n_u / 2;
        u[i0] = 0.0;
        let (alpha, chi): (Vec<f64>, Vec<f64>) = match clock {
            Clock::Pausing => {
                let a: Vec<f64> = u.iter().map(|&x| util::activation(x, theta)).collect();
                (a.clone(), a)
            }
            Clock::Dist | Clock::DistThr(_) => (u.iter().map(|x| x.abs()).collect(), vec![1.0; n_u]),
            Clock::Sq => (u.iter().map(|x| x * x).collect(), vec![1.0; n_u]),
        };
        let db = 1.0 / nb as f64;
        let b: Vec<f64> = (0..nb).map(|j| (j as f64 + 0.5) * db).collect();
        let mut h = vec![0.0; nb * n_u];
        for j in 0..nb {
            for i in 0..n_u {
                h[j * n_u + i] = match clock {
                    Clock::DistThr(kappa) => {
                        let bk = 1.0 - (-gam * kappa).exp();
                        nu + eta * (if b[j] > bk { 1.0 } else { 0.0 }) * chi[i]
                    }
                    _ => nu + eta * b[j] * chi[i],
                };
            }
        }
        let mut vb: Vec<f64> = (0..nb).map(|j| gam * (1.0 - (j as f64 + 1.0) * db)).collect();
        vb[nb - 1] = 0.0;
        let mut r = vec![0.0; nb * n_u];
        for j in 0..nb {
            for i in 0..n_u {
                r[j * n_u + i] = vb[j] * alpha[i] / db;
            }
        }
        let w1 = u.clone();
        let w2: Vec<f64> = u.iter().map(|x| 2.0 * x).collect();
        let hmax: Vec<f64> = chi.iter().map(|c| nu + eta * c).collect();
        Model { clock, lam, theta, eta, nu, gam, sig, l, n_u, nb, du, u, i0, alpha, chi, db, b, h, r, w1, w2, hmax }
    }

    /// K = -L_u (mass form) tridiagonal coefficients for push f.
    pub fn ucoef(&self, f: f64) -> UCoef {
        let (du, d, n) = (self.du, 0.5 * self.sig * self.sig, self.n_u);
        let mut kl = vec![0.0; n];
        let mut kd = vec![0.0; n];
        let mut ku = vec![0.0; n];
        for k in 0..n - 1 {
            let uf = -self.l + (k as f64 + 1.0) * du;
            let mu = -self.lam * uf + f;
            let a = (0.5 * mu + d / du) / du;
            let bb = (0.5 * mu - d / du) / du;
            assert!(a >= 0.0 && bb <= 0.0, "cell Peclet number too large: refine du");
            kd[k] += a;
            ku[k] = -bb;
            kl[k + 1] = a;
            kd[k + 1] += -bb;
        }
        UCoef { kl, kd, ku }
    }

    /// One implicit sweep over `nb` b-slices (in place). For every slice j it solves
    ///   (smat + kd + R_j + H_j) X_j - kl X_{j,i-1} - ku X_{j,i+1} = srhs OLD_j + SRC_j + R_{j-1} X_{j-1}
    /// for q (SRC = 0), m1 (SRC = w1 q) and m2 (SRC = w2 m1); the buffers hold OLD on entry and the
    /// solution on exit.
    #[allow(clippy::too_many_arguments)]
    pub fn sweep(&self, coef: &UCoef, smat: f64, srhs: f64, nb: usize, r: &[f64], h: &[f64], w1: &[f64], w2: &[f64],
                 q: &mut [f64], mut m1: Option<&mut [f64]>, mut m2: Option<&mut [f64]>, nf: usize, out: &mut SweepOut) {
        let _ftz = FtzGuard::new();
        let n = self.n_u;
        let (kl, kd, ku) = (&coef.kl, &coef.kd, &coef.ku);
        let mut cp = vec![0.0; n];
        let mut im = vec![0.0; n];
        let mut d = vec![0.0; n];
        for v in [&mut out.p, &mut out.c] {
            v.iter_mut().for_each(|x| *x = 0.0);
        }
        if nf > 1 {
            out.pm1.iter_mut().for_each(|x| *x = 0.0);
            out.cm1.iter_mut().for_each(|x| *x = 0.0);
        }
        if nf > 2 {
            out.pm2.iter_mut().for_each(|x| *x = 0.0);
            out.cm2.iter_mut().for_each(|x| *x = 0.0);
        }
        for j in 0..nb {
            let off = j * n;
            let rj = &r[off..off + n];
            let hj = &h[off..off + n];
            // factorisation of the tridiagonal block of slice j
            {
                let mut m = smat + kd[0] + rj[0] + hj[0];
                im[0] = 1.0 / m;
                cp[0] = -ku[0] * im[0];
                for i in 1..n {
                    m = smat + kd[i] + rj[i] + hj[i] + kl[i] * cp[i - 1];
                    im[i] = 1.0 / m;
                    cp[i] = -ku[i] * im[i];
                }
            }
            // field 0
            solve_field(q, None, w1, srhs, j, n, r, kl, &cp, &im, &mut d);
            for i in 0..n {
                let x = q[off + i];
                out.p[i] += x;
                out.c[i] += hj[i] * x;
            }
            if nf > 1 {
                let m1 = m1.as_deref_mut().expect("m1 buffer");
                solve_field(m1, Some(&q[off..off + n]), w1, srhs, j, n, r, kl, &cp, &im, &mut d);
                for i in 0..n {
                    let x = m1[off + i];
                    out.pm1[i] += x;
                    out.cm1[i] += hj[i] * x;
                }
                if nf > 2 {
                    let m2b = m2.as_deref_mut().expect("m2 buffer");
                    solve_field(m2b, Some(&m1[off..off + n]), w2, srhs, j, n, r, kl, &cp, &im, &mut d);
                    for i in 0..n {
                        let x = m2b[off + i];
                        out.pm2[i] += x;
                        out.cm2[i] += hj[i] * x;
                    }
                }
            }
        }
    }

    /// Weight vectors of the moment fields: w1 = u and w2 = 2u (moments of the area I = int U dt).
    pub fn weights(&self) -> (&[f64], &[f64]) {
        (&self.w1, &self.w2)
    }
}

/// Solve one field on slice j in place: x_j = T_j^{-1} (srhs old_j + w src_j + R_{j-1} x_{j-1}).
#[allow(clippy::too_many_arguments)]
#[inline]
fn solve_field(x: &mut [f64], src: Option<&[f64]>, w: &[f64], srhs: f64, j: usize, n: usize, r: &[f64], kl: &[f64],
               cp: &[f64], im: &[f64], d: &mut [f64]) {
    let off = j * n;
    for i in 0..n {
        let mut v = srhs * x[off + i];
        if let Some(s) = src {
            v += w[i] * s[i];
        }
        if j > 0 {
            v += r[off - n + i] * x[off - n + i];
        }
        d[i] = v;
    }
    d[0] *= im[0];
    for i in 1..n {
        d[i] = (d[i] + kl[i] * d[i - 1]) * im[i];
    }
    x[off + n - 1] = d[n - 1];
    for i in (0..n - 1).rev() {
        x[off + i] = d[i] - cp[i] * x[off + i + 1];
    }
}

fn wsum(w: &[f64], x: &[f64]) -> f64 {
    let t: Vec<f64> = w.iter().zip(x).map(|(a, b)| a * b).collect();
    np_sum(&t)
}

fn wsum2(w: &[f64], x: &[f64]) -> f64 {
    let t: Vec<f64> = w.iter().zip(x).map(|(a, b)| a * a * b).collect();
    np_sum(&t)
}

/// Result of a time-integrated (resolvent) episode solve: exact in time.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Resolvent {
    pub etau: f64,
    pub ei: f64,
    pub eu2int: f64,
    pub mass_out: f64,
    #[serde(skip)]
    pub occ: Vec<f64>,
    #[serde(skip)]
    pub rev: Vec<f64>,
    pub eiu: f64,
    pub ei_alt: f64,
    pub ei2: f64,
    pub ei2_alt: f64,
}

/// Economy B, time-integrated episode solve with push f; nf fields (1: q, 2: + E[I; .], 3: + E[I^2; .]).
pub fn resolvent_b(m: &Model, f: f64, nf: usize) -> Resolvent {
    let (nb, n) = (m.nb, m.n_u);
    let mut q = vec![0.0; nb * n];
    q[m.i0] = 1.0;
    let mut m1 = if nf > 1 { vec![0.0; nb * n] } else { vec![] };
    let mut m2 = if nf > 2 { vec![0.0; nb * n] } else { vec![] };
    let mut out = SweepOut::new(n);
    let coef = m.ucoef(f);
    m.sweep(&coef, 0.0, 1.0, nb, &m.r, &m.h, &m.w1, &m.w2, &mut q,
            if nf > 1 { Some(&mut m1) } else { None }, if nf > 2 { Some(&mut m2) } else { None }, nf, &mut out);
    let u = &m.u;
    let mut r = Resolvent {
        etau: np_sum(&out.p),
        ei: wsum(u, &out.p),
        eu2int: wsum2(u, &out.p),
        mass_out: np_sum(&out.c),
        occ: out.p.clone(),
        rev: out.c.clone(),
        ..Default::default()
    };
    if nf > 1 {
        r.eiu = wsum(u, &out.cm1);
        r.ei_alt = np_sum(&out.cm1);
    }
    if nf > 2 {
        r.ei2 = np_sum(&out.cm2);
        let two_u: Vec<f64> = u.iter().map(|x| 2.0 * x).collect();
        r.ei2_alt = wsum(&two_u, &out.pm1);
    }
    r
}

pub fn s_from_moments(eiu: f64, ei2: f64, etau: f64, lam: f64, sig: f64) -> f64 {
    (eiu + lam * ei2) / (sig * sig * etau)
}

/// S_B by a central difference of M(f) = E[I_tau] / E[tau]; returns (S, M(+delta), M(-delta)).
pub fn sb_fd(m: &Model, delta: f64) -> (f64, f64, f64) {
    let rp = resolvent_b(m, delta, 1);
    let rm = resolvent_b(m, -delta, 1);
    let mp = rp.ei / rp.etau;
    let mm = rm.ei / rm.etau;
    ((mp - mm) / (2.0 * delta), mp, mm)
}

/// Which per-step slice sums to keep.
#[derive(Clone, Copy, Debug, Default)]
pub struct Store {
    pub p: bool,
    pub c: bool,
    pub pm1: bool,
    pub cm1: bool,
    pub pm2: bool,
    pub cm2: bool,
}

impl Store {
    pub fn pc() -> Self {
        Store { p: true, c: true, ..Default::default() }
    }
    pub fn none() -> Self {
        Store::default()
    }
}

/// Per-step stored slice sums, each N x Nu row-major (row n is output time ts[n + 1]).
#[derive(Clone, Debug, Default)]
pub struct Stored {
    pub p: Vec<f64>,
    pub c: Vec<f64>,
    pub pm1: Vec<f64>,
    pub cm1: Vec<f64>,
    pub pm2: Vec<f64>,
    pub cm2: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct Snap {
    pub q: Vec<f64>,
    pub m1: Option<Vec<f64>>,
    pub m2: Option<Vec<f64>>,
}

/// Result of a time-stepped episode solve (B or P).
#[derive(Clone, Debug, Default)]
pub struct Stepped {
    pub etau: f64,
    pub ei: f64,
    pub eu2int: f64,
    pub eiu: f64,
    pub ei2: f64,
    pub ei2_alt: f64,
    pub occ: Vec<f64>,
    pub rev: Vec<f64>,
    pub surv: Vec<f64>,
    pub dens: Vec<f64>,
    pub nsteps: usize,
    pub ts: Vec<f64>,
    pub mass_left: f64,
    pub store: Stored,
    pub snaps: Vec<(usize, Snap)>,
    pub maxdiff_p: f64,
    pub maxdiff_c: f64,
}

struct Acc {
    etau: f64,
    ei: f64,
    eu2int: f64,
    eiu: f64,
    ei2: f64,
    ei2_alt: f64,
    occ: Vec<f64>,
    rev: Vec<f64>,
}

impl Acc {
    fn new(n: usize) -> Self {
        Acc { etau: 0.0, ei: 0.0, eu2int: 0.0, eiu: 0.0, ei2: 0.0, ei2_alt: 0.0, occ: vec![0.0; n], rev: vec![0.0; n] }
    }
    fn add(&mut self, u: &[f64], dt: f64, out: &SweepOut, nf: usize) -> f64 {
        let sp = np_sum(&out.p);
        self.etau += dt * sp;
        self.ei += dt * wsum(u, &out.p);
        self.eu2int += dt * wsum2(u, &out.p);
        for i in 0..u.len() {
            self.occ[i] += dt * out.p[i];
            self.rev[i] += dt * out.c[i];
        }
        if nf > 1 {
            self.eiu += dt * wsum(u, &out.cm1);
        }
        if nf > 2 {
            self.ei2 += dt * np_sum(&out.cm2);
            let two_u: Vec<f64> = u.iter().map(|x| 2.0 * x).collect();
            self.ei2_alt += dt * wsum(&two_u, &out.pm1);
        }
        sp
    }
}

fn store_row(st: &mut Stored, which: &Store, out: &SweepOut) {
    if which.p {
        st.p.extend_from_slice(&out.p);
    }
    if which.c {
        st.c.extend_from_slice(&out.c);
    }
    if which.pm1 {
        st.pm1.extend_from_slice(&out.pm1);
    }
    if which.cm1 {
        st.cm1.extend_from_slice(&out.cm1);
    }
    if which.pm2 {
        st.pm2.extend_from_slice(&out.pm2);
    }
    if which.cm2 {
        st.cm2.extend_from_slice(&out.cm2);
    }
}

/// Economy B, backward-Euler time stepping of q (and optionally the moment fields).
/// `snaps`: step indices n (output time ts[n + 1]) at which copies of the fields are kept.
#[allow(clippy::too_many_arguments)]
pub fn ts_b(m: &Model, ts: &[f64], f: f64, nf: usize, store: Store, mass_tol: f64, snaps: &[usize]) -> Stepped {
    let (nb, n) = (m.nb, m.n_u);
    let mut q = vec![0.0; nb * n];
    q[m.i0] = 1.0;
    let mut m1 = if nf > 1 { vec![0.0; nb * n] } else { vec![] };
    let mut m2 = if nf > 2 { vec![0.0; nb * n] } else { vec![] };
    let nsteps_max = ts.len() - 1;
    let mut out = SweepOut::new(n);
    let coef = m.ucoef(f);
    let mut acc = Acc::new(n);
    let mut surv = Vec::with_capacity(nsteps_max);
    let mut dens = Vec::with_capacity(nsteps_max);
    let mut st = Stored::default();
    let mut snapd = Vec::new();
    let mut nlast = nsteps_max;
    for k in 0..nsteps_max {
        let dt = ts[k + 1] - ts[k];
        let s = 1.0 / dt;
        m.sweep(&coef, s, s, nb, &m.r, &m.h, &m.w1, &m.w2, &mut q,
                if nf > 1 { Some(&mut m1) } else { None }, if nf > 2 { Some(&mut m2) } else { None }, nf, &mut out);
        if snaps.contains(&k) {
            snapd.push((k, Snap { q: q.clone(), m1: if nf > 1 { Some(m1.clone()) } else { None }, m2: if nf > 2 { Some(m2.clone()) } else { None } }));
        }
        store_row(&mut st, &store, &out);
        let sp = acc.add(&m.u, dt, &out, nf);
        surv.push(sp);
        dens.push(np_sum(&out.c));
        if mass_tol > 0.0 && sp < mass_tol {
            nlast = k + 1;
            break;
        }
    }
    let mass_left = surv[nlast - 1];
    Stepped {
        etau: acc.etau, ei: acc.ei, eu2int: acc.eu2int, eiu: acc.eiu, ei2: acc.ei2, ei2_alt: acc.ei2_alt,
        occ: acc.occ, rev: acc.rev, surv, dens, nsteps: nlast, ts: ts[..nlast + 1].to_vec(), mass_left,
        store: st, snaps: snapd, maxdiff_p: 0.0, maxdiff_c: 0.0,
    }
}

/// hbar = c / p where p is representable; elsewhere the nearest valid value in u (outward
/// extension), then clipped to the a priori bounds [nu, nu + eta chi(u)]; hbar = nu where chi = 0.
pub fn make_hbar(m: &Model, p: &[f64], c: &[f64], pfloor: f64) -> (Vec<f64>, usize) {
    let n = m.n_u;
    let rows = p.len() / n;
    let mut hb = vec![f64::NAN; rows * n];
    let mut nbad = 0;
    for row in 0..rows {
        let off = row * n;
        let valid: Vec<bool> = (0..n).map(|i| p[off + i] > pfloor).collect();
        for i in 0..n {
            if valid[i] {
                hb[off + i] = c[off + i] / p[off + i];
            }
        }
        // nearest valid index at or to the left / right
        let mut left = vec![-1i64; n];
        let mut best = -1i64;
        for i in 0..n {
            if valid[i] {
                best = best.max(i as i64);
            }
            left[i] = best;
        }
        let mut right = vec![n as i64; n];
        let mut bestr = n as i64;
        for i in (0..n).rev() {
            if valid[i] {
                bestr = bestr.min(i as i64);
            }
            right[i] = bestr;
        }
        let snapshot: Vec<f64> = hb[off..off + n].to_vec();
        for i in 0..n {
            if !valid[i] {
                nbad += 1;
                let mut s = if i >= m.i0 { left[i] } else { right[i] };
                if s < 0 || s >= n as i64 {
                    s = m.i0 as i64;
                }
                hb[off + i] = snapshot[s as usize];
            }
        }
        for i in 0..n {
            let mut v = hb[off + i];
            if v.is_nan() {
                v = m.nu;
            }
            v = v.max(m.nu).min(m.hmax[i]);
            if m.chi[i] == 0.0 {
                v = m.nu;
            }
            hb[off + i] = v;
        }
    }
    (hb, nbad)
}

/// Economy P: one-dimensional backward Euler with the tabulated hazard hbar(t_n, u_i).
#[allow(clippy::too_many_arguments)]
pub fn ts_p(m: &Model, ts: &[f64], hbar: &[f64], f: f64, nf: usize, store: Store, compare: Option<(&[f64], &[f64])>) -> Stepped {
    let n = m.n_u;
    let nsteps = hbar.len() / n;
    let mut q = vec![0.0; n];
    q[m.i0] = 1.0;
    let mut m1 = if nf > 1 { vec![0.0; n] } else { vec![] };
    let mut m2 = if nf > 2 { vec![0.0; n] } else { vec![] };
    let r0 = vec![0.0; n];
    let mut out = SweepOut::new(n);
    let coef = m.ucoef(f);
    let mut acc = Acc::new(n);
    let mut surv = Vec::with_capacity(nsteps);
    let mut dens = Vec::with_capacity(nsteps);
    let mut st = Stored::default();
    let (mut mdp, mut mdc) = (0.0f64, 0.0f64);
    for k in 0..nsteps {
        let dt = ts[k + 1] - ts[k];
        let s = 1.0 / dt;
        let hrow = &hbar[k * n..(k + 1) * n];
        m.sweep(&coef, s, s, 1, &r0, hrow, &m.w1, &m.w2, &mut q,
                if nf > 1 { Some(&mut m1) } else { None }, if nf > 2 { Some(&mut m2) } else { None }, nf, &mut out);
        store_row(&mut st, &store, &out);
        let sp = acc.add(&m.u, dt, &out, nf);
        surv.push(sp);
        dens.push(np_sum(&out.c));
        if let Some((cp, cc)) = compare {
            for i in 0..n {
                mdp = mdp.max((out.p[i] - cp[k * n + i]).abs());
                mdc = mdc.max((out.c[i] - cc[k * n + i]).abs());
            }
        }
    }
    let mass_left = *surv.last().unwrap_or(&0.0);
    Stepped {
        etau: acc.etau, ei: acc.ei, eu2int: acc.eu2int, eiu: acc.eiu, ei2: acc.ei2, ei2_alt: acc.ei2_alt,
        occ: acc.occ, rev: acc.rev, surv, dens, nsteps, ts: ts[..nsteps + 1].to_vec(), mass_left, store: st,
        snaps: vec![], maxdiff_p: mdp, maxdiff_c: mdc,
    }
}

/// S_P by a central difference of M_P(f) with hbar held fixed.
pub fn sp_fd(m: &Model, ts: &[f64], hbar: &[f64], delta: f64) -> (f64, f64, f64) {
    let rp = ts_p(m, ts, hbar, delta, 1, Store::none(), None);
    let rm = ts_p(m, ts, hbar, -delta, 1, Store::none(), None);
    let mp = rp.ei / rp.etau;
    let mm = rm.ei / rm.etau;
    ((mp - mm) / (2.0 * delta), mp, mm)
}

/// Summary statistics from the occupation measure and the completion (revision) measure.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Stats {
    pub etau: f64,
    pub f: f64,
    pub var_u: f64,
    pub ed2: f64,
    pub ed4: f64,
    pub kurt: f64,
    pub alo: f64,
    pub var_u_formula: f64,
    pub rev_mass: f64,
    pub s_formula: Option<f64>,
    pub eiu: Option<f64>,
    pub ei2: Option<f64>,
    pub ei2_alt: Option<f64>,
    pub s_girsanov_split: Option<f64>,
}

/// Moments needed by `stats`.
pub struct MomentSource<'a> {
    pub etau: f64,
    pub ei: f64,
    pub eu2int: f64,
    pub rev: &'a [f64],
    pub eiu: Option<f64>,
    pub ei2: Option<f64>,
    pub ei2_alt: Option<f64>,
}

impl Resolvent {
    pub fn source(&self, nf: usize) -> MomentSource<'_> {
        MomentSource { etau: self.etau, ei: self.ei, eu2int: self.eu2int, rev: &self.rev,
                       eiu: if nf > 1 { Some(self.eiu) } else { None }, ei2: if nf > 2 { Some(self.ei2) } else { None },
                       ei2_alt: if nf > 2 { Some(self.ei2_alt) } else { None } }
    }
}

impl Stepped {
    pub fn source(&self, nf: usize) -> MomentSource<'_> {
        MomentSource { etau: self.etau, ei: self.ei, eu2int: self.eu2int, rev: &self.rev,
                       eiu: if nf > 1 { Some(self.eiu) } else { None }, ei2: if nf > 2 { Some(self.ei2) } else { None },
                       ei2_alt: if nf > 2 { Some(self.ei2_alt) } else { None } }
    }
}

pub fn stats_from(m: &Model, r: &MomentSource) -> Stats {
    let u = &m.u;
    let etau = r.etau;
    let f = 1.0 / etau;
    let var = r.eu2int / etau - (r.ei / etau).powi(2);
    let tot = np_sum(r.rev);
    let d2 = wsum2(u, r.rev) / tot;
    let u4: Vec<f64> = u.iter().zip(r.rev).map(|(x, w)| x.powi(4) * w).collect();
    let d4 = np_sum(&u4) / tot;
    let kurt = d4 / (d2 * d2);
    let s2 = m.sig * m.sig;
    let mut out = Stats {
        etau, f, var_u: var / s2, ed2: d2, ed4: d4, kurt, alo: kurt / (6.0 * f),
        var_u_formula: if m.lam > 0.0 { (s2 - f * d2) / (2.0 * m.lam) / s2 } else { f64::NAN },
        rev_mass: tot, ..Default::default()
    };
    if let (Some(eiu), Some(ei2)) = (r.eiu, r.ei2) {
        out.s_formula = Some(s_from_moments(eiu, ei2, etau, m.lam, m.sig));
        out.eiu = Some(eiu);
        out.ei2 = Some(ei2);
        out.ei2_alt = r.ei2_alt;
        out.s_girsanov_split = Some(var / s2 + m.lam * ei2 / (2.0 * s2 * etau));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// theta = 0, eta = 0 is the Calvo rule: E tau = 1 / nu, and S = 1 / (lam + nu).
    #[test]
    fn calvo_limit() {
        let lam = 1.0;
        let nu = 1.0;
        let l = 6.5;
        let m = Model::new(lam, 0.0, 0.0, nu, l, util::odd(2.0 * l / 0.02), 4);
        let r = resolvent_b(&m, 0.0, 3);
        assert!((r.etau - 1.0 / nu).abs() < 1e-10, "E tau = {}", r.etau);
        let s = stats_from(&m, &r.source(3));
        let sf = s.s_formula.unwrap();
        assert!((sf - 1.0 / (lam + nu)).abs() < 2e-4, "S = {sf}");
        let (sfd, _, _) = sb_fd(&m, 0.002);
        assert!((sfd - 1.0 / (lam + nu)).abs() < 2e-4, "S fd = {sfd}");
    }
}

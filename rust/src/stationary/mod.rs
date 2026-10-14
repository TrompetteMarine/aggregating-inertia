//! Finite-volume forward generator L* of the stationary population on (u, a). Port of the reference
//! `fvsolver.py` and `march.c`.
//!
//! Model (push f frozen): dU = (-lam U + f) dt + sig dW, dA = alpha(U) dt; completions at hazard
//! H(u, a) = nu + eta chi(u) (1 - exp(-gam a)), after which (U, A) -> (R(U), 0); alpha = chi = act_theta.
//!
//! Discretisation (unknowns are cell masses m[j, i], j = a-cell, i = u-cell, a-major ordering):
//! * u: uniform symmetric cells on [-Umax, Umax], Scharfetter-Gummel drift-diffusion fluxes, zero
//!   flux at the ends;
//! * a: cells [a_j, a_{j+1}] exponentially stretched (fine near a = 0) up to Amax, plus one last cell
//!   lumping a >= Amax; first-order upwind transport at rate alpha(u_i) / da_j;
//! * killing at rate H (cell average of 1 - exp(-gam a) over each finite a-cell);
//! * reinjection of the killed mass of cell (i, j) at (R(u_i), a-cell 0), split between the two
//!   u-cells that bracket R(u_i) (conserves mass and first moment).
//!
//! L = K + E_T Q_T: K is block lower bidiagonal in a with tridiagonal blocks (forward marching) and the
//! reinjection E_T Q_T has rank n_T, the number of u-cells receiving mass. Solves with L - z use the
//! Woodbury identity with an n_T x n_T capacitance matrix: a direct method. The model is symmetric
//! under u -> -u, so L splits into an even sector (the stationary density, eigenvalue 0) and an odd
//! sector (b = -d_u pi, the response kernel and all mean-field instabilities), each on the half grid.

pub mod analysis;
pub mod arnoldi;

use crate::util::{self, quad::leggauss, np_sum};
use nalgebra::{ComplexField, DMatrix, DVector};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

/// Reset rule after a completion.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResetKind {
    /// full reset R = 0
    Zero,
    /// capped (piecewise linear) partial reset R(u) = sgn(u) min(|u|, c)
    Pl,
    /// smooth partial reset R(u) = sgn(u) int_0^{|u|} rho
    S,
}

impl ResetKind {
    pub fn code(&self) -> &'static str {
        match self {
            ResetKind::Zero => "0",
            ResetKind::Pl => "pl",
            ResetKind::S => "s",
        }
    }
    pub fn parse(s: &str) -> Self {
        match s {
            "0" => ResetKind::Zero,
            "pl" => ResetKind::Pl,
            "s" => ResetKind::S,
            _ => panic!("unknown reset {s}"),
        }
    }
}

fn gl80() -> &'static (Vec<f64>, Vec<f64>) {
    static GL: OnceLock<(Vec<f64>, Vec<f64>)> = OnceLock::new();
    GL.get_or_init(|| leggauss(80))
}

/// int_0^z (1 - s(t)) dt for z in [0, 1] (composite Gauss-Legendre, four panels of 80 points).
fn phi_int(z: f64) -> f64 {
    if z <= 0.0 {
        return 0.0;
    }
    if z >= 1.0 {
        return 0.5;
    }
    let (gx, gw) = gl80();
    let edges = util::linspace(0.0, z, 5);
    let mut tot = 0.0;
    for k in 0..4 {
        let (lo, hi) = (edges[k], edges[k + 1]);
        let mut acc = 0.0;
        for (x, w) in gx.iter().zip(gw) {
            let t = 0.5 * (hi - lo) * x + 0.5 * (hi + lo);
            acc += w * (1.0 - util::smoothstep_expit(t));
        }
        tot += 0.5 * (hi - lo) * acc;
    }
    tot
}

#[derive(Clone, Copy, Debug)]
pub struct Reset {
    pub kind: ResetKind,
    pub c: f64,
    b: f64,
    s0: f64,
}

impl Reset {
    pub fn new(kind: ResetKind, c: f64) -> Self {
        Reset { kind, c, b: 1.6 * c, s0: 0.4 * c }
    }
    pub fn apply(&self, u: f64) -> f64 {
        if self.kind == ResetKind::Zero || self.c == 0.0 {
            return 0.0;
        }
        let x = u.abs();
        let sgn = if u > 0.0 { 1.0 } else if u < 0.0 { -1.0 } else { 0.0 };
        match self.kind {
            ResetKind::Pl => sgn * x.min(self.c),
            ResetKind::S => {
                let v = if x <= self.s0 {
                    x
                } else if x < self.b {
                    self.s0 + (self.b - self.s0) * phi_int((x - self.s0) / (self.b - self.s0))
                } else {
                    self.c
                };
                sgn * v
            }
            ResetKind::Zero => 0.0,
        }
    }
}

/// Bernoulli function B(x) = x / (e^x - 1).
#[inline]
pub fn bern(x: f64) -> f64 {
    if x.abs() > 1e-12 { x / x.exp_m1() } else { 1.0 }
}

/// B'(x).
#[inline]
pub fn dbern(x: f64) -> f64 {
    if x.abs() < 1e-2 {
        -0.5 + x / 6.0 - x.powi(3) / 180.0 + x.powi(5) / 5040.0
    } else {
        let em = x.exp_m1();
        (em - x * (em + 1.0)) / (em * em)
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Params {
    pub theta: f64,
    pub eta: f64,
    pub nu: f64,
    pub lam: f64,
    pub gam: f64,
    pub sig: f64,
    pub reset: ResetKind,
    pub c: f64,
}

impl Params {
    /// Parameters with sig = 1 and the reset cap c = theta / 2.
    pub fn new(theta: f64, eta: f64, nu: f64, lam: f64, gam: f64, reset: ResetKind) -> Self {
        Params { theta, eta, nu, lam, gam, sig: 1.0, reset, c: theta / 2.0 }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct GridSpec {
    pub umax: f64,
    /// number of u-cells (even)
    pub n_u: usize,
    /// number of finite a-cells (0: a single lumped cell, only when eta = 0)
    pub na_f: usize,
    pub amax: f64,
    pub a_s: f64,
}

/// Heuristic domain: Umax covers 2 theta + 3 and the stationary tail; Amax = 20 / gam.
pub fn default_grid(p: &Params, h: f64, na_f: usize) -> GridSpec {
    let tail = if p.lam > 0.0 { 5.0 * p.sig / p.lam.sqrt() } else { 25.0 * p.sig / (2.0 * p.nu).sqrt() };
    let umax = (2.0 * p.theta + 3.0).max(tail);
    let n_u = 2 * (umax / h).ceil() as usize;
    let umax = n_u as f64 * h / 2.0;
    if p.eta == 0.0 {
        return GridSpec { umax, n_u, na_f: 0, amax: 0.0, a_s: 1.0 };
    }
    let amax = 20.0 / p.gam;
    let a_s = (0.5 * (1.0 / p.gam).min(1.0 / (p.eta * p.gam).sqrt()).min(1.0 / (p.nu + p.eta))).max(0.02);
    GridSpec { umax, n_u, na_f, amax, a_s }
}

/// Discretised operator pieces on the full grid.
#[derive(Clone, Debug)]
pub struct Disc {
    pub p: Params,
    pub g: GridSpec,
    pub f: f64,
    pub uf: Vec<f64>,
    pub uc: Vec<f64>,
    pub du: Vec<f64>,
    pub n_u: usize,
    pub na: usize,
    pub af: Vec<f64>,
    pub da: Vec<f64>,
    pub g1: Vec<f64>,
    pub alpha: Vec<f64>,
    /// hazard, Na x Nu
    pub h: Vec<f64>,
    pub out: Vec<f64>,
    pub ain: Vec<f64>,
    pub dr_r: Vec<f64>,
    pub dr_l: Vec<f64>,
    pub lo: Vec<f64>,
    pub up: Vec<f64>,
    pub dgu: Vec<f64>,
    pub rv: Vec<f64>,
    /// reset interpolation by source cell: (target cell, weight)
    pub w: Vec<Vec<(usize, f64)>>,
}

impl Disc {
    pub fn new(p: &Params, g: &GridSpec, f: f64) -> Self {
        assert!(g.n_u % 2 == 0);
        if g.na_f == 0 {
            assert!(p.eta == 0.0, "single lumped a-cell only valid for eta = 0");
        }
        let half = util::linspace(0.0, g.umax, g.n_u / 2 + 1);
        let mut uf: Vec<f64> = half.iter().rev().map(|x| -x).collect();
        uf.extend_from_slice(&half[1..]);
        let n_u = g.n_u;
        let uc: Vec<f64> = (0..n_u).map(|i| 0.5 * (uf[i + 1] + uf[i])).collect();
        let du: Vec<f64> = (0..n_u).map(|i| uf[i + 1] - uf[i]).collect();
        let (af, da) = if g.na_f == 0 {
            (vec![0.0], vec![])
        } else {
            let kappa = (g.amax / g.a_s).ln_1p();
            let s = util::linspace(0.0, 1.0, g.na_f + 1);
            let mut af: Vec<f64> = s.iter().map(|&x| g.a_s * (kappa * x).exp_m1()).collect();
            af[0] = 0.0;
            *af.last_mut().unwrap() = g.amax;
            let da: Vec<f64> = (0..g.na_f).map(|j| af[j + 1] - af[j]).collect();
            (af, da)
        };
        let na = da.len() + 1;
        let gam = p.gam;
        let mut g1 = vec![0.0; na];
        if !da.is_empty() {
            for j in 0..na - 1 {
                let a0 = af[j];
                g1[j] = 1.0 - (-gam * a0).exp() * (-(-gam * da[j]).exp_m1()) / (gam * da[j]);
            }
            g1[na - 1] = -(-gam * af[af.len() - 1]).exp_m1();
        } else {
            g1[na - 1] = 1.0;
        }
        let alpha: Vec<f64> = uc.iter().map(|&x| util::activation_expit(x, p.theta)).collect();
        let mut h = vec![0.0; na * n_u];
        let mut out = vec![0.0; na * n_u];
        let mut ain = vec![0.0; na * n_u];
        for j in 0..na {
            for i in 0..n_u {
                h[j * n_u + i] = p.nu + p.eta * alpha[i] * g1[j];
            }
        }
        if !da.is_empty() {
            for j in 0..na - 1 {
                for i in 0..n_u {
                    out[j * n_u + i] = alpha[i] / da[j];
                    ain[(j + 1) * n_u + i] = alpha[i] / da[j];
                }
            }
        }
        let d = 0.5 * p.sig * p.sig;
        let mut r_r = vec![0.0; n_u - 1];
        let mut r_l = vec![0.0; n_u - 1];
        let mut dr_r = vec![0.0; n_u - 1];
        let mut dr_l = vec![0.0; n_u - 1];
        for k in 0..n_u - 1 {
            let hk = uc[k + 1] - uc[k];
            let v = -p.lam * uf[k + 1] + f;
            let pe = v * hk / d;
            r_r[k] = (d / hk) * bern(-pe) / du[k];
            r_l[k] = (d / hk) * bern(pe) / du[k + 1];
            dr_r[k] = -dbern(-pe) / du[k];
            dr_l[k] = dbern(pe) / du[k + 1];
        }
        let mut lo = vec![0.0; n_u];
        let mut up = vec![0.0; n_u];
        let mut dgu = vec![0.0; n_u];
        for k in 0..n_u - 1 {
            lo[k + 1] = r_r[k];
            up[k] = r_l[k];
            dgu[k] -= r_r[k];
            dgu[k + 1] -= r_l[k];
        }
        let rfun = Reset::new(p.reset, p.c);
        let rv: Vec<f64> = uc.iter().map(|&x| rfun.apply(x)).collect();
        let w = interp_weights(&uc, &rv);
        Disc { p: *p, g: *g, f, uf, uc, du, n_u, na, af, da, g1, alpha, h, out, ain, dr_r, dr_l, lo, up, dgu, rv, w }
    }

    /// (dDu / df) m for m of shape (Na, Nu).
    pub fn ddu_apply(&self, m: &[f64]) -> Vec<f64> {
        let n = self.n_u;
        let mut o = vec![0.0; m.len()];
        for j in 0..self.na {
            let row = &m[j * n..(j + 1) * n];
            let orow = &mut o[j * n..(j + 1) * n];
            for k in 0..n - 1 {
                orow[k + 1] += self.dr_r[k] * row[k];
            }
            for k in 0..n - 1 {
                orow[k] += self.dr_l[k] * row[k + 1];
            }
            for k in 0..n - 1 {
                orow[k] -= self.dr_r[k] * row[k];
            }
            for k in 0..n - 1 {
                orow[k + 1] -= self.dr_l[k] * row[k + 1];
            }
        }
        o
    }
}

/// Linear interpolation weights placing the mass of source cell i at position t[i]
/// (scipy-free version of the reference `_interp_matrix`).
fn interp_weights(uc: &[f64], t: &[f64]) -> Vec<Vec<(usize, f64)>> {
    let n = uc.len();
    let mut w = Vec::with_capacity(n);
    for &ti in t.iter() {
        let mut k = uc.partition_point(|&x| x <= ti) as i64 - 1;
        k = k.clamp(0, n as i64 - 2);
        let k = k as usize;
        let w0 = ((uc[k + 1] - ti) / (uc[k + 1] - uc[k])).clamp(0.0, 1.0);
        let mut v = vec![];
        if w0 != 0.0 {
            v.push((k, w0));
        }
        if 1.0 - w0 != 0.0 {
            v.push((k + 1, 1.0 - w0));
        }
        w.push(v);
    }
    w
}

/// A symmetry sector of the operator: sign None (full grid), +1 (even), -1 (odd, half grid u > 0).
#[derive(Clone, Debug)]
pub struct Sector {
    pub sign: Option<i32>,
    pub n: usize,
    pub na: usize,
    pub nn: usize,
    pub pos: Vec<usize>,
    pub lo: Vec<f64>,
    pub up: Vec<f64>,
    pub uw: Vec<f64>,
    pub mw: Vec<f64>,
    pub uc: Vec<f64>,
    pub h: Vec<f64>,
    pub ain: Vec<f64>,
    pub dgk: Vec<f64>,
    /// reinjection weights by source column: (target row, weight)
    pub wcols: Vec<Vec<(usize, f64)>>,
    pub targets: Vec<usize>,
    /// dense W restricted to the target rows, nT x n
    pub wt: Vec<f64>,
}

impl Sector {
    pub fn new(d: &Disc, sign: Option<i32>) -> Self {
        let (n_full, na) = (d.n_u, d.na);
        let hn = n_full / 2;
        let (pos, lo, up, dgu, wcols_raw, uw, mw): (Vec<usize>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<Vec<(usize, f64)>>, Vec<f64>, Vec<f64>);
        match sign {
            None => {
                pos = (0..n_full).collect();
                lo = d.lo.clone();
                up = d.up.clone();
                dgu = d.dgu.clone();
                wcols_raw = d.w.clone();
                uw = d.uc.clone();
                mw = vec![1.0; n_full];
            }
            Some(s) => {
                let sf = s as f64;
                pos = (hn..n_full).collect();
                let mir: Vec<usize> = (0..hn).map(|k| hn - 1 - k).collect();
                let mut l = pos.iter().map(|&i| d.lo[i]).collect::<Vec<_>>();
                l[0] = 0.0;
                lo = l;
                up = pos.iter().map(|&i| d.up[i]).collect();
                let mut dg: Vec<f64> = pos.iter().map(|&i| d.dgu[i]).collect();
                dg[0] += sf * d.lo[hn];
                dgu = dg;
                // W[pos][:, pos] + sign * W[pos][:, mir], by sector column ip
                let mut cols = vec![vec![]; hn];
                for ip in 0..hn {
                    let mut acc: Vec<(usize, f64)> = vec![];
                    let mut add = |t: usize, w: f64| {
                        if t >= hn {
                            let r = t - hn;
                            if let Some(e) = acc.iter_mut().find(|e| e.0 == r) {
                                e.1 += w;
                            } else {
                                acc.push((r, w));
                            }
                        }
                    };
                    for &(t, w) in &d.w[pos[ip]] {
                        add(t, w);
                    }
                    for &(t, w) in &d.w[mir[ip]] {
                        add(t, sf * w);
                    }
                    cols[ip] = acc;
                }
                wcols_raw = cols;
                uw = pos.iter().map(|&i| (1.0 - sf) * d.uc[i]).collect();
                mw = vec![1.0 + sf; hn];
            }
        }
        let n = pos.len();
        let mut h = vec![0.0; na * n];
        let mut ain = vec![0.0; na * n];
        let mut dgk = vec![0.0; na * n];
        for j in 0..na {
            for (k, &i) in pos.iter().enumerate() {
                h[j * n + k] = d.h[j * n_full + i];
                ain[j * n + k] = d.ain[j * n_full + i];
                dgk[j * n + k] = dgu[k] - d.h[j * n_full + i] - d.out[j * n_full + i];
            }
        }
        // drop zeros and entries below 1e-15
        let wcols: Vec<Vec<(usize, f64)>> = wcols_raw
            .into_iter()
            .map(|c| c.into_iter().filter(|&(_, w)| w != 0.0 && w.abs() >= 1e-15).collect())
            .collect();
        let mut targets: Vec<usize> = wcols.iter().flat_map(|c| c.iter().map(|e| e.0)).collect();
        targets.sort_unstable();
        targets.dedup();
        let nt = targets.len();
        let mut wt = vec![0.0; nt * n];
        for (ip, c) in wcols.iter().enumerate() {
            for &(t, w) in c {
                let a = targets.binary_search(&t).unwrap();
                wt[a * n + ip] += w;
            }
        }
        let uc = pos.iter().map(|&i| d.uc[i]).collect();
        Sector { sign, n, na, nn: n * na, pos, lo, up, uw, mw, uc, h, ain, dgk, wcols, targets, wt }
    }

    pub fn nt(&self) -> usize {
        self.targets.len()
    }

    /// y = L x (the sector operator, including reinjection).
    pub fn matvec(&self, x: &[f64]) -> Vec<f64> {
        let (n, na) = (self.n, self.na);
        let mut y = vec![0.0; self.nn];
        for j in 0..na {
            let o = j * n;
            for i in 0..n {
                let mut v = self.dgk[o + i] * x[o + i];
                if i > 0 {
                    v += self.lo[i] * x[o + i - 1];
                }
                if i + 1 < n {
                    v += self.up[i] * x[o + i + 1];
                }
                if j > 0 {
                    v += self.ain[o + i] * x[o - n + i];
                }
                y[o + i] = v;
            }
        }
        if !self.targets.is_empty() {
            let hx = self.hazard_sum(x);
            for (ip, c) in self.wcols.iter().enumerate() {
                for &(t, w) in c {
                    y[t] += w * hx[ip];
                }
            }
        }
        y
    }

    /// sum_j H[j, i] x[j, i]
    pub fn hazard_sum<T: ComplexField<RealField = f64> + Copy>(&self, x: &[T]) -> Vec<T> {
        let n = self.n;
        let mut s = vec![T::zero(); n];
        for j in 0..self.na {
            for i in 0..n {
                s[i] += T::from_real(self.h[j * n + i]) * x[j * n + i];
            }
        }
        s
    }

    pub fn inner_u(&self, x: &[f64]) -> f64 {
        let w: Vec<f64> = (0..self.nn).map(|k| self.uw[k % self.n] * x[k]).collect();
        np_sum(&w)
    }

    pub fn inner_u_c(&self, x: &[Complex64]) -> Complex64 {
        let mut s = Complex64::new(0.0, 0.0);
        for k in 0..self.nn {
            s += x[k] * self.uw[k % self.n];
        }
        s
    }

    pub fn mass(&self, x: &[f64]) -> f64 {
        let w: Vec<f64> = (0..self.nn).map(|k| self.mw[k % self.n] * x[k]).collect();
        np_sum(&w)
    }
}

/// Direct solver for (L_sector - z I) x = y: block marching plus Woodbury (rank n_T).
pub struct Shifted<T: ComplexField<RealField = f64> + Copy> {
    pub z: T,
    mlt: Vec<T>,
    winv: Vec<T>,
    /// Q_T K^{-1} E_T, nT x nT
    pub qz: Option<DMatrix<T>>,
    lu: Option<nalgebra::LU<T, nalgebra::Dyn, nalgebra::Dyn>>,
}

impl<T: ComplexField<RealField = f64> + Copy> Shifted<T> {
    pub fn new(sec: &Sector, z: T, singular_ok: bool) -> Self {
        let (n, na) = (sec.n, sec.na);
        let mut mlt = vec![T::zero(); na * n];
        let mut winv = vec![T::zero(); na * n];
        for j in 0..na {
            let o = j * n;
            let d = |i: usize| T::from_real(sec.dgk[o + i]) - z;
            let mut w = d(0);
            winv[o] = T::one() / w;
            mlt[o] = T::zero();
            for i in 1..n {
                let m = T::from_real(sec.lo[i]) * winv[o + i - 1];
                mlt[o + i] = m;
                w = d(i) - m * T::from_real(sec.up[i - 1]);
                winv[o + i] = T::one() / w;
            }
        }
        let mut sh = Shifted { z, mlt, winv, qz: None, lu: None };
        let nt = sec.nt();
        if nt > 0 {
            // S[k, i] = sum_j H[j, i] x_k[j, i], x_k = K^{-1} e_(0, target k)
            let mut s = vec![T::zero(); nt * n];
            let mut y = vec![T::zero(); sec.nn];
            for (k, &t) in sec.targets.iter().enumerate() {
                y.iter_mut().for_each(|v| *v = T::zero());
                y[t] = T::one();
                let x = sh.march(sec, &y);
                let hs = sec.hazard_sum(&x);
                s[k * n..(k + 1) * n].copy_from_slice(&hs);
            }
            let mut qz = DMatrix::<T>::zeros(nt, nt);
            for a in 0..nt {
                for k in 0..nt {
                    let mut acc = T::zero();
                    for i in 0..n {
                        acc += T::from_real(sec.wt[a * n + i]) * s[k * n + i];
                    }
                    qz[(a, k)] = acc;
                }
            }
            if !singular_ok {
                let m = DMatrix::<T>::identity(nt, nt) + &qz;
                sh.lu = Some(m.lu());
            }
            sh.qz = Some(qz);
        }
        sh
    }

    /// x = K_z^{-1} y by forward marching in a with a Thomas solve per block.
    pub fn march(&self, sec: &Sector, y: &[T]) -> Vec<T> {
        let (n, na) = (sec.n, sec.na);
        let mut x = vec![T::zero(); na * n];
        for j in 0..na {
            let o = j * n;
            if j == 0 {
                x[0] = y[0];
                for i in 1..n {
                    x[i] = y[i] - self.mlt[i] * x[i - 1];
                }
            } else {
                x[o] = y[o] - T::from_real(sec.ain[o]) * x[o - n];
                for i in 1..n {
                    x[o + i] = (y[o + i] - T::from_real(sec.ain[o + i]) * x[o - n + i]) - self.mlt[o + i] * x[o + i - 1];
                }
            }
            x[o + n - 1] *= self.winv[o + n - 1];
            for i in (0..n - 1).rev() {
                x[o + i] = (x[o + i] - T::from_real(sec.up[i]) * x[o + i + 1]) * self.winv[o + i];
            }
        }
        x
    }

    pub fn solve(&self, sec: &Sector, y: &[T]) -> Vec<T> {
        let x1 = self.march(sec, y);
        let nt = sec.nt();
        if nt == 0 {
            return x1;
        }
        let hs = sec.hazard_sum(&x1);
        let n = sec.n;
        let r: Vec<T> = (0..nt)
            .map(|a| {
                let mut acc = T::zero();
                for i in 0..n {
                    acc += T::from_real(sec.wt[a * n + i]) * hs[i];
                }
                acc
            })
            .collect();
        let v = self.lu.as_ref().expect("factorisation").solve(&DVector::from_column_slice(&r)).expect("singular capacitance");
        let mut y2 = y.to_vec();
        for (a, &t) in sec.targets.iter().enumerate() {
            y2[t] -= v[a];
        }
        self.march(sec, &y2)
    }
}

/// Information about the stationary solve.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct StatInfo {
    pub p_colsum_err: f64,
    pub pmin: f64,
}

/// Stationary masses on a sector containing the zero eigenvalue (full or even), from the
/// reset-to-reset chain P = -Q_T K^{-1} E_T (column stochastic on the target cells).
pub fn stationary(sec: &Sector) -> (Vec<f64>, StatInfo) {
    let sh = Shifted::<f64>::new(sec, 0.0, true);
    let qz = sh.qz.as_ref().expect("a sector with reinjection");
    let nt = sec.nt();
    let p = -qz.clone();
    let mut colsum_err: f64 = 0.0;
    for k in 0..nt {
        let s: f64 = (0..nt).map(|a| p[(a, k)]).sum();
        colsum_err = colsum_err.max((s - 1.0).abs());
    }
    let pmin = p.iter().cloned().fold(f64::INFINITY, f64::min);
    let mut a = DMatrix::<f64>::identity(nt, nt) - &p;
    for k in 0..nt {
        a[(nt - 1, k)] = 1.0;
    }
    let mut rhs = vec![0.0; nt];
    rhs[nt - 1] = 1.0;
    let w = util::linalg::solve_dense(&a, &rhs);
    let mut y = vec![0.0; sec.nn];
    for (k, &t) in sec.targets.iter().enumerate() {
        y[t] = w[k];
    }
    let mut pi: Vec<f64> = sh.march(sec, &y).into_iter().map(|v| -v).collect();
    let mass = sec.mass(&pi);
    pi.iter_mut().for_each(|v| *v /= mass);
    (pi, StatInfo { p_colsum_err: colsum_err, pmin })
}

/// Stationary statistics.
#[derive(Clone, Debug, Serialize)]
pub struct Stats {
    pub mass: f64,
    pub mean: f64,
    pub var: f64,
    #[serde(rename = "F")]
    pub f: f64,
    pub kurt: f64,
    pub kurt_over_6f: f64,
    pub share_nonzero: f64,
    pub lump_mass: f64,
    pub edge_mass: f64,
    pub a0_mass: f64,
    pub k0: f64,
    pub p_colsum_err: f64,
}

/// Everything for one parameter point on one grid.
pub struct Model {
    pub p: Params,
    pub g: GridSpec,
    pub d: Disc,
    pub even: Sector,
    pub odd: Sector,
    pub stat_info: StatInfo,
    pub pi_half: Vec<f64>,
    /// stationary masses on the full grid, Na x Nu
    pub pi: Vec<f64>,
    /// b = (dDu/df) pi restricted to the odd sector
    pub b: Vec<f64>,
    sh0: OnceLock<Shifted<f64>>,
}

impl Model {
    pub fn new(p: Params, g: GridSpec) -> Self {
        let d = Disc::new(&p, &g, 0.0);
        let even = Sector::new(&d, Some(1));
        let odd = Sector::new(&d, Some(-1));
        let (pi_h, info) = stationary(&even);
        let hn = d.n_u / 2;
        let (na, nf) = (d.na, d.n_u);
        let mut pi = vec![0.0; na * nf];
        for j in 0..na {
            for k in 0..hn {
                pi[j * nf + hn + k] = pi_h[j * hn + k];
                pi[j * nf + hn - 1 - k] = pi_h[j * hn + k];
            }
        }
        let bfull = d.ddu_apply(&pi);
        let mut b = vec![0.0; na * hn];
        for j in 0..na {
            for k in 0..hn {
                b[j * hn + k] = bfull[j * nf + hn + k];
            }
        }
        Model { p, g, d, even, odd, stat_info: info, pi_half: pi_h, pi, b, sh0: OnceLock::new() }
    }

    pub fn stats(&self) -> Stats {
        let (d, pi) = (&self.d, &self.pi);
        let (na, n) = (d.na, d.n_u);
        let u = &d.uc;
        let ui = |k: usize| u[k % n];
        let mass = np_sum(pi);
        let mean = np_sum(&(0..pi.len()).map(|k| pi[k] * ui(k)).collect::<Vec<_>>());
        let var = np_sum(&(0..pi.len()).map(|k| pi[k] * ui(k) * ui(k)).collect::<Vec<_>>()) - mean * mean;
        let hpi: Vec<f64> = (0..pi.len()).map(|k| d.h[k] * pi[k]).collect();
        let f = np_sum(&hpi);
        let delta = |k: usize| d.rv[k % n] - ui(k);
        let m2 = np_sum(&(0..pi.len()).map(|k| hpi[k] * delta(k).powi(2)).collect::<Vec<_>>()) / f;
        let m4 = np_sum(&(0..pi.len()).map(|k| hpi[k] * delta(k).powi(4)).collect::<Vec<_>>()) / f;
        let kurt = m4 / (m2 * m2);
        let nonzero = np_sum(&(0..pi.len()).map(|k| if delta(k).abs() > 1e-12 { hpi[k] } else { 0.0 }).collect::<Vec<_>>()) / f;
        let lump = if na > 1 { np_sum(&pi[(na - 1) * n..]) } else { f64::NAN };
        let mut edge = 0.0;
        for j in 0..na {
            for i in (0..3).chain(n - 3..n) {
                edge += pi[j * n + i];
            }
        }
        Stats {
            mass, mean, var: var / (self.p.sig * self.p.sig), f, kurt, kurt_over_6f: kurt / (6.0 * f), share_nonzero: nonzero,
            lump_mass: lump, edge_mass: edge, a0_mass: np_sum(&pi[..n]), k0: self.odd.inner_u(&self.b),
            p_colsum_err: self.stat_info.p_colsum_err,
        }
    }

    pub fn sh0(&self) -> &Shifted<f64> {
        self.sh0.get_or_init(|| Shifted::new(&self.odd, 0.0, false))
    }

    /// S = <u, (-L)^{-1} b> on the odd sector.
    pub fn s_resolvent(&self) -> f64 {
        let y: Vec<f64> = self.b.iter().map(|v| -v).collect();
        let x = self.sh0().solve(&self.odd, &y);
        self.odd.inner_u(&x)
    }

    /// S by a central difference of the stationary average in f.
    pub fn s_fd(&self, delta: f64) -> f64 {
        let mut ms = [0.0; 2];
        for (k, f) in [delta, -delta].into_iter().enumerate() {
            let dfull = Disc::new(&self.p, &self.g, f);
            let (pif, _) = stationary(&Sector::new(&dfull, None));
            let n = dfull.n_u;
            ms[k] = np_sum(&(0..pif.len()).map(|i| pif[i] * dfull.uc[i % n]).collect::<Vec<_>>());
        }
        (ms[0] - ms[1]) / (2.0 * delta)
    }

    /// Laplace transform khat(z) = <u, (z - L)^{-1} b> at real z.
    pub fn khat_real(&self, z: f64) -> f64 {
        let x = if z == 0.0 { self.sh0().solve(&self.odd, &self.b) } else { Shifted::new(&self.odd, z, false).solve(&self.odd, &self.b) };
        -self.odd.inner_u(&x)
    }

    /// khat at complex z.
    pub fn khat(&self, z: Complex64) -> Complex64 {
        if z.im == 0.0 {
            return Complex64::new(self.khat_real(z.re), 0.0);
        }
        let sh = Shifted::<Complex64>::new(&self.odd, z, false);
        let bc: Vec<Complex64> = self.b.iter().map(|&v| Complex64::new(v, 0.0)).collect();
        let x = sh.solve(&self.odd, &bc);
        -self.odd.inner_u_c(&x)
    }

    /// k(t) = <u, e^{L t} b> by Crank-Nicolson with a Rannacher start (four implicit-Euler half steps).
    pub fn kernel(&self, t_end: f64, dt1: f64, t1: f64, dt2: f64, record_every: usize) -> (Vec<f64>, Vec<f64>) {
        let sec = &self.odd;
        let mut x = self.b.clone();
        let mut ts = vec![0.0];
        let mut ks = vec![sec.inner_u(&x)];
        let mut t = 0.0;
        let mut phases = vec![(dt1, t1.min(t_end))];
        if t_end > t1 {
            phases.push((dt2, t_end));
        }
        let mut first = true;
        for (dt, tend) in phases {
            let z = 2.0 / dt;
            let sh = Shifted::<f64>::new(sec, z, false);
            if first {
                for _ in 0..4 {
                    let y: Vec<f64> = x.iter().map(|v| -z * v).collect();
                    x = sh.solve(sec, &y);
                    t += dt / 2.0;
                    ts.push(t);
                    ks.push(sec.inner_u(&x));
                }
                first = false;
            }
            let nsteps = ((tend - t) / dt).round() as usize;
            for s in 0..nsteps {
                let lx = sec.matvec(&x);
                let y: Vec<f64> = x.iter().zip(&lx).map(|(a, b)| -(z * a + b)).collect();
                x = sh.solve(sec, &y);
                t += dt;
                if (s + 1) % record_every == 0 || s == nsteps - 1 {
                    ts.push(t);
                    ks.push(sec.inner_u(&x));
                }
            }
        }
        (ts, ks)
    }
}

/// theta = 0 with a full reset: exact renewal quantities by quadrature.
pub fn renewal_reference(eta: f64, nu: f64, lam: f64, gam: f64, sig: f64) -> serde_json::Value {
    use crate::util::quad::quad_inf;
    let surv = |a: f64| (-nu * a - eta * (a + (-gam * a).exp_m1() / gam)).exp();
    let haz = |a: f64| nu + eta * (-(-gam * a).exp_m1());
    let gf = |a: f64, r: f64| if r == 0.0 { a } else { -(-r * a).exp_m1() / r };
    let int = |f: &dyn Fn(f64) -> f64| quad_inf(f, 0.0, 1e-14, 1e-13, 500).0;
    let mu = int(&surv);
    let s = int(&|a| surv(a) * gf(a, lam)) / mu;
    let v = sig * sig * int(&|a| surv(a) * gf(a, 2.0 * lam)) / mu;
    let phi = |a: f64| haz(a) * surv(a);
    let m2 = sig * sig * int(&|a| phi(a) * gf(a, 2.0 * lam));
    let m4 = 3.0 * sig.powi(4) * int(&|a| phi(a) * gf(a, 2.0 * lam).powi(2));
    let kurt = m4 / (m2 * m2);
    serde_json::json!({"S": s, "var": v / (sig * sig), "F": 1.0 / mu, "kurt_over_6F": kurt * mu / 6.0})
}

/// theta = 0 with a full reset: k(t) = exp(-lam t) Psi(t), Psi the stationary residual-life survival.
pub fn renewal_kernel(eta: f64, nu: f64, lam: f64, ts: &[f64], gam: f64) -> Vec<f64> {
    use crate::util::quad::quad_inf;
    let surv = |a: f64| (-nu * a - eta * (a + (-gam * a).exp_m1() / gam)).exp();
    let mu = quad_inf(surv, 0.0, 1e-14, 1e-13, 500).0;
    ts.iter()
        .map(|&t| {
            let psi = quad_inf(surv, t, 1e-14, 1e-13, 500).0 / mu;
            (-lam * t).exp() * psi
        })
        .collect()
}


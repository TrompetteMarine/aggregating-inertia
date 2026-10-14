//! Kernel statistics, spectral gap, Nyquist curve, first crossing and the argument-principle
//! certificate. Port of the reference `analysis.py` (and `dominant_root_curve` of `make_data.py`).
//!
//! Conventions (odd sector): k(t) = <u, e^{L t} b>, khat(z) = <u, (z - L)^{-1} b>, khat(0) = S; the
//! characteristic equation 1 = g khat(z) holds iff z is an eigenvalue of L_odd + g b u^T. Since khat has
//! no poles in Re z >= 0 and vanishes at infinity, roots enter Re z > 0 only through the imaginary
//! axis, at g = 1 / khat(i w) where khat(i w) is real and positive.

use super::{arnoldi, Model};
use crate::util::{self, roots::{brentq, BrentOpts}};
use num_complex::Complex64;
use serde::Serialize;
use std::cell::RefCell;
use std::collections::HashMap;

#[derive(Clone, Debug, Serialize)]
pub struct KernelStats {
    #[serde(rename = "T")]
    pub t_end: f64,
    pub kmin: f64,
    pub tmin: f64,
    pub int_k: f64,
    pub int_t_absk: f64,
    pub int_absk: f64,
    #[serde(rename = "TV")]
    pub tv: f64,
    pub k_t: f64,
    pub k0: f64,
    pub first_neg_t: Option<f64>,
    pub neg_mass: f64,
}

/// The kernel on [0, T] and its statistics; T = min(600, max(40, 36 / gap)) unless given.
pub fn kernel_stats(model: &Model, t_end: Option<f64>, dt1: f64, t1: f64, dt2: f64, gap: Option<f64>) -> (KernelStats, Vec<f64>, Vec<f64>) {
    let t_end = t_end.unwrap_or_else(|| {
        let gap = gap.unwrap_or_else(|| slowest_rate(model));
        600f64.min(40f64.max(36.0 / gap))
    });
    let (t, k) = model.kernel(t_end, dt1, t1, dt2, 1);
    let i = util::argmin(&k);
    let n = t.len();
    let (mut integ, mut absmom1, mut absint, mut tv, mut negmass) = (vec![], vec![], vec![], vec![], vec![]);
    for s in 1..n {
        let dtv = t[s] - t[s - 1];
        integ.push(0.5 * (k[s] + k[s - 1]) * dtv);
        absmom1.push(0.5 * (t[s] * k[s].abs() + t[s - 1] * k[s - 1].abs()) * dtv);
        absint.push(0.5 * (k[s].abs() + k[s - 1].abs()) * dtv);
        tv.push((k[s] - k[s - 1]).abs());
        negmass.push(0.5 * (k[s].min(0.0) + k[s - 1].min(0.0)) * dtv);
    }
    let first_neg = k.iter().position(|&v| v < 0.0).map(|p| t[p]);
    let st = KernelStats {
        t_end, kmin: k[i], tmin: t[i], int_k: util::np_sum(&integ), int_t_absk: util::np_sum(&absmom1),
        int_absk: util::np_sum(&absint), tv: util::np_sum(&tv), k_t: k[n - 1], k0: k[0], first_neg_t: first_neg,
        neg_mass: util::np_sum(&negmass),
    };
    (st, t, k)
}

/// Spectral gap of L_odd: minus the real part of its rightmost eigenvalue (shift-invert at 0).
pub fn slowest_rate(model: &Model) -> f64 {
    let ev = arnoldi::rightmost_eigs(model, 0.0, 0.0, 6, 1e-9);
    -ev.iter().map(|e| e.lambda.re).fold(f64::NEG_INFINITY, f64::max)
}

/// Cached evaluation of w -> khat(i w).
pub struct KhatEvaluator<'a> {
    pub m: &'a Model,
    cache: RefCell<HashMap<u64, Complex64>>,
    pub nevals: RefCell<usize>,
}

impl<'a> KhatEvaluator<'a> {
    pub fn new(m: &'a Model) -> Self {
        KhatEvaluator { m, cache: RefCell::new(HashMap::new()), nevals: RefCell::new(0) }
    }
    pub fn eval(&self, w: f64) -> Complex64 {
        let key = w.to_bits();
        if let Some(v) = self.cache.borrow().get(&key) {
            return *v;
        }
        let z = if w != 0.0 { Complex64::new(0.0, w) } else { Complex64::new(0.0, 0.0) };
        let v = self.m.khat(z);
        self.cache.borrow_mut().insert(key, v);
        *self.nevals.borrow_mut() += 1;
        v
    }
}

fn angle(z: Complex64) -> f64 {
    z.im.atan2(z.re)
}

/// Samples w -> khat(i w) on [0, w_max], refining until consecutive points are within
/// rel_step * max(S, |khat|) of each other and the argument changes by less than 0.25 rad.
pub fn adaptive_curve(kh: &KhatEvaluator, s: f64, w_max: f64, n0: usize, rel_step: f64, max_pts: usize) -> (Vec<f64>, Vec<Complex64>) {
    let mut ws = vec![0.0];
    ws.extend(util::geomspace(1e-3, w_max, n0));
    let mut vals: Vec<Complex64> = ws.iter().map(|&w| kh.eval(w)).collect();
    let mut i = 0;
    while i < ws.len() - 1 && ws.len() < max_pts {
        let (a, b) = (vals[i], vals[i + 1]);
        let scale = a.norm().min(b.norm()).max(0.05 * s);
        let dang = if a != Complex64::new(0.0, 0.0) && b != Complex64::new(0.0, 0.0) { angle(b / a).abs() } else { std::f64::consts::PI };
        if (b - a).norm() > rel_step * s.max(scale) || dang > 0.25 {
            let wm = 0.5 * (ws[i] + ws[i + 1]);
            if ws[i + 1] - ws[i] < 1e-7 {
                i += 1;
                continue;
            }
            ws.insert(i + 1, wm);
            vals.insert(i + 1, kh.eval(wm));
        } else {
            i += 1;
        }
    }
    (ws, vals)
}

fn npsign(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 }
}

/// w > 0 where Im khat(i w) changes sign with Re khat > 0 on either side, refined by Brent.
pub fn positive_real_crossings(kh: &KhatEvaluator, ws: &[f64], vals: &[Complex64]) -> Vec<(f64, f64)> {
    let mut out = vec![];
    for i in 1..ws.len() - 1 {
        if ws[i] == 0.0 {
            continue;
        }
        let (a, b) = (vals[i].im, vals[i + 1].im);
        if a == 0.0 || npsign(a) != npsign(b) {
            if vals[i].re <= 0.0 && vals[i + 1].re <= 0.0 {
                continue;
            }
            let opts = BrentOpts { xtol: 1e-12, rtol: 1e-12, maxiter: 100 };
            match brentq(|w| kh.eval(w).im, ws[i], ws[i + 1], opts) {
                Ok(wc) => out.push((wc, kh.eval(wc).re)),
                Err(crate::util::roots::RootError::SameSign) => continue,
                Err(e) => panic!("{e}"),
            }
        }
    }
    out
}

#[derive(Clone, Debug, Serialize)]
pub struct FirstCrossing {
    #[serde(rename = "type")]
    pub kind: String,
    pub g: f64,
    pub omega: f64,
    #[serde(rename = "S")]
    pub s: f64,
    #[serde(rename = "invS")]
    pub inv_s: f64,
    pub crossings: Vec<(f64, f64)>,
    pub nevals: usize,
    pub max_abs_khat_w_pos: f64,
    pub min_re_khat: f64,
    #[serde(skip)]
    pub curve: (Vec<f64>, Vec<Complex64>),
}

pub fn first_crossing(model: &Model, w_max: f64, n0: usize) -> FirstCrossing {
    let kh = KhatEvaluator::new(model);
    let s = kh.eval(0.0).re;
    let (ws, vals) = adaptive_curve(&kh, s, w_max, n0, 0.03, 1500);
    let cr = positive_real_crossings(&kh, &ws, &vals);
    let best = cr.iter().filter(|(_, x)| *x > 0.0).cloned().fold(None, |acc: Option<(f64, f64)>, c| match acc {
        Some(b) if b.1 >= c.1 => Some(b),
        _ => Some(c),
    });
    let (kind, g, omega) = match best {
        Some((w, x)) if x > s => ("oscillatory".to_string(), 1.0 / x, w),
        _ => ("real".to_string(), 1.0 / s, 0.0),
    };
    let max_abs = vals[1..].iter().map(|v| v.norm()).fold(0.0, f64::max);
    let min_re = vals.iter().map(|v| v.re).fold(f64::INFINITY, f64::min);
    let nevals = *kh.nevals.borrow();
    FirstCrossing { kind, g, omega, s, inv_s: 1.0 / s, crossings: cr, nevals, max_abs_khat_w_pos: max_abs, min_re_khat: min_re, curve: (ws, vals) }
}

#[derive(Clone, Debug, Serialize)]
pub struct Certificate {
    pub g: f64,
    #[serde(rename = "N")]
    pub n: f64,
    #[serde(rename = "N_round")]
    pub n_round: i64,
    #[serde(rename = "Omega")]
    pub omega: f64,
    #[serde(rename = "M1")]
    pub m1: f64,
    #[serde(rename = "B")]
    pub b: f64,
    pub nsteps: usize,
    #[serde(rename = "min_absF")]
    pub min_abs_f: f64,
    pub tail_bound: f64,
    #[serde(rename = "F0")]
    pub f0: f64,
}

/// Number of zeros of F(z) = 1 - g khat(z) in Re z > 0 by the argument principle on the imaginary
/// axis, with step control |d khat(i w)/dw| <= M1 := int t |k| and |khat(i w)| <= (k(0) + TV(k)) / w
/// (bounds from the time-stepped kernel times a safety factor).
pub fn certify_count(model: &Model, g: f64, ks: &KernelStats, w_cap: Option<f64>, safety: f64, max_evals: usize) -> Result<Certificate, String> {
    let m1 = safety * ks.int_t_absk;
    let bb = safety * (ks.k0.abs() + ks.tv);
    let omega = w_cap.unwrap_or(safety * g * bb);
    let kh = KhatEvaluator::new(model);
    let f = |w: f64| Complex64::new(1.0, 0.0) - kh.eval(w) * g;
    let mut w = 0.0;
    let f0 = f(0.0);
    if f0.im.abs() >= 1e-12 {
        return Err("F(0) not real".into());
    }
    let theta0 = if f0.re > 0.0 { 0.0 } else { std::f64::consts::PI };
    let mut theta = theta0;
    let mut fprev = f0;
    let mut minabs = f0.norm();
    let mut nsteps = 0;
    while w < omega {
        let step = 0.5 * fprev.norm() / (g * m1);
        if step < 1e-9 {
            return Err(format!("step underflow at w={w}: |F|={} (g too close to a crossing)", fprev.norm()));
        }
        let wn = (w + step).min(omega);
        let fnew = f(wn);
        theta += angle(fnew / fprev);
        minabs = minabs.min(fnew.norm());
        w = wn;
        fprev = fnew;
        nsteps += 1;
        if nsteps > max_evals {
            return Err("too many steps".into());
        }
    }
    let tail_bound = g * bb / omega;
    if tail_bound >= 1.0 {
        return Err("tail bound >= 1".into());
    }
    let theta_inf = theta - angle(fprev);
    let n = (theta0 - theta_inf) / std::f64::consts::PI;
    Ok(Certificate { g, n, n_round: n.round() as i64, omega, m1, b: bb, nsteps, min_abs_f: minabs, tail_bound, f0: f0.re })
}

#[derive(Clone, Debug, Serialize)]
pub struct RootCurve {
    #[serde(rename = "gS")]
    pub gs: Vec<f64>,
    pub x: Vec<f64>,
    #[serde(rename = "S")]
    pub s: f64,
    #[serde(rename = "M1")]
    pub m1: f64,
    pub gap: f64,
    pub lin_err: f64,
}

/// x_g solving g khat(x) = 1 on (-gap, inf) for g = gS / S (Theorem 2(b)); khat is real and
/// decreasing there, so the root is bracketed around the slowing-down approximation and refined by
/// Brent's method.
pub fn dominant_root_curve(model: &Model, s: f64, m1: f64, gs: &[f64], gap: f64) -> RootCurve {
    let kh = |x: f64| model.khat_real(x);
    let mut xs = vec![];
    for &r in gs {
        let g = r / s;
        let fx = |x: f64| g * kh(x) - 1.0;
        if (r - 1.0).abs() < 1e-12 {
            xs.push(0.0);
            continue;
        }
        let x0 = (r - 1.0) * s / m1;
        let w = 0.1 * x0.abs() + 1e-3;
        let floor = -gap * (1.0 - 1e-6);
        let (mut lo, mut hi) = ((x0 - w).max(floor), x0 + w);
        while fx(hi) > 0.0 {
            hi += 2.0 * w;
        }
        while fx(lo) < 0.0 && lo > floor {
            lo = (lo - 2.0 * w).max(floor);
        }
        if fx(lo) >= 0.0 {
            let opts = BrentOpts { xtol: 1e-11, ..Default::default() };
            xs.push(brentq(fx, lo, hi, opts).expect("bracketed root"));
        } else {
            xs.push(f64::NAN);
        }
    }
    let mut lin_err: f64 = 0.0;
    for (k, &r) in gs.iter().enumerate() {
        let near = (r - 1.0).abs() <= 0.1 + 1e-12 && (r - 1.0).abs() > 1e-9;
        if near {
            let lin = (r - 1.0) * s / m1;
            lin_err = lin_err.max(((xs[k] - lin) / lin).abs());
        }
    }
    RootCurve { gs: gs.to_vec(), x: xs, s, m1, gap, lin_err }
}

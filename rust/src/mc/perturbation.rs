//! First-order formula of Theorem 4(c) under a Calvo base with full resets (port of
//! `perturbation_check.py`). Base hazard h0 = nu; perturbation H_eps = nu + eps psi with
//! psi_s = -phi(X_s) + E0[phi(X_s) | U_s], X_s = I_s U_s / (lam + nu), phi = tanh. Prediction:
//!   d/deps E_eps[I_tau^2] at 0 = 2 int_0^inf e^{-nu s} E[Cov(phi(X_s), X_s | U_s)] ds,
//! since under the base rule survival is independent of the path. Monte Carlo with common random
//! numbers at +eps and -eps, exact OU steps and thinning for the completions.

use super::{chunked, mean_std};
use crate::util::{linspace, quad::hermegauss, rng::Xoshiro, trapezoid};
use serde_json::{json, Map, Value};

struct Setup {
    lam: f64,
    nu: f64,
    sig: f64,
    x: Vec<f64>,
    w: Vec<f64>,
}

fn g(a: f64, r: f64) -> f64 {
    if r == 0.0 { a } else { -(-r * a).exp_m1() / r }
}

impl Setup {
    fn moments(&self, s: f64) -> (f64, f64, f64) {
        let (lam, sig) = (self.lam, self.sig);
        let vu = sig * sig * g(s, 2.0 * lam);
        let ciu = sig * sig / lam * (g(s, lam) - g(s, 2.0 * lam));
        let vi = sig * sig / (lam * lam) * (s - 2.0 * g(s, lam) + g(s, 2.0 * lam));
        (vu, ciu, vi)
    }
    fn cond_i(&self, s: f64, u: f64) -> (f64, f64) {
        let (vu, ciu, vi) = self.moments(s);
        let c = if vu > 0.0 { ciu / vu } else { 0.0 };
        let v = if vu > 0.0 { (vi - ciu * ciu / vu).max(0.0) } else { vi };
        (c * u, v.sqrt())
    }
    fn ephi_given_u(&self, s: f64, u: f64) -> f64 {
        let (m, sd) = self.cond_i(s, u);
        let mut acc = 0.0;
        for (xk, wk) in self.x.iter().zip(&self.w) {
            let i = m + sd * xk;
            acc += (i * u / (self.lam + self.nu)).tanh() * wk;
        }
        acc
    }
    fn prediction(&self) -> f64 {
        let ss = linspace(1e-4, 25.0, 2500);
        let vals: Vec<f64> = ss
            .iter()
            .map(|&s| {
                let (vu, _, _) = self.moments(s);
                let mut tot = 0.0;
                for (ux, uw) in self.x.iter().zip(&self.w) {
                    let u = vu.sqrt() * ux;
                    let (m, sd) = self.cond_i(s, u);
                    let (mut ephix, mut ephi, mut ex) = (0.0, 0.0, 0.0);
                    for (xk, wk) in self.x.iter().zip(&self.w) {
                        let i = m + sd * xk;
                        let xx = i * u / (self.lam + self.nu);
                        ephix += xx.tanh() * xx * wk;
                        ephi += xx.tanh() * wk;
                        ex += xx * wk;
                    }
                    tot += uw * (ephix - ephi * ex);
                }
                (-self.nu * s).exp() * tot
            })
            .collect();
        2.0 * trapezoid(&vals, &ss)
    }
}

/// E[I_tau^2] per path under H_eps (common random numbers through the per-path streams).
fn mc(st: &Setup, eps: f64, n: usize, dt: f64, seed: u64) -> Vec<f64> {
    let (lam, nu, sig) = (st.lam, st.nu, st.sig);
    let a = (-lam * dt).exp();
    let sd = sig * (-(-2.0 * lam * dt).exp_m1() / (2.0 * lam)).sqrt();
    let parts = chunked(n, 64, |range| {
        let mut out = Vec::with_capacity(range.len());
        for i in range {
            let mut r = Xoshiro::for_path(seed, i as u64);
            let (mut u, mut ii, mut t) = (0.0, 0.0, 0.0);
            loop {
                let z = r.gauss();
                let v = r.unif();
                let psi = if t > 0.0 { -(ii * u / (lam + nu)).tanh() + st.ephi_given_u(t, u) } else { 0.0 };
                let h = nu + eps * psi;
                if v < h * dt {
                    out.push(ii * ii);
                    break;
                }
                let un = u * a + sd * z;
                ii += 0.5 * (u + un) * dt;
                u = un;
                t += dt;
            }
        }
        out
    });
    parts.into_iter().flatten().collect()
}

pub fn check(n: usize) -> Map<String, Value> {
    let (lam, nu, sig) = (1.0, 1.0, 1.0);
    let (x, w) = hermegauss(40);
    let ws: f64 = w.iter().sum();
    let st = Setup { lam, nu, sig, x, w: w.iter().map(|v| v / ws).collect() };
    let p = st.prediction();
    let (dt, eps, seed) = (0.005, 0.1, 7u64);
    let rp = mc(&st, eps, n, dt, seed);
    let rm = mc(&st, -eps, n, dt, seed);
    let r0 = mc(&st, 0.0, n, dt, seed);
    let d: Vec<f64> = rp.iter().zip(&rm).map(|(a, b)| (a - b) / (2.0 * eps)).collect();
    let (dm, ds) = mean_std(&d, 0);
    let (m0, s0) = mean_std(&r0, 0);
    let nf = n as f64;
    let mut out = Map::new();
    for (k, v) in [("lam", lam), ("nu", nu), ("sig", sig), ("eps", eps), ("dt", dt), ("prediction", p), ("mc_derivative", dm),
                   ("mc_se", ds / nf.sqrt()), ("E_I2_base", m0), ("E_I2_base_se", s0 / nf.sqrt()),
                   ("E_I2_base_exact", 2.0 * sig * sig / (nu * (nu + lam) * (nu + 2.0 * lam)))] {
        out.insert(k.into(), json!(v));
    }
    out.insert("N".into(), json!(n));
    out.insert("seed".into(), json!(seed));
    out
}

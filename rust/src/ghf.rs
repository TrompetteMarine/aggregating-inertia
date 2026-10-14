//! One-dimensional stationary solver for Markov completion rules H = Lam(u) with full resets.
//! Port of the reference `ghf1d.py`.
//!
//! Population generator on cell masses p (cells uniform on [-L, L], odd count, u = 0 a cell centre):
//!   L_f p = A_f p - Lam p + e_0 <Lam, p>,
//! A_f = drift-diffusion of dU = (-lam U + f) dt + sig dW with Scharfetter-Gummel fluxes, zero flux at
//! +-L. Occupation of one episode: q = (Lam - A_f)^{-1} e_0; stationary average M(f) = <u, q> / <1, q>.
//! Response kernel at the stationary state pi_f: k(t) = <u, e^{t L_f} b>, b = (dA_f / df) pi_f.

use crate::stationary::{bern, dbern};
use crate::util::{linalg::thomas, np_sum};

pub struct Ghf {
    pub n: usize,
    pub h: f64,
    pub u: Vec<f64>,
    pub i0: usize,
    pub lam: f64,
    pub sig: f64,
    /// Lam(u_i)
    pub haz: Vec<f64>,
}

struct Drift {
    rr: Vec<f64>,
    rl: Vec<f64>,
    drr: Vec<f64>,
    drl: Vec<f64>,
}

impl Ghf {
    pub fn new<F: Fn(f64) -> f64>(lam_fn: F, lam: f64, sig: f64, l: f64, h: f64) -> Self {
        let mut n = (2.0 * l / h).ceil() as usize;
        if n % 2 == 0 {
            n += 1;
        }
        let h = 2.0 * l / n as f64;
        let mut u: Vec<f64> = (0..n).map(|i| -l + (i as f64 + 0.5) * h).collect();
        u[n / 2] = 0.0;
        let haz = u.iter().map(|&x| lam_fn(x)).collect();
        Ghf { n, h, u, i0: n / 2, lam, sig, haz }
    }

    fn drift(&self, f: f64) -> Drift {
        let (h, d) = (self.h, 0.5 * self.sig * self.sig);
        let n = self.n;
        let (mut rr, mut rl, mut drr, mut drl) = (vec![0.0; n - 1], vec![0.0; n - 1], vec![0.0; n - 1], vec![0.0; n - 1]);
        for k in 0..n - 1 {
            let uf = -(n as f64) * h / 2.0 + h * (k as f64 + 1.0);
            let v = -self.lam * uf + f;
            let pe = v * h / d;
            rr[k] = (d / h) * bern(-pe) / h;
            rl[k] = (d / h) * bern(pe) / h;
            drr[k] = -dbern(-pe) / h;
            drl[k] = dbern(pe) / h;
        }
        Drift { rr, rl, drr, drl }
    }

    /// Solve (diag(Lam) - A_f) x = rhs.
    fn solve_t(&self, dr: &Drift, rhs: &[f64]) -> Vec<f64> {
        let n = self.n;
        let mut a = vec![0.0; n];
        let mut b = vec![0.0; n];
        let mut c = vec![0.0; n];
        for i in 0..n {
            b[i] = self.haz[i] + if i < n - 1 { dr.rr[i] } else { 0.0 } + if i > 0 { dr.rl[i - 1] } else { 0.0 };
            if i > 0 {
                a[i] = -dr.rr[i - 1];
            }
            if i < n - 1 {
                c[i] = -dr.rl[i];
            }
        }
        thomas(&a, &b, &c, rhs)
    }

    pub fn occupation(&self, f: f64) -> Vec<f64> {
        let dr = self.drift(f);
        let mut e0 = vec![0.0; self.n];
        e0[self.i0] = 1.0;
        self.solve_t(&dr, &e0)
    }

    pub fn m(&self, f: f64) -> f64 {
        let q = self.occupation(f);
        let num: f64 = self.u.iter().zip(&q).map(|(a, b)| a * b).sum();
        num / np_sum(&q)
    }

    fn pi_b(&self, f: f64, dr: &Drift) -> (Vec<f64>, Vec<f64>) {
        let mut e0 = vec![0.0; self.n];
        e0[self.i0] = 1.0;
        let q = self.solve_t(dr, &e0);
        let s = np_sum(&q);
        let pi: Vec<f64> = q.iter().map(|x| x / s).collect();
        let _ = f;
        // b = dA pi: dA has sub-diagonal drr, super-diagonal drl, main -(drr[i] + drl[i-1])
        let n = self.n;
        let mut b = vec![0.0; n];
        for i in 0..n {
            let mut v = 0.0;
            if i < n - 1 {
                v -= dr.drr[i] * pi[i];
                v += dr.drl[i] * pi[i + 1];
            }
            if i > 0 {
                v -= dr.drl[i - 1] * pi[i];
                v += dr.drr[i - 1] * pi[i - 1];
            }
            b[i] = v;
        }
        (pi, b)
    }

    /// y = L_f x = A_f x - Lam x + e_0 <Lam, x>.
    fn gen_apply(&self, dr: &Drift, x: &[f64]) -> Vec<f64> {
        let n = self.n;
        let mut y = vec![0.0; n];
        for i in 0..n {
            let mut v = -self.haz[i] * x[i];
            if i < n - 1 {
                v -= dr.rr[i] * x[i];
                v += dr.rl[i] * x[i + 1];
            }
            if i > 0 {
                v -= dr.rl[i - 1] * x[i];
                v += dr.rr[i - 1] * x[i - 1];
            }
            y[i] = v;
        }
        let inflow: f64 = self.haz.iter().zip(x).map(|(a, b)| a * b).sum();
        y[self.i0] += inflow;
        y
    }

    /// k(t) at the given (equally spaced, increasing) times by Taylor steps with ||h L||_1 <= 1/2.
    pub fn kernel(&self, f: f64, ts: &[f64]) -> Vec<f64> {
        let dr = self.drift(f);
        let (_, b) = self.pi_b(f, &dr);
        let n = self.n;
        // ||L||_1: maximum absolute column sum
        let mut colmax: f64 = 0.0;
        for i in 0..n {
            let mut s = self.haz[i];
            if i < n - 1 {
                s += 2.0 * dr.rr[i];
            }
            if i > 0 {
                s += 2.0 * dr.rl[i - 1];
            }
            s += self.haz[i];
            colmax = colmax.max(s);
        }
        let mut v = b.clone();
        let mut out = Vec::with_capacity(ts.len());
        let mut tcur = 0.0;
        for &t in ts {
            let span = t - tcur;
            if span > 0.0 {
                let steps = ((span * colmax) / 0.5).ceil().max(1.0) as usize;
                let hstep = span / steps as f64;
                for _ in 0..steps {
                    let mut term = v.clone();
                    let mut acc = v.clone();
                    for k in 1..80 {
                        let lt = self.gen_apply(&dr, &term);
                        let kf = k as f64;
                        term = lt.iter().map(|x| x * hstep / kf).collect();
                        let mut tn: f64 = 0.0;
                        for (a, tv) in acc.iter_mut().zip(&term) {
                            *a += tv;
                            tn = tn.max(tv.abs());
                        }
                        let an = acc.iter().fold(0.0f64, |m, x| m.max(x.abs()));
                        if tn <= 1e-18 * an.max(1e-300) {
                            break;
                        }
                    }
                    v = acc;
                }
                tcur = t;
            }
            out.push(self.u.iter().zip(&v).map(|(a, b)| a * b).sum());
        }
        out
    }

    /// Integral of the kernel at pi_f: <u, x> with -L_f x = b and <1, x> = 0.
    pub fn s_tilted(&self, f: f64) -> f64 {
        let dr = self.drift(f);
        let (_, b) = self.pi_b(f, &dr);
        let y = self.solve_t(&dr, &b);
        let mut e0 = vec![0.0; self.n];
        e0[self.i0] = 1.0;
        let z0 = self.solve_t(&dr, &e0);
        let alpha = -np_sum(&y) / np_sum(&z0);
        let x: Vec<f64> = y.iter().zip(&z0).map(|(a, b)| a + alpha * b).collect();
        self.u.iter().zip(&x).map(|(a, b)| a * b).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Calvo: M(f) = f / (lam + nu) and the kernel is exp(-(lam + nu) t).
    #[test]
    fn calvo() {
        let (lam, nu) = (1.0, 0.5);
        let g = Ghf::new(|_| nu, lam, 1.0, 12.0, 0.02);
        for f in [0.1, 1.0] {
            assert!((g.m(f) * (lam + nu) / f - 1.0).abs() < 1e-3);
        }
        assert!((g.s_tilted(0.0) - 1.0 / (lam + nu)).abs() < 1e-3);
        let gk = Ghf::new(|_| nu, lam, 1.0, 12.0, 0.04);
        let ts = [0.0, 0.5, 1.0, 2.0];
        let k = gk.kernel(0.0, &ts);
        for (t, kv) in ts.iter().zip(&k) {
            assert!((kv - (-(lam + nu) * t).exp()).abs() < 5e-3, "t={t} k={kv}");
        }
    }
}

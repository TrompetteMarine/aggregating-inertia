//! Shift-invert Arnoldi for the eigenvalues of L_odd + g b u^T nearest to a real shift (the role of
//! scipy's ARPACK `eigs` in the reference). The Krylov dimension grows until the wanted Ritz pairs
//! satisfy the ARPACK criterion ||r|| <= tol |theta|; every returned eigenvalue carries the residual
//! ||A v - lambda v|| / ||v|| of its Ritz vector, computed with the operator itself.

use super::{Model, Shifted};
use crate::util::rng::Xoshiro;
use nalgebra::DMatrix;
use num_complex::Complex64;

/// An eigenvalue estimate with its residual.
#[derive(Clone, Copy, Debug)]
pub struct Eig {
    pub lambda: Complex64,
    pub residual: f64,
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Eigenvalues of A = L_odd + g b u^T nearest to the real shift sigma (nev of them), sorted by
/// decreasing real part.
pub fn rightmost_eigs(model: &Model, g: f64, sigma: f64, nev: usize, tol: f64) -> Vec<Eig> {
    let sec = &model.odd;
    let n = sec.nn;
    let sh = Shifted::<f64>::new(sec, sigma, false);
    let uw = |x: &[f64]| sec.inner_u(x);
    let b = &model.b;
    let pb = sh.solve(sec, b);
    let denom = 1.0 + g * uw(&pb);
    let op = |y: &[f64]| -> Vec<f64> {
        let mut x = sh.solve(sec, y);
        let c = g * uw(&x) / denom;
        if c != 0.0 {
            for (xi, pi) in x.iter_mut().zip(&pb) {
                *xi -= c * pi;
            }
        }
        x
    };
    let apply_a = |x: &[f64]| -> Vec<f64> {
        let mut y = sec.matvec(x);
        let c = g * uw(x);
        if c != 0.0 {
            for (yi, bi) in y.iter_mut().zip(b) {
                *yi += c * bi;
            }
        }
        y
    };
    let mut rng = Xoshiro::from_seed(20261014);
    let v0: Vec<f64> = (0..n).map(|_| rng.unif() - 0.5).collect();
    let mut m = 30usize.max(2 * nev + 1).min(n);
    loop {
        let (ritz, conv) = arnoldi(&op, &v0, m, nev, tol);
        if conv || m >= n.min(480) {
            // residuals with the operator A itself, for the record
            let mut out: Vec<Eig> = ritz
                .into_iter()
                .map(|(theta, vec)| {
                    let lam = Complex64::new(sigma, 0.0) + Complex64::new(1.0, 0.0) / theta;
                    let (vr, vi): (Vec<f64>, Vec<f64>) = vec.iter().map(|z| (z.re, z.im)).unzip();
                    let (ar, ai) = (apply_a(&vr), apply_a(&vi));
                    let mut r2 = 0.0;
                    let mut v2 = 0.0;
                    for k in 0..n {
                        let av = Complex64::new(ar[k], ai[k]);
                        let r = av - lam * vec[k];
                        r2 += r.norm_sqr();
                        v2 += vec[k].norm_sqr();
                    }
                    Eig { lambda: lam, residual: (r2 / v2).sqrt() }
                })
                .collect();
            out.sort_by(|a, b| b.lambda.re.partial_cmp(&a.lambda.re).unwrap());
            return out;
        }
        m = (2 * m).min(n);
    }
}

/// Arnoldi with full reorthogonalisation; returns the nev Ritz pairs of largest modulus (values
/// and vectors in the original space) and whether they meet the tolerance.
fn arnoldi<F: Fn(&[f64]) -> Vec<f64>>(op: &F, v0: &[f64], m: usize, nev: usize, tol: f64) -> (Vec<(Complex64, Vec<Complex64>)>, bool) {
    let n = v0.len();
    let mut v: Vec<Vec<f64>> = Vec::with_capacity(m + 1);
    let nv = norm(v0);
    v.push(v0.iter().map(|x| x / nv).collect());
    let mut h = DMatrix::<f64>::zeros(m + 1, m);
    let mut mm = m;
    for j in 0..m {
        let mut w = op(&v[j]);
        for _pass in 0..2 {
            for i in 0..=j {
                let c: f64 = v[i].iter().zip(&w).map(|(a, b)| a * b).sum();
                for (wk, vk) in w.iter_mut().zip(&v[i]) {
                    *wk -= c * vk;
                }
                h[(i, j)] += c;
            }
        }
        let hn = norm(&w);
        h[(j + 1, j)] = hn;
        if hn < 1e-14 {
            mm = j + 1;
            break;
        }
        v.push(w.iter().map(|x| x / hn).collect());
    }
    let hm = h.view((0, 0), (mm, mm)).into_owned();
    let vals = hm.complex_eigenvalues();
    let mut idx: Vec<usize> = (0..mm).collect();
    idx.sort_by(|&a, &b| vals[b].norm().partial_cmp(&vals[a].norm()).unwrap());
    let hc: DMatrix<Complex64> = hm.map(|x| Complex64::new(x, 0.0));
    let beta = if mm < h.nrows() { h[(mm, mm - 1)] } else { 0.0 };
    let mut conv = true;
    let mut out = vec![];
    for &k in idx.iter().take(nev.min(mm)) {
        let theta = vals[k];
        // Ritz vector of H by inverse iteration
        let shift = theta + Complex64::new(1e-10 * theta.norm().max(1e-300), 1e-10 * theta.norm().max(1e-300));
        let a = &hc - DMatrix::<Complex64>::identity(mm, mm) * shift;
        let lu = a.lu();
        let mut y = nalgebra::DVector::<Complex64>::from_element(mm, Complex64::new(1.0, 0.0));
        for _ in 0..3 {
            if let Some(s) = lu.solve(&y) {
                let nrm = s.norm();
                y = s / Complex64::new(nrm, 0.0);
            }
        }
        let est = beta * y[mm - 1].norm() / y.norm();
        if est > tol * theta.norm() {
            conv = false;
        }
        let mut x = vec![Complex64::new(0.0, 0.0); n];
        for (i, vi) in v.iter().enumerate().take(mm) {
            for (xk, &vk) in x.iter_mut().zip(vi) {
                *xk += y[i] * vk;
            }
        }
        out.push((theta, x));
    }
    (out, conv)
}

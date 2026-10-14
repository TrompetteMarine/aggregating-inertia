//! Quadrature: adaptive Gauss-Kronrod (21 points) on finite intervals and on [a, inf), and Gauss
//! rules (Legendre, probabilists' Hermite) by the Golub-Welsch method refined by Newton steps.

use nalgebra::{DMatrix, SymmetricEigen};
use std::collections::BinaryHeap;

// Gauss-Kronrod 21-point abscissae and weights (QUADPACK qk21).
const XGK: [f64; 11] = [
    0.995657163025808080735527280689003,
    0.973906528517171720077964012084452,
    0.930157491355708226001207180059508,
    0.865063366688984510732096688423493,
    0.780817726586416897063717578345042,
    0.679409568299024406234327365114874,
    0.562757134668604683339000099272694,
    0.433395394129247190799265943165784,
    0.294392862701460198131126603103866,
    0.148874338981631210884826001129720,
    0.000000000000000000000000000000000,
];
const WGK: [f64; 11] = [
    0.011694638867371874278064396062192,
    0.032558162307964727478818972459390,
    0.054755896574351996031381300244580,
    0.075039674810919952767043140916190,
    0.093125454583697605535065465083366,
    0.109387158802297641899210590325805,
    0.123491976262065851077208929815070,
    0.134709217311473325928054001771707,
    0.142775938577060080797094273138717,
    0.147739104901338491374841515972068,
    0.149445554002916905664936468389821,
];
const WG: [f64; 5] = [
    0.066671344308688137593568809893332,
    0.149451349150580593145776339657697,
    0.219086362515982043995534934228163,
    0.269266719309996355091226921569469,
    0.295524224714752870173892994651338,
];

fn gk21<F: FnMut(f64) -> f64>(f: &mut F, a: f64, b: f64) -> (f64, f64) {
    let c = 0.5 * (a + b);
    let h = 0.5 * (b - a);
    let fc = f(c);
    let mut resk = fc * WGK[10];
    let mut resg = 0.0;
    for j in 0..10 {
        let x = h * XGK[j];
        let f1 = f(c - x);
        let f2 = f(c + x);
        resk += WGK[j] * (f1 + f2);
        if j % 2 == 1 {
            resg += WG[j / 2] * (f1 + f2);
        }
    }
    let res = resk * h;
    let err = ((resk - resg) * h).abs();
    (res, err)
}

#[derive(PartialEq)]
struct Seg {
    err: f64,
    a: f64,
    b: f64,
    val: f64,
}
impl Eq for Seg {}
impl PartialOrd for Seg {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Seg {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.err.partial_cmp(&o.err).unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Adaptive Gauss-Kronrod on [a, b]; stops when the error estimate is below max(epsabs, epsrel |I|).
pub fn quad<F: FnMut(f64) -> f64>(mut f: F, a: f64, b: f64, epsabs: f64, epsrel: f64, limit: usize) -> (f64, f64) {
    let (v, e) = gk21(&mut f, a, b);
    let mut heap = BinaryHeap::new();
    heap.push(Seg { err: e, a, b, val: v });
    let (mut total, mut toterr) = (v, e);
    let mut n = 1;
    while toterr > epsabs.max(epsrel * total.abs()) && n < limit {
        let s = heap.pop().unwrap();
        let m = 0.5 * (s.a + s.b);
        let (v1, e1) = gk21(&mut f, s.a, m);
        let (v2, e2) = gk21(&mut f, m, s.b);
        total += v1 + v2 - s.val;
        toterr += e1 + e2 - s.err;
        heap.push(Seg { err: e1, a: s.a, b: m, val: v1 });
        heap.push(Seg { err: e2, a: m, b: s.b, val: v2 });
        n += 1;
    }
    // recompute the sum from the pieces to avoid drift in the running total
    let total: f64 = heap.iter().map(|s| s.val).sum();
    let toterr: f64 = heap.iter().map(|s| s.err).sum();
    (total, toterr)
}

/// Integral over [a, inf) by the substitution x = a + (1 - t) / t, t in (0, 1].
pub fn quad_inf<F: FnMut(f64) -> f64>(mut f: F, a: f64, epsabs: f64, epsrel: f64, limit: usize) -> (f64, f64) {
    let g = |t: f64| {
        if t <= 0.0 {
            0.0
        } else {
            let x = a + (1.0 - t) / t;
            let v = f(x) / (t * t);
            if v.is_finite() { v } else { 0.0 }
        }
    };
    quad(g, 0.0, 1.0, epsabs, epsrel, limit)
}

/// Gauss-Legendre nodes and weights on [-1, 1] (numpy.polynomial.legendre.leggauss).
pub fn leggauss(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut j = DMatrix::<f64>::zeros(n, n);
    for k in 1..n {
        let b = k as f64 / ((4 * k * k - 1) as f64).sqrt();
        j[(k, k - 1)] = b;
        j[(k - 1, k)] = b;
    }
    let eig = SymmetricEigen::new(j);
    let mut x: Vec<f64> = eig.eigenvalues.iter().cloned().collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut w = vec![0.0; n];
    for (i, xi) in x.iter_mut().enumerate() {
        // Newton refinement on P_n
        for _ in 0..3 {
            let (p, dp) = legendre(n, *xi);
            *xi -= p / dp;
        }
        let (_, dp) = legendre(n, *xi);
        w[i] = 2.0 / ((1.0 - *xi * *xi) * dp * dp);
    }
    (x, w)
}

fn legendre(n: usize, x: f64) -> (f64, f64) {
    let (mut p0, mut p1) = (1.0, x);
    for k in 2..=n {
        let kf = k as f64;
        let p2 = ((2.0 * kf - 1.0) * x * p1 - (kf - 1.0) * p0) / kf;
        p0 = p1;
        p1 = p2;
    }
    let dp = n as f64 * (x * p1 - p0) / (x * x - 1.0);
    (p1, dp)
}

/// Probabilists' Gauss-Hermite nodes and weights for the weight exp(-x^2/2)
/// (numpy.polynomial.hermite_e.hermegauss; weights sum to sqrt(2 pi)).
pub fn hermegauss(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut j = DMatrix::<f64>::zeros(n, n);
    for k in 1..n {
        let b = (k as f64).sqrt();
        j[(k, k - 1)] = b;
        j[(k - 1, k)] = b;
    }
    let eig = SymmetricEigen::new(j);
    let mut x: Vec<f64> = eig.eigenvalues.iter().cloned().collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut w = vec![0.0; n];
    let norm = (2.0 * std::f64::consts::PI).sqrt();
    for (i, xi) in x.iter_mut().enumerate() {
        for _ in 0..3 {
            let (p, dp) = hermite_e_normalised(n, *xi);
            *xi -= p / dp;
        }
        // w = sqrt(2 pi) n! / (n He_{n-1}(x)^2) written with normalised polynomials to avoid overflow
        let (_, _, q) = hermite_e_norm_pair(n, *xi);
        w[i] = norm / (n as f64 * q * q);
    }
    (x, w)
}

/// Normalised probabilists' Hermite h_k = He_k / sqrt(k!): returns (h_n, h_n').
fn hermite_e_normalised(n: usize, x: f64) -> (f64, f64) {
    let (hn, _, hnm1) = hermite_e_norm_pair(n, x);
    // He_n' = n He_{n-1}  =>  h_n' = sqrt(n) h_{n-1}
    (hn, (n as f64).sqrt() * hnm1)
}

fn hermite_e_norm_pair(n: usize, x: f64) -> (f64, f64, f64) {
    // h_{k+1} = (x h_k - sqrt(k) h_{k-1}) / sqrt(k+1)
    let (mut hm1, mut h) = (0.0, 1.0);
    for k in 0..n {
        let kf = k as f64;
        let hp = (x * h - kf.sqrt() * hm1) / (kf + 1.0).sqrt();
        hm1 = h;
        h = hp;
    }
    // returns (h_n, unused, h_{n-1}); the weight formula uses He_{n-1}^2/(n-1)! = h_{n-1}^2 and
    // w = sqrt(2 pi) (n-1)! / (n He_{n-1}^2) * ... = sqrt(2 pi) / (n h_{n-1}^2)
    (h, 0.0, hm1)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn quad_inf_exp() {
        let (v, _) = quad_inf(|x| (-x).exp(), 0.0, 1e-14, 1e-13, 500);
        assert!((v - 1.0).abs() < 1e-12);
        let (v, _) = quad_inf(|x| x * x * (-2.0 * x).exp(), 0.0, 1e-14, 1e-13, 500);
        assert!((v - 0.25).abs() < 1e-12);
    }
    #[test]
    fn legendre_rule() {
        let (x, w) = leggauss(80);
        let s: f64 = w.iter().sum();
        assert!((s - 2.0).abs() < 1e-13);
        let i4: f64 = x.iter().zip(&w).map(|(x, w)| w * x.powi(4)).sum();
        assert!((i4 - 0.4).abs() < 1e-13);
    }
    #[test]
    fn hermite_rule() {
        let (x, w) = hermegauss(40);
        let s: f64 = w.iter().sum();
        let norm = (2.0 * std::f64::consts::PI).sqrt();
        assert!((s / norm - 1.0).abs() < 1e-12);
        let m2: f64 = x.iter().zip(&w).map(|(x, w)| w * x * x).sum::<f64>() / s;
        let m4: f64 = x.iter().zip(&w).map(|(x, w)| w * x.powi(4)).sum::<f64>() / s;
        assert!((m2 - 1.0).abs() < 1e-12 && (m4 - 3.0).abs() < 1e-11);
    }
}

//! Dense linear algebra helpers on top of nalgebra, and a tridiagonal solver.

use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;

/// Solve A x = b for a small dense real system (LU with partial pivoting).
pub fn solve_dense(a: &DMatrix<f64>, b: &[f64]) -> Vec<f64> {
    let lu = a.clone().lu();
    let x = lu.solve(&DVector::from_column_slice(b)).expect("singular dense system");
    x.iter().cloned().collect()
}

/// Least squares min ||A c - y|| for a tall full-rank A (Householder QR), as numpy.linalg.lstsq.
pub fn lstsq(a: &DMatrix<f64>, y: &[f64]) -> Vec<f64> {
    let svd = a.clone().svd(true, true);
    let c = svd.solve(&DVector::from_column_slice(y), 1e-15).expect("lstsq failed");
    c.iter().cloned().collect()
}

/// Thomas algorithm for a tridiagonal system with sub-diagonal `a` (a[0] unused), diagonal `b`,
/// super-diagonal `c` (c[n-1] unused). No pivoting: for M-matrices.
pub fn thomas(a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut cp = vec![0.0; n];
    let mut dp = vec![0.0; n];
    cp[0] = c[0] / b[0];
    dp[0] = d[0] / b[0];
    for i in 1..n {
        let m = b[i] - a[i] * cp[i - 1];
        cp[i] = if i < n - 1 { c[i] / m } else { 0.0 };
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m;
    }
    let mut x = vec![0.0; n];
    x[n - 1] = dp[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = dp[i] - cp[i] * x[i + 1];
    }
    x
}

/// LU factorisation of a small dense complex or real matrix, reused for many right-hand sides.
pub enum DenseLu {
    Real(nalgebra::LU<f64, nalgebra::Dyn, nalgebra::Dyn>),
    Cplx(nalgebra::LU<Complex64, nalgebra::Dyn, nalgebra::Dyn>),
}

impl DenseLu {
    pub fn solve_real(&self, r: &[f64]) -> Vec<f64> {
        match self {
            DenseLu::Real(lu) => lu.solve(&DVector::from_column_slice(r)).expect("singular").iter().cloned().collect(),
            DenseLu::Cplx(_) => panic!("real solve on a complex factorisation"),
        }
    }
    pub fn solve_cplx(&self, r: &[Complex64]) -> Vec<Complex64> {
        match self {
            DenseLu::Cplx(lu) => lu.solve(&DVector::from_column_slice(r)).expect("singular").iter().cloned().collect(),
            DenseLu::Real(_) => panic!("complex solve on a real factorisation"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn thomas_solves() {
        let n = 6;
        let a = vec![-1.0; n];
        let b = vec![4.0; n];
        let c = vec![-1.0; n];
        let x0: Vec<f64> = (0..n).map(|i| i as f64 + 1.0).collect();
        let mut d = vec![0.0; n];
        for i in 0..n {
            d[i] = b[i] * x0[i] + if i > 0 { a[i] * x0[i - 1] } else { 0.0 } + if i < n - 1 { c[i] * x0[i + 1] } else { 0.0 };
        }
        let x = thomas(&a, &b, &c, &d);
        for i in 0..n {
            assert!((x[i] - x0[i]).abs() < 1e-12);
        }
    }
    #[test]
    fn lstsq_fits_polynomial() {
        let fs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3];
        let mut a = DMatrix::<f64>::zeros(6, 3);
        let mut y = vec![0.0; 6];
        for (i, &f) in fs.iter().enumerate() {
            a[(i, 0)] = f;
            a[(i, 1)] = f.powi(3);
            a[(i, 2)] = f.powi(5);
            y[i] = 0.5 * f - 0.2 * f.powi(3) + 0.1 * f.powi(5);
        }
        let c = lstsq(&a, &y);
        assert!((c[0] - 0.5).abs() < 1e-10 && (c[1] + 0.2).abs() < 1e-8 && (c[2] - 0.1).abs() < 1e-6);
    }
}

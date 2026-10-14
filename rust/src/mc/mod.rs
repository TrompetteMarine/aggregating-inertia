//! Monte Carlo checks.
//!
//! * [`episode`]: economy B (true clock) and economy P (tabulated projected hazard) path by path with
//!   common random numbers and the killing integrated out (port of `mc.c` and `run_mc.py`); the
//!   generator and the per-path streams are those of the C code, so the paths are reproduced exactly.
//! * [`kernel`]: the response kernel k(t) by a coupled Euler simulation of a displaced stationary
//!   population, compared with the finite-volume kernel (`mc_kernel.py`, `compare_kernel_mc.py`).
//! * [`identity`]: the age-only formula for S and the fluctuation-response identity under the Calvo
//!   rule (`identity_checks.py`).
//! * [`perturbation`]: the first-order formula of Theorem 4(c) (`perturbation_check.py`).
//!
//! The Python checks drew from numpy's generator; here every path has its own xoshiro256** stream, so
//! these estimates agree with the reference up to Monte Carlo error, not digit for digit.

pub mod episode;
pub mod identity;
pub mod kernel;
pub mod perturbation;

/// Mean and standard deviation (ddof) of a slice.
pub fn mean_std(x: &[f64], ddof: usize) -> (f64, f64) {
    let n = x.len() as f64;
    let m = crate::util::np_sum(x) / n;
    let ss: Vec<f64> = x.iter().map(|v| (v - m) * (v - m)).collect();
    (m, (crate::util::np_sum(&ss) / (n - ddof as f64)).sqrt())
}

/// Sums accumulated over particles in fixed chunks and combined in chunk order, so that results do
/// not depend on the number of threads.
pub fn chunked<F>(n: usize, chunks: usize, f: F) -> Vec<Vec<f64>>
where
    F: Fn(std::ops::Range<usize>) -> Vec<f64> + Sync,
{
    use rayon::prelude::*;
    let size = n.div_ceil(chunks);
    (0..chunks).into_par_iter().map(|c| f(c * size..((c + 1) * size).min(n))).collect()
}

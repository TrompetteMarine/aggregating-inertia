//! Small numerical utilities shared by the solvers. Where the reference pipeline (Python + C) used a
//! numpy or scipy routine, the function here follows that routine's arithmetic, so that the Rust and
//! reference results agree to rounding.

pub mod fp;
pub mod interp;
pub mod linalg;
pub mod quad;
pub mod rng;
pub mod roots;

/// Smooth step s(z) = e^{-1/z} / (e^{-1/z} + e^{-1/(1-z)}) on (0, 1), 0 for z <= 0 and 1 for z >= 1.
/// This is the form of the episode solver and the Monte Carlo code.
#[inline]
pub fn smoothstep(z: f64) -> f64 {
    let z = z.clamp(0.0, 1.0);
    if z >= 1.0 {
        1.0
    } else if z > 0.0 {
        let a = (-1.0 / z).exp();
        let b = (-1.0 / (1.0 - z)).exp();
        a / (a + b)
    } else {
        0.0
    }
}

/// The same smooth step written as expit(1/(1-z) - 1/z), the form of the stationary solver.
#[inline]
pub fn smoothstep_expit(z: f64) -> f64 {
    let z = z.clamp(0.0, 1.0);
    if z >= 1.0 {
        1.0
    } else if z > 0.0 {
        expit(1.0 / (1.0 - z) - 1.0 / z)
    } else {
        0.0
    }
}

/// Logistic function as scipy.special.expit computes it.
#[inline]
pub fn expit(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Activation of the pausing clock: alpha(u) = chi(u) = s((|u| - theta) / theta); identically one
/// when theta = 0 (episode-solver form).
#[inline]
pub fn activation(u: f64, theta: f64) -> f64 {
    if theta == 0.0 { 1.0 } else { smoothstep((u.abs() - theta) / theta) }
}

/// Activation in the stationary-solver form.
#[inline]
pub fn activation_expit(u: f64, theta: f64) -> f64 {
    if theta == 0.0 { 1.0 } else { smoothstep_expit((u.abs() - theta) / theta) }
}

/// Graded time grid: steps grow geometrically from `dt_min` to `dt_max` until T is passed; with
/// `refine > 1` every step is split into `refine` equal substeps (as c9core.time_grid).
pub fn time_grid(t_end: f64, dt_max: f64, dt_min: f64, growth: f64, refine: usize) -> Vec<f64> {
    let mut ts = vec![0.0];
    let mut dt = dt_min;
    while *ts.last().unwrap() < t_end {
        let next = ts.last().unwrap() + dt;
        ts.push(next);
        dt = dt_max.min(dt * growth);
    }
    if refine > 1 {
        let mut fine = Vec::with_capacity((ts.len() - 1) * refine + 1);
        fine.push(0.0);
        for k in 0..ts.len() - 1 {
            let h = (ts[k + 1] - ts[k]) / refine as f64;
            for j in 1..=refine {
                fine.push(ts[k] + h * j as f64);
            }
        }
        ts = fine;
    }
    ts
}

/// Nearest odd integer at or above round(n).
pub fn odd(n: f64) -> usize {
    let n = n.round() as usize;
    if n % 2 == 1 { n } else { n + 1 }
}

/// Default half-width of the assessment domain of the episode solver, by pull.
pub fn default_l(lam: f64) -> f64 {
    if lam == 0.0 {
        18.0
    } else if lam == 0.25 {
        12.0
    } else if lam == 1.0 {
        6.5
    } else if lam == 2.0 {
        5.5
    } else {
        8.0
    }
}

/// numpy.linspace(start, stop, num) with endpoint.
pub fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    if num == 0 {
        return vec![];
    }
    if num == 1 {
        return vec![start];
    }
    let div = (num - 1) as f64;
    let step = (stop - start) / div;
    let mut y: Vec<f64> = (0..num).map(|i| i as f64 * step + start).collect();
    y[num - 1] = stop;
    y
}

/// numpy.geomspace(start, stop, num) with endpoint (positive arguments).
pub fn geomspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    let ls = linspace(start.log10(), stop.log10(), num);
    let mut y: Vec<f64> = ls.iter().map(|&e| 10f64.powf(e)).collect();
    if num > 0 {
        y[0] = start;
    }
    if num > 1 {
        y[num - 1] = stop;
    }
    y
}

/// Richardson extrapolation in the time step for first-order backward Euler with refinement factors
/// `refs` (for example 1, 2, 4): returns (R2, R3), R3 only when the last three factors are r, 2r, 4r.
pub fn rich_t(vals: &[f64], refs: &[usize]) -> (Option<f64>, Option<f64>) {
    let n = refs.len();
    let mut r2 = None;
    let mut r3 = None;
    if n >= 2 {
        r2 = Some(2.0 * vals[n - 1] - vals[n - 2]);
    }
    if n >= 3 && refs[n - 2] == 2 * refs[n - 3] && refs[n - 1] == 4 * refs[n - 3] {
        r3 = Some((8.0 * vals[n - 1] - 6.0 * vals[n - 2] + vals[n - 3]) / 3.0);
    }
    (r2, r3)
}

/// First-order Richardson in 1/Nb with the fine grid twice the coarse one.
#[inline]
pub fn rich_b(x_coarse: f64, x_fine: f64) -> f64 {
    2.0 * x_fine - x_coarse
}

/// Second-order Richardson in the cell width.
#[inline]
pub fn rich_u(x1: f64, h1: f64, x2: f64, h2: f64) -> f64 {
    (h1 * h1 * x2 - h2 * h2 * x1) / (h1 * h1 - h2 * h2)
}

/// Sum with numpy's pairwise algorithm (blocks of 128, eight accumulators), so that sums of long
/// arrays agree with numpy's to the last bit in the common case.
pub fn np_sum(a: &[f64]) -> f64 {
    pairwise(a)
}

fn pairwise(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0;
        for &x in a {
            res += x;
        }
        res
    } else if n <= 128 {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        while i < n - (n % 8) {
            for k in 0..8 {
                r[k] += a[i + k];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise(&a[..n2]) + pairwise(&a[n2..])
    }
}

/// Dot product accumulated left to right.
#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Trapezoid rule as numpy.trapezoid(y, x).
pub fn trapezoid(y: &[f64], x: &[f64]) -> f64 {
    let terms: Vec<f64> = (1..y.len()).map(|i| (x[i] - x[i - 1]) * (y[i] + y[i - 1]) / 2.0).collect();
    np_sum(&terms)
}

/// scipy.integrate.cumulative_trapezoid(y, x, initial=0).
pub fn cumulative_trapezoid(y: &[f64], x: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(y.len());
    out.push(0.0);
    let mut acc = 0.0;
    for i in 1..y.len() {
        acc += (x[i] - x[i - 1]) * (y[i] + y[i - 1]) / 2.0;
        out.push(acc);
    }
    out
}

/// Masses of cells with the given faces, rebinned onto `edges` through the piecewise-linear CDF.
pub fn binned_from_cdf(mass: &[f64], faces: &[f64], edges: &[f64]) -> Vec<f64> {
    let mut cdf = Vec::with_capacity(mass.len() + 1);
    cdf.push(0.0);
    let mut acc = 0.0;
    for m in mass {
        acc += m;
        cdf.push(acc);
    }
    let vals: Vec<f64> = edges.iter().map(|&e| interp::interp(e, faces, &cdf)).collect();
    (1..vals.len()).map(|k| vals[k] - vals[k - 1]).collect()
}

/// Index of the minimum (first one on ties), as numpy.argmin.
pub fn argmin(a: &[f64]) -> usize {
    let mut best = 0;
    for i in 1..a.len() {
        if a[i] < a[best] {
            best = i;
        }
    }
    best
}

/// Index of the maximum (first one on ties), as numpy.argmax.
pub fn argmax(a: &[f64]) -> usize {
    let mut best = 0;
    for i in 1..a.len() {
        if a[i] > a[best] {
            best = i;
        }
    }
    best
}

pub fn max_of(a: &[f64]) -> f64 {
    a.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
}

pub fn min_of(a: &[f64]) -> f64 {
    a.iter().cloned().fold(f64::INFINITY, f64::min)
}

/// Format like Python's "{:.Ne}" (mantissa with N decimals, exponent with sign and at least two digits).
pub fn py_exp(x: f64, decimals: usize) -> String {
    let s = format!("{:.*e}", decimals, x);
    // Rust gives "3.3e-15" or "3.3e2"; Python gives "3.3e-15" and "3.3e+02".
    let (m, e) = s.split_once('e').unwrap();
    let ev: i32 = e.parse().unwrap();
    let sign = if ev < 0 { '-' } else { '+' };
    format!("{}e{}{:02}", m, sign, ev.abs())
}

/// 3.3e-15 -> 3\times10^{-15} (one significant digit), as make_data.tex_sci.
pub fn tex_sci(x: f64) -> String {
    let s = py_exp(x, 0);
    let (m, e) = s.split_once('e').unwrap();
    let ev: i32 = e.parse().unwrap();
    format!("{}\\times10^{{{}}}", m, ev)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_and_steps() {
        let ts = time_grid(1.0, 0.04, 1e-3, 1.1, 1);
        assert_eq!(ts[0], 0.0);
        assert!(*ts.last().unwrap() >= 1.0);
        let tf = time_grid(1.0, 0.04, 1e-3, 1.1, 2);
        assert_eq!(tf.len(), 2 * (ts.len() - 1) + 1);
        assert_eq!(odd(650.0), 651);
        assert_eq!(odd(651.0), 651);
    }

    #[test]
    fn formatting() {
        assert_eq!(py_exp(3.3e-15, 0), "3e-15");
        assert_eq!(tex_sci(3.3e-15), "3\\times10^{-15}");
        assert_eq!(py_exp(1234.5, 1), "1.2e+03");
    }

    #[test]
    fn pairwise_sum_matches_naive() {
        let a: Vec<f64> = (0..1000).map(|i| (i as f64).sin()).collect();
        let s1 = np_sum(&a);
        let s2: f64 = a.iter().sum();
        assert!((s1 - s2).abs() < 1e-12);
    }
}

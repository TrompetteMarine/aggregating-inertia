//! Independent check of k(t) by a coupled Monte Carlo (Euler): a stationary population is displaced
//! by +eps and -eps at t = 0; the two copies share every Gaussian increment and every uniform used
//! for the completion decisions, and k(t) ~ (mean_+ - mean_-) / (2 eps). Port of `mc_kernel.py`,
//! with the comparison against the finite-volume kernel of `compare_kernel_mc.py`.

use super::chunked;
use crate::stationary::{analysis, default_grid, Model, Params, Reset, ResetKind};
use crate::util::{activation_expit, interp::interp_many, rng::Xoshiro, trapezoid};
use serde_json::{json, Map, Value};

#[derive(Clone, Copy, Debug)]
pub struct KernelCase {
    pub name: &'static str,
    pub theta: f64,
    pub eta: f64,
    pub nu: f64,
    pub lam: f64,
    pub reset: ResetKind,
}

pub fn cases() -> Vec<KernelCase> {
    const NU06: f64 = 0.7136347625525;
    vec![
        KernelCase { name: "V6_Rpl", theta: 0.7, eta: 0.6, nu: NU06, lam: 1.0, reset: ResetKind::Pl },
        KernelCase { name: "BENCH_R0", theta: 0.7, eta: 0.6, nu: NU06, lam: 1.0, reset: ResetKind::Zero },
        KernelCase { name: "X_R0_th035_eta20", theta: 0.35, eta: 20.0, nu: 0.2, lam: 0.25, reset: ResetKind::Zero },
    ]
}

/// (t, k, se): the coupled Monte Carlo kernel recorded every `rec`.
#[allow(clippy::too_many_arguments)]
pub fn run(c: &KernelCase, n: usize, dt: f64, burn: f64, t_end: f64, eps: f64, seed: u64, rec: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (sig, gam): (f64, f64) = (1.0, 1.0);
    let rr = Reset::new(c.reset, c.theta / 2.0);
    let act = |u: f64| activation_expit(u, c.theta);
    let nburn = (burn / dt) as usize;
    let nsteps = (t_end / dt).round() as usize;
    let every = (rec / dt).round() as usize;
    let nrec = nsteps / every;
    let sq = dt.sqrt();
    // per chunk: [sum d at each record, sum d^2 at each record]
    let parts = chunked(n, 64, |range| {
        let mut sums = vec![0.0; 2 * nrec];
        for i in range {
            let mut r = Xoshiro::for_path(seed, i as u64);
            let mut u = 0.5 * r.gauss();
            let mut a = 0.0;
            for _ in 0..nburn {
                let h = c.nu + c.eta * act(u) * (-(-gam * a).exp_m1());
                let v = r.unif();
                let z = r.gauss();
                if v < h * dt {
                    u = rr.apply(u);
                    a = 0.0;
                } else {
                    u = u - c.lam * u * dt + sig * sq * z;
                    a += act(u) * dt;
                }
            }
            let (mut up, mut ap, mut um, mut am) = (u + eps, a, u - eps, a);
            let mut k = 0;
            for s in 0..nsteps {
                let z = r.gauss();
                let v = r.unif();
                for (uu, aa) in [(&mut up, &mut ap), (&mut um, &mut am)] {
                    let h = c.nu + c.eta * act(*uu) * (-(-gam * *aa).exp_m1());
                    if v < h * dt {
                        *uu = rr.apply(*uu);
                        *aa = 0.0;
                    } else {
                        *uu = *uu - c.lam * *uu * dt + sig * sq * z;
                        *aa += act(*uu) * dt;
                    }
                }
                if (s + 1) % every == 0 {
                    let d = (up - um) / (2.0 * eps);
                    sums[k] += d;
                    sums[nrec + k] += d * d;
                    k += 1;
                }
            }
        }
        sums
    });
    let mut tot = vec![0.0; 2 * nrec];
    for p in parts {
        for (t, v) in tot.iter_mut().zip(p) {
            *t += v;
        }
    }
    let nf = n as f64;
    let mut ts = vec![0.0];
    let mut ks = vec![1.0];
    let mut ses = vec![0.0];
    for k in 0..nrec {
        let m = tot[k] / nf;
        let var = (tot[nrec + k] / nf - m * m).max(0.0);
        ts.push(((k + 1) * every) as f64 * dt);
        ks.push(m);
        ses.push(var.sqrt() / nf.sqrt());
    }
    (ts, ks, ses)
}

/// Monte Carlo kernels for the three cases and their comparison with the finite-volume kernel on a
/// fine grid (h = 0.0125, 800 processing cells).
pub fn comparison(n: usize) -> Map<String, Value> {
    let mut res = Map::new();
    for c in cases() {
        let (tm, km, se) = run(&c, n, 0.01, 20.0, 5.0, 0.05, 7, 0.05);
        let p = Params::new(c.theta, c.eta, c.nu, c.lam, 1.0, c.reset);
        let m = Model::new(p, default_grid(&p, 0.0125, 800));
        let (ks, t, k) = analysis::kernel_stats(&m, Some(40.0), 0.005, 4.0, 0.025, None);
        let kfv = interp_many(&tm, &t, &k);
        let z: Vec<f64> = (1..tm.len()).map(|i| (km[i] - kfv[i]) / se[i]).collect();
        let rms = (z.iter().map(|v| v * v).sum::<f64>() / z.len() as f64).sqrt();
        res.insert(c.name.into(), json!({
            "params": {"theta": c.theta, "eta": c.eta, "nu": c.nu, "lam": c.lam, "reset": c.reset.code()},
            "max_abs_diff": km.iter().zip(&kfv).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max),
            "max_abs_z": z.iter().map(|v| v.abs()).fold(0.0, f64::max),
            "rms_z": rms, "fv_kmin": ks.kmin, "fv_tmin": ks.tmin, "fv_int_k": ks.int_k,
            "S": m.s_resolvent(), "mc_int_k": trapezoid(&km, &tm),
            "t": tm, "k_fv": kfv, "k_mc": km, "se": se,
        }));
    }
    res
}

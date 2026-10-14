//! Spot checks of two identities used in the paper (port of `identity_checks.py`).
//!  (1) Proposition 5(c): under an age-only hazard, S = E_pi[(1 - exp(-lam a)) / lam], compared with
//!      the exact renewal values of the episode-solver controls (theta = 0).
//!  (2) Proposition 8: E_pi[U_T W_T] / sig^2 = int_0^T k for the Calvo rule, k(t) = exp(-(lam + nu) t),
//!      by Monte Carlo of the stationary population with exact OU steps.

use super::chunked;
use crate::util::{quad::quad_inf, rng::Xoshiro};
use serde_json::{json, Map, Value};

pub fn checks(controls: &[Value], n: usize) -> Map<String, Value> {
    let mut out = Map::new();
    let mut rows = vec![];
    let mut maxd: f64 = 0.0;
    for c in controls {
        let (lam, eta, nu, gam) = (c["lam"].as_f64().unwrap(), c["eta"].as_f64().unwrap(), c["nu"].as_f64().unwrap(), 1.0);
        let lam_int = |a: f64| nu * a + eta * (a - (1.0 - (-gam * a).exp()) / gam);
        let surv = |a: f64| (-lam_int(a)).exp();
        let etau = quad_inf(surv, 0.0, 1.49e-8, 1.49e-8, 400).0;
        let g = |b: f64| if lam > 0.0 { (1.0 - (-lam * b).exp()) / lam } else { b };
        let s_age = quad_inf(|b| surv(b) * g(b), 0.0, 1.49e-8, 1.49e-8, 400).0 / etau;
        let s_ex = c["exact"]["S"].as_f64().unwrap();
        maxd = maxd.max((s_age - s_ex).abs());
        rows.push(json!({"lam": lam, "eta": eta, "nu": nu, "S_age_formula": s_age, "S_exact": s_ex, "abs_diff": (s_age - s_ex).abs()}));
    }
    out.insert("age_only".into(), json!({"rows": rows, "max_abs_diff": maxd}));

    // (2) fluctuation-response identity, Calvo rule
    let (lam, nu, sig, t_end, dt): (f64, f64, f64, f64, f64) = (1.0, 1.0, 1.0, 3.0, 0.002);
    let e = (-lam * dt).exp();
    let sd = sig * ((1.0 - (-2.0 * lam * dt).exp()) / (2.0 * lam)).sqrt();
    let cov = sig * (1.0 - e) / lam;
    let rho = cov / (sd * dt.sqrt());
    let pr = -(-nu * dt).exp_m1();
    let nsteps = (t_end / dt).round() as usize;
    let parts = chunked(n, 64, |range| {
        let (mut s1, mut s2) = (0.0, 0.0);
        for i in range {
            let mut r = Xoshiro::for_path(11, i as u64);
            let a0 = r.exp1() / nu;
            let mut u = r.gauss() * sig * ((1.0 - (-2.0 * lam * a0).exp()) / (2.0 * lam)).sqrt();
            let mut w = 0.0;
            for _ in 0..nsteps {
                let (z1, z2) = (r.gauss(), r.gauss());
                let dw = dt.sqrt() * z1;
                let noise = sd * (rho * z1 + (1.0 - rho * rho).max(0.0).sqrt() * z2);
                u = e * u + noise;
                w += dw;
                if r.unif() < pr {
                    u = 0.0;
                }
            }
            let est = u * w / sig;
            s1 += est;
            s2 += est * est;
        }
        vec![s1, s2]
    });
    let (s1, s2) = parts.iter().fold((0.0, 0.0), |acc, p| (acc.0 + p[0], acc.1 + p[1]));
    let nf = n as f64;
    let mc = s1 / nf;
    let se = (s2 / nf - mc * mc).max(0.0).sqrt() / nf.sqrt();
    let exact = (1.0 - (-(lam + nu) * t_end).exp()) / (lam + nu);
    out.insert("fluct_response_calvo".into(), json!({"lam": lam, "nu": nu, "T": t_end, "dt": dt, "N": n, "mc": mc, "se": se, "exact": exact}));
    out
}

//! Validation of both solvers and the Monte Carlo checks (the reference `run_validation.py`,
//! `validate.py`, `checks/running_refinement.py`, `run_claim1.py`, `run_mc.py`, `mc_kernel.py`,
//! `compare_kernel_mc.py`, `checks/identity_checks.py`, `checks/perturbation_check.py`).

use super::{log, read_json, write_json, Paths};
use crate::episode::{claim1, run};
use crate::mc;
use crate::stationary::{default_grid, renewal_reference, Model, Params, ResetKind};
use crate::util::{quad::quad_inf, roots::{brentq, BrentOpts}};
use anyhow::Result;
use rayon::prelude::*;
use serde_json::{json, Map, Value};
use std::time::Instant;

const GAM: f64 = 1.0;

fn surv(a: f64, eta: f64, nu: f64) -> f64 {
    (-nu * a - eta * (a + (-GAM * a).exp_m1() / GAM)).exp()
}
fn haz(a: f64, eta: f64, nu: f64) -> f64 {
    nu + eta * (-(-GAM * a).exp_m1())
}
fn gfun(a: f64, r: f64) -> f64 {
    if r == 0.0 { a } else { -(-r * a).exp_m1() / r }
}
fn integral(f: &dyn Fn(f64) -> f64) -> f64 {
    quad_inf(f, 0.0, 1e-13, 1e-12, 400).0
}

/// The base rate that keeps E[tau] = 1 / nu0 when eta > 0.
pub fn matched_nu(eta: f64, nu0: f64) -> f64 {
    if eta == 0.0 {
        return nu0;
    }
    brentq(|n| integral(&|a| surv(a, eta, n)) - 1.0 / nu0, 1e-4, nu0, BrentOpts { xtol: 1e-14, ..Default::default() }).unwrap()
}

/// Exact renewal quantities for theta = 0 (age-only hazard) by quadrature, with E[I U] and E[I^2].
pub fn exact_stats(eta: f64, nu: f64, lam: f64, sig: f64) -> Value {
    let mu = integral(&|a| surv(a, eta, nu));
    let s = integral(&|a| surv(a, eta, nu) * gfun(a, lam)) / mu;
    let v = sig * sig * integral(&|a| surv(a, eta, nu) * gfun(a, 2.0 * lam)) / mu;
    let phi = |a: f64| haz(a, eta, nu) * surv(a, eta, nu);
    let m2 = sig * sig * integral(&|a| phi(a) * gfun(a, 2.0 * lam));
    let m4 = 3.0 * sig.powi(4) * integral(&|a| phi(a) * gfun(a, 2.0 * lam).powi(2));
    let kurt = m4 / (m2 * m2);
    let vint = |a: f64| if lam == 0.0 { sig * sig * a.powi(3) / 3.0 } else { sig * sig / (lam * lam) * (a - 2.0 * gfun(a, lam) + gfun(a, 2.0 * lam)) };
    let eint2 = integral(&|a| phi(a) * vint(a));
    let girs = v / (sig * sig) + (lam / (2.0 * sig * sig)) * eint2 / mu;
    let ciu = |a: f64| if lam == 0.0 { sig * sig * a * a / 2.0 } else { sig * sig / lam * (gfun(a, lam) - gfun(a, 2.0 * lam)) };
    let eiu = integral(&|a| phi(a) * ciu(a));
    json!({"S": s, "V": v, "F": 1.0 / mu, "kurt": kurt, "albl": kurt * mu / 6.0, "girsanov": girs, "EI2": eint2, "EIU": eiu,
           "Etau": mu, "S_formula": (eiu + lam * eint2) / (sig * sig * mu)})
}

/// Controls of the episode solver in the theta = 0 limit (episode_solver_controls.json).
pub fn episode_controls(p: &Paths) -> Result<()> {
    let mut pts = vec![];
    for lam in [0.0, 0.25, 1.0] {
        for eta in [0.0, 0.6, 2.0] {
            pts.push((lam, eta));
        }
    }
    let rows: Vec<Value> = pts
        .par_iter()
        .map(|&(lam, eta)| {
            let nu = matched_nu(eta, 1.0);
            let ex = exact_stats(eta, nu, lam, 1.0);
            let fp = run::full_point(lam, 0.0, eta, nu, &run::FullOpts::default());
            let mut num = Map::new();
            for k in ["SB_formula", "SB_fd", "SP_formula", "SP_fd", "gap_formula", "gap_fd", "VarU", "F", "Kurt", "ALO", "EI2_B", "EIU_B", "Etau"] {
                num.insert(k.into(), fp.get(k).cloned().unwrap_or(Value::Null));
            }
            let mut err = Map::new();
            for k in ["SB_formula", "SB_fd", "VarU", "F", "ALO", "Kurt"] {
                err.insert(k.into(), json!(run::get(&fp, &format!("{k}_err_u")) + run::get(&fp, &format!("{k}_err_b"))));
            }
            log(&format!("control lam={lam} eta={eta}: S exact={:.6} S_B={:.6}", ex["S"].as_f64().unwrap(), run::get(&fp, "SB_fd")));
            json!({"lam": lam, "eta": eta, "nu": nu, "exact": ex, "num": num, "err": err})
        })
        .collect();
    write_json(&p.validation().join("episode_solver_controls.json"), &Value::Array(rows))
}

struct Case {
    name: &'static str,
    p: Params,
    reference: Option<Value>,
    src: &'static str,
}

fn stationary_cases() -> (Vec<Case>, Vec<Case>) {
    let nu06 = run::NU6;
    let nu2 = matched_nu(2.0, 1.0);
    let pr = |theta, eta, nu, lam, r| Params::new(theta, eta, nu, lam, 1.0, r);
    let renewal = vec![
        Case { name: "V1 theta=0 R0 eta=0 nu=1", p: pr(0.0, 0.0, 1.0, 1.0, ResetKind::Zero),
               reference: Some(json!({"S": 0.5, "var": 1.0 / 3.0, "F": 1.0, "kurt_over_6F": 0.6})), src: "exact" },
        Case { name: "V2 theta=0 R0 eta=0.6", p: pr(0.0, 0.6, nu06, 1.0, ResetKind::Zero),
               reference: Some(renewal_reference(0.6, nu06, 1.0, 1.0, 1.0)), src: "renewal quadrature" },
        Case { name: "V3 theta=0 R0 eta=2", p: pr(0.0, 2.0, nu2, 1.0, ResetKind::Zero),
               reference: Some(renewal_reference(2.0, nu2, 1.0, 1.0, 1.0)), src: "renewal quadrature" },
    ];
    let bench = vec![
        Case { name: "V4 eta=0 theta=0.7 Rpl c=0.35 lam=1", p: pr(0.7, 0.0, 1.0, 1.0, ResetKind::Pl),
               reference: Some(json!({"S": 0.64119, "var": 0.36157, "kurt_over_6F": 1.18671, "F": 1.0})), src: "independent 1D solver" },
        Case { name: "V5 eta=0 theta=0.7 Rpl c=0.35 lam=0", p: pr(0.7, 0.0, 1.0, 0.0, ResetKind::Pl),
               reference: Some(json!({"S": 1.35647, "var": 1.09546, "kurt_over_6F": 1.49498, "F": 1.0})), src: "independent 1D solver / closed form" },
        Case { name: "V6 eta=0.6 theta=0.7 Rpl c=0.35 lam=1", p: pr(0.7, 0.6, nu06, 1.0, ResetKind::Pl),
               reference: Some(json!({"S": 0.677, "var": 0.379, "F": 0.728})), src: "Monte Carlo, Euler dt=0.01 (S biased low ~0.3%)" },
        Case { name: "BENCH R0 theta=0.7 eta=0.6 lam=1", p: pr(0.7, 0.6, nu06, 1.0, ResetKind::Zero), reference: None,
               src: "shared benchmark (no reference)" },
    ];
    (renewal, bench)
}

const QTY: [&str; 5] = ["S_res", "S_fd", "var", "F", "kurt_over_6F"];

fn run_case(c: &Case) -> Vec<Map<String, Value>> {
    let hs = [0.025, 0.0125, 0.00625];
    let nas: Vec<usize> = if c.p.eta > 0.0 { vec![200, 400, 800, 1600] } else { vec![0] };
    let mut jobs = vec![];
    for &h in &hs {
        for &na in &nas {
            jobs.push((h, na));
        }
    }
    jobs.par_iter()
        .map(|&(h, na)| {
            let mut g = default_grid(&c.p, h, na.max(1));
            if c.p.eta == 0.0 {
                g.na_f = 0;
            }
            let t0 = Instant::now();
            let m = Model::new(c.p, g);
            let st = m.stats();
            let mut r = Map::new();
            r.insert("h".into(), json!(h));
            r.insert("Na_f".into(), json!(na));
            r.insert("Nu".into(), json!(g.n_u));
            for (k, v) in [("Umax", g.umax), ("Amax", g.amax), ("a_s", g.a_s), ("S_res", m.s_resolvent()), ("S_fd", m.s_fd(1e-3)),
                           ("var", st.var), ("F", st.f), ("kurt_over_6F", st.kurt_over_6f), ("mean", st.mean), ("k0", st.k0),
                           ("lump_mass", st.lump_mass), ("edge_mass", st.edge_mass), ("P_colsum_err", st.p_colsum_err)] {
                r.insert(k.into(), if v.is_finite() { json!(v) } else { Value::Null });
            }
            r.insert("secs".into(), json!(t0.elapsed().as_secs_f64()));
            log(&format!("   {} h={h} Na_f={na} S_res={:.8}", c.name, run::get(&r, "S_res")));
            r
        })
        .collect()
}

/// Richardson: first order in a (finest two a-grids at each h), then second order in u.
fn extrapolate(rows: &[Map<String, Value>], q: &str, eta: f64) -> Value {
    let mut hs: Vec<f64> = rows.iter().map(|r| run::get(r, "h")).collect();
    hs.sort_by(|a, b| b.partial_cmp(a).unwrap());
    hs.dedup();
    let get = |h: f64, n: u64| rows.iter().find(|r| run::get(r, "h") == h && r["Na_f"].as_u64() == Some(n)).map(|r| run::get(r, q)).unwrap();
    let mut out = Map::new();
    let mut sa = std::collections::HashMap::new();
    let hf = hs[hs.len() - 1];
    if eta > 0.0 {
        let mut nas: Vec<u64> = rows.iter().map(|r| r["Na_f"].as_u64().unwrap()).collect();
        nas.sort();
        nas.dedup();
        let k = nas.len();
        let mut sa_prev = std::collections::HashMap::new();
        for &h in &hs {
            sa.insert(h.to_bits(), 2.0 * get(h, nas[k - 1]) - get(h, nas[k - 2]));
            sa_prev.insert(h.to_bits(), 2.0 * get(h, nas[k - 2]) - get(h, nas[k - 3]));
        }
        out.insert("raw_finest".into(), json!(get(hf, nas[k - 1])));
        out.insert("d_a_last".into(), json!(get(hf, nas[k - 1]) - get(hf, nas[k - 2])));
        out.insert("d_u_last".into(), json!(get(hf, nas[k - 1]) - get(hs[hs.len() - 2], nas[k - 1])));
        out.insert("a_extrap_err".into(), json!((sa[&hf.to_bits()] - sa_prev[&hf.to_bits()]).abs() / 3.0));
    } else {
        for &h in &hs {
            sa.insert(h.to_bits(), get(h, 0));
        }
        out.insert("raw_finest".into(), json!(sa[&hf.to_bits()]));
        out.insert("d_a_last".into(), json!(0.0));
        out.insert("d_u_last".into(), json!(sa[&hf.to_bits()] - sa[&hs[hs.len() - 2].to_bits()]));
        out.insert("a_extrap_err".into(), json!(0.0));
    }
    let n = hs.len();
    let s = |h: f64| sa[&h.to_bits()];
    let su = (4.0 * s(hs[n - 1]) - s(hs[n - 2])) / 3.0;
    let su_prev = (4.0 * s(hs[n - 2]) - s(hs[n - 3])) / 3.0;
    out.insert("extrap".into(), json!(su));
    out.insert("u_extrap_err".into(), json!((su - su_prev).abs()));
    let aerr = out["a_extrap_err"].as_f64().unwrap();
    out.insert("err_est".into(), json!(aerr + (su - su_prev).abs().max((s(hs[n - 1]) - s(hs[n - 2])).abs() / 3.0)));
    out.insert("raw_err_est".into(), json!((su - out["raw_finest"].as_f64().unwrap()).abs()));
    Value::Object(out)
}

/// Grid refinement of the stationary solver (stationary_solver_renewal_limit.json and
/// stationary_solver_benchmark.json).
pub fn stationary_refinement(p: &Paths) -> Result<()> {
    let (renewal, bench) = stationary_cases();
    for (cases, file) in [(renewal, "stationary_solver_renewal_limit.json"), (bench, "stationary_solver_benchmark.json")] {
        let mut out = vec![];
        for c in cases {
            log(&format!("refinement {}", c.name));
            let rows = run_case(&c);
            let mut ex = Map::new();
            for q in QTY {
                ex.insert(q.into(), extrapolate(&rows, q, c.p.eta));
            }
            out.push(json!({"name": c.name, "params": c.p, "ref": c.reference, "ref_src": c.src, "rows": rows, "extrap": ex}));
        }
        write_json(&p.validation().join(file), &Value::Array(out))?;
    }
    Ok(())
}

/// Grid refinement of the stationary solver at the running economy (running_refinement.json).
pub fn running_refinement(p: &Paths) -> Result<()> {
    let prm = Params::new(0.7, 5.0, 1.0, 1.0, 1.0, ResetKind::Zero);
    let grids = [(0.02, 160), (0.01, 160), (0.02, 320), (0.01, 320), (0.005, 320), (0.005, 640)];
    let rows: Vec<Value> = grids
        .par_iter()
        .map(|&(h, na)| {
            let m = Model::new(prm, default_grid(&prm, h, na));
            json!({"h": h, "Na_f": na, "S": m.s_resolvent()})
        })
        .collect();
    let get = |h: f64, n: u64| rows.iter().find(|r| r["h"].as_f64() == Some(h) && r["Na_f"].as_u64() == Some(n)).unwrap()["S"].as_f64().unwrap();
    let extrap = 2.0 * get(0.005, 640) - get(0.005, 320);
    write_json(&p.validation().join("running_refinement.json"),
               &json!({"rows": rows, "S_extrap_Na": extrap, "note": "first-order Richardson in 1/Na_f at h = 0.005"}))
}

/// The discrete form of Theorem 3 at five economies (record_identity.json).
pub fn record_identity(p: &Paths) -> Result<()> {
    let rows: Vec<Value> = claim1::points()
        .par_iter()
        .map(|&(lam, theta, eta, nu)| Value::Object(claim1::point(lam, theta, eta, nu)))
        .collect();
    write_json(&p.validation().join("record_identity.json"), &Value::Array(rows))
}

/// Monte Carlo benchmark of the episode solver (mc_benchmark.json).
pub fn mc_benchmark(p: &Paths, npaths: usize) -> Result<()> {
    let r = mc::episode::benchmark(1.0, 0.7, 0.6, run::NU6, npaths, &[0.002]);
    write_json(&p.validation().join("mc_benchmark.json"), &Value::Object(r))
}

/// Monte Carlo kernels against the finite-volume kernel (kernel_mc_comparison.json).
pub fn kernel_mc(p: &Paths, n: usize) -> Result<()> {
    write_json(&p.validation().join("kernel_mc_comparison.json"), &Value::Object(mc::kernel::comparison(n)))
}

/// Age-only formula and the fluctuation-response identity (identity_checks.json); needs the controls.
pub fn identity_checks(p: &Paths, n: usize) -> Result<()> {
    let ctl = read_json(&p.validation().join("episode_solver_controls.json"))?;
    let r = mc::identity::checks(ctl.as_array().unwrap(), n);
    write_json(&p.validation().join("identity_checks.json"), &Value::Object(r))
}

/// First-order perturbation formula (perturbation.json).
pub fn perturbation(p: &Paths, n: usize) -> Result<()> {
    write_json(&p.validation().join("perturbation.json"), &Value::Object(mc::perturbation::check(n)))
}

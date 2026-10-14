//! Drivers for one parameter point: S_B and S_P two ways each, renewal statistics, the discrete
//! identity of Theorem 3, and the extrapolations in the time step, the clock grid and the cell width.
//! Port of the reference `c9run.py`. Results are JSON objects with the reference key names.

use super::*;
use crate::util::{default_l, odd, rich_b, rich_t, rich_u, time_grid};
use serde_json::{json, Map, Value};
use std::time::Instant;

pub const NU6: f64 = 0.7136347625525;
/// Finite-difference half-step in f (truncation error O(delta^2) ~ 1e-7).
pub const DELTA: f64 = 0.002;

/// Settings of `run_point` (defaults as in the reference).
#[derive(Clone, Debug)]
pub struct RunOpts {
    pub nb: usize,
    pub du: f64,
    pub dtmax: f64,
    pub dtmin: f64,
    pub growth: f64,
    pub refs: Vec<usize>,
    pub t_end: Option<f64>,
    pub delta: f64,
    pub mass_tol: f64,
    pub nf_b: usize,
    pub clock: Clock,
}

impl Default for RunOpts {
    fn default() -> Self {
        RunOpts { nb: 200, du: 0.02, dtmax: 0.04, dtmin: 1e-3, growth: 1.1, refs: vec![1, 2, 4], t_end: None,
                  delta: DELTA, mass_tol: 1e-17, nf_b: 1, clock: Clock::Pausing }
    }
}

pub fn horizon(lam: f64, nu: f64) -> f64 {
    if lam > 0.0 { 40.0 / nu } else { 55.0 / nu }
}

pub fn model_for(clock: Clock, lam: f64, theta: f64, eta: f64, nu: f64, l: f64, n_u: usize, nb: usize) -> Model {
    Model::with_clock(clock, lam, theta, eta, nu, l, n_u, nb, 1.0, 1.0)
}

fn num(x: f64) -> Value {
    if x.is_finite() { json!(x) } else { Value::Null }
}

/// Time-integrated solve of B on one grid: statistics and S_B two ways.
pub fn resolvent_point(clock: Clock, lam: f64, theta: f64, eta: f64, nu: f64, du: f64, nb: usize, delta: f64) -> Map<String, Value> {
    let l = default_l(lam);
    let m = model_for(clock, lam, theta, eta, nu, l, odd(2.0 * l / du), nb);
    let r = resolvent_b(&m, 0.0, 3);
    let s = stats_from(&m, &r.source(3));
    let sfd = sb_fd(&m, delta).0;
    let mut o = Map::new();
    o.insert("du".into(), num(m.du));
    o.insert("Nb".into(), json!(nb));
    o.insert("SB_formula".into(), num(s.s_formula.unwrap()));
    o.insert("SB_fd".into(), num(sfd));
    o.insert("Etau".into(), num(s.etau));
    o.insert("F".into(), num(s.f));
    o.insert("VarU".into(), num(s.var_u));
    o.insert("VarU_formula".into(), num(s.var_u_formula));
    o.insert("Kurt".into(), num(s.kurt));
    o.insert("ALO".into(), num(s.alo));
    o.insert("EI2_B".into(), num(r.ei2));
    o.insert("EIU_B".into(), num(r.eiu));
    o.insert("EI2alt_B".into(), num(r.ei2_alt));
    o.insert("EU2int".into(), num(r.eu2int));
    o.insert("mass_out".into(), num(r.mass_out));
    o
}

/// Everything kept from one refinement of `run_point` when the caller asks for it.
pub struct Kept {
    pub rb: Stepped,
    pub hb: Vec<f64>,
    pub rp: Stepped,
}

pub fn get(o: &Map<String, Value>, k: &str) -> f64 {
    o.get(k).and_then(|v| v.as_f64()).unwrap_or(f64::NAN)
}

/// One grid (du, Nb): B by the resolvent, then for each time refinement B time-stepped, hbar = c/p,
/// P time-stepped (formula and finite difference), Richardson in dt.
pub fn run_point(lam: f64, theta: f64, eta: f64, nu: f64, opts: &RunOpts, keep: bool) -> (Map<String, Value>, Vec<(usize, Kept)>, Model) {
    let l = default_l(lam);
    let n_u = odd(2.0 * l / opts.du);
    let t_end = opts.t_end.unwrap_or_else(|| horizon(lam, nu));
    let m = model_for(opts.clock, lam, theta, eta, nu, l, n_u, opts.nb);
    let t0 = Instant::now();
    let mut out = Map::new();
    for (k, v) in [("lam", lam), ("theta", theta), ("eta", eta), ("nu", nu), ("L", l)] {
        out.insert(k.into(), num(v));
    }
    out.insert("Nu".into(), json!(n_u));
    out.insert("Nb".into(), json!(opts.nb));
    for (k, v) in [("du", m.du), ("dtmax", opts.dtmax), ("dtmin", opts.dtmin), ("growth", opts.growth), ("delta", opts.delta)] {
        out.insert(k.into(), num(v));
    }
    out.insert("refs".into(), json!(opts.refs));
    let rb_res = resolvent_b(&m, 0.0, 3);
    let sb = stats_from(&m, &rb_res.source(3));
    let (sbfd, _, _) = sb_fd(&m, opts.delta);
    for (k, v) in [("SB_formula", sb.s_formula.unwrap()), ("SB_fd", sbfd), ("Etau", sb.etau), ("F", sb.f), ("VarU", sb.var_u),
                   ("VarU_formula", sb.var_u_formula), ("Kurt", sb.kurt), ("ALO", sb.alo), ("Ed2", sb.ed2),
                   ("EIU_B", rb_res.eiu), ("EI2_B", rb_res.ei2), ("EI2alt_B", rb_res.ei2_alt), ("EU2int", rb_res.eu2int),
                   ("mass_out_B", rb_res.mass_out)] {
        out.insert(k.into(), num(v));
    }
    let mut runs = Vec::new();
    let mut kept = Vec::new();
    let mut series: std::collections::BTreeMap<&str, Vec<f64>> = Default::default();
    for &rf in &opts.refs {
        let t1 = Instant::now();
        let ts = time_grid(t_end, opts.dtmax, opts.dtmin, opts.growth, rf);
        let rb = ts_b(&m, &ts, 0.0, opts.nf_b, Store::pc(), opts.mass_tol, &[]);
        let (hb, nbad) = make_hbar(&m, &rb.store.p, &rb.store.c, 1e-280);
        let rp = ts_p(&m, &rb.ts, &hb, 0.0, 3, Store::none(), Some((&rb.store.p, &rb.store.c)));
        let sp = stats_from(&m, &rp.source(3));
        let (spfd, _, _) = sp_fd(&m, &rb.ts, &hb, opts.delta);
        let pmax = util::max_of(&rb.store.p);
        let maxabs = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
        let mut d = Map::new();
        d.insert("ref".into(), json!(rf));
        d.insert("nsteps".into(), json!(rb.nsteps));
        d.insert("Tend".into(), num(*rb.ts.last().unwrap()));
        d.insert("mass_left".into(), num(rb.mass_left));
        d.insert("nbad".into(), json!(nbad));
        for (k, v) in [("SP_formula", sp.s_formula.unwrap()), ("SP_fd", spfd), ("EIU_P", rp.eiu), ("EI2_P", rp.ei2),
                       ("EI2alt_P", rp.ei2_alt), ("Etau_P", sp.etau), ("VarU_P", sp.var_u), ("Kurt_P", sp.kurt), ("Etau_Bts", rb.etau),
                       ("maxdiff_p", rp.maxdiff_p), ("maxdiff_c", rp.maxdiff_c), ("pmax", pmax),
                       ("occ_maxdiff", maxabs(&rp.occ, &rb.occ)), ("rev_maxdiff", maxabs(&rp.rev, &rb.rev)),
                       ("surv_maxdiff", maxabs(&rp.surv, &rb.surv)), ("dens_maxdiff", maxabs(&rp.dens, &rb.dens)),
                       ("occ_vs_resolvent", maxabs(&rb.occ, &rb_res.occ))] {
            d.insert(k.into(), num(v));
        }
        if opts.nf_b == 3 {
            d.insert("SB_formula_ts".into(), num(stats_from(&m, &rb.source(3)).s_formula.unwrap()));
        }
        d.insert("secs".into(), num(t1.elapsed().as_secs_f64()));
        series.entry("SP_formula").or_default().push(sp.s_formula.unwrap());
        series.entry("SP_fd").or_default().push(spfd);
        series.entry("EI2_P").or_default().push(rp.ei2);
        series.entry("EIU_P").or_default().push(rp.eiu);
        runs.push(Value::Object(d));
        if keep {
            kept.push((rf, Kept { rb, hb, rp }));
        }
    }
    out.insert("runs".into(), Value::Array(runs));
    for key in ["SP_formula", "SP_fd", "EI2_P", "EIU_P"] {
        let vals = &series[key];
        let (r2, r3) = rich_t(vals, &opts.refs);
        let last = *vals.last().unwrap();
        out.insert(format!("{key}_raw"), num(last));
        out.insert(key.into(), num(r3.or(r2).unwrap_or(last)));
        out.insert(format!("{key}_R2"), num(r2.unwrap_or(f64::NAN)));
        let dterr = match (r2, r3) {
            (Some(a), Some(b)) => (b - a).abs(),
            (Some(a), None) => (a - last).abs(),
            _ => f64::NAN,
        };
        out.insert(format!("{key}_dterr"), num(dterr));
    }
    let gf = get(&out, "SB_formula") - get(&out, "SP_formula");
    let gd = get(&out, "SB_fd") - get(&out, "SP_fd");
    let ge = lam * (get(&out, "EI2_B") - get(&out, "EI2_P")) / (2.0 * m.sig * m.sig * get(&out, "Etau"));
    out.insert("gap_formula".into(), num(gf));
    out.insert("gap_fd".into(), num(gd));
    out.insert("gap_EI2".into(), num(ge));
    out.insert("secs".into(), num(t0.elapsed().as_secs_f64()));
    (out, kept, m)
}

/// Settings of `full_point`.
#[derive(Clone, Debug)]
pub struct FullOpts {
    pub du: f64,
    pub nbs: Vec<usize>,
    pub refs: Vec<usize>,
    pub fine_du: (f64, f64),
    pub fine_nb: (usize, usize),
    pub clock: Clock,
}

impl Default for FullOpts {
    fn default() -> Self {
        FullOpts { du: 0.02, nbs: vec![100, 200], refs: vec![1, 2, 4], fine_du: (0.02, 0.01), fine_nb: (800, 1600), clock: Clock::Pausing }
    }
}

/// Gap from time-stepped runs at (du, Nb in nbs), extrapolated in dt and Nb; S_B from fine resolvent
/// grids extrapolated in du and Nb; S_P = S_B - gap (for the formula and the finite-difference versions).
pub fn full_point(lam: f64, theta: f64, eta: f64, nu: f64, fo: &FullOpts) -> Map<String, Value> {
    let t0 = Instant::now();
    let pts: Vec<Map<String, Value>> = fo
        .nbs
        .iter()
        .map(|&nb| {
            let ro = RunOpts { nb, du: fo.du, refs: fo.refs.clone(), clock: fo.clock, ..Default::default() };
            run_point(lam, theta, eta, nu, &ro, false).0
        })
        .collect();
    let (h1, h2) = fo.fine_du;
    let (n1, n2) = fo.fine_nb;
    let mut res = std::collections::HashMap::new();
    for &h in &[h1, h2] {
        for &nb in &[n1, n2] {
            res.insert((h.to_bits(), nb), resolvent_point(fo.clock, lam, theta, eta, nu, h, nb, DELTA));
        }
    }
    let g = |h: f64, nb: usize, k: &str| get(&res[&(h.to_bits(), nb)], k);
    let mut out = Map::new();
    for (k, v) in [("lam", lam), ("theta", theta), ("eta", eta), ("nu", nu)] {
        out.insert(k.into(), num(v));
    }
    for key in ["SB_formula", "SB_fd", "Etau", "F", "VarU", "VarU_formula", "Kurt", "ALO", "EI2_B", "EIU_B"] {
        let x1 = rich_b(g(h1, n1, key), g(h1, n2, key));
        let x2 = rich_b(g(h2, n1, key), g(h2, n2, key));
        let (hh1, hh2) = (g(h1, n2, "du"), g(h2, n2, "du"));
        let v = if x1.is_nan() { f64::NAN } else { rich_u(x1, hh1, x2, hh2) };
        out.insert(key.into(), num(v));
        out.insert(format!("{key}_err_u"), num(if x1.is_nan() { f64::NAN } else { (v - x2).abs() }));
        out.insert(format!("{key}_err_b"), num(if x1.is_nan() { f64::NAN } else { (x2 - g(h2, n2, key)).abs() }));
    }
    for which in ["formula", "fd"] {
        let gk = format!("gap_{which}");
        let gs: Vec<f64> = pts.iter().map(|p| get(p, &gk)).collect();
        let gap = if gs.len() >= 2 { rich_b(gs[gs.len() - 2], gs[gs.len() - 1]) } else { gs[gs.len() - 1] };
        out.insert(format!("{gk}_byNb"), json!(gs));
        out.insert(gk.clone(), num(gap));
        out.insert(format!("{gk}_err_b"), num((gap - gs[gs.len() - 1]).abs()));
        out.insert(format!("{gk}_err_t"), num(get(pts.last().unwrap(), &format!("SP_{which}_dterr"))));
        let sb = get(&out, &format!("SB_{which}"));
        out.insert(format!("SP_{which}"), num(sb - gap));
    }
    let eg: Vec<f64> = pts.iter().map(|p| get(p, "gap_EI2")).collect();
    out.insert("gap_EI2".into(), num(if eg.len() >= 2 { rich_b(eg[eg.len() - 2], eg[eg.len() - 1]) } else { eg[eg.len() - 1] }));
    let diffs: Vec<f64> = pts.iter().map(|p| get(p, "EI2_B") - get(p, "EI2_P")).collect();
    let dd = if diffs.len() >= 2 { rich_b(diffs[diffs.len() - 2], diffs[diffs.len() - 1]) } else { diffs[diffs.len() - 1] };
    out.insert("EI2_P".into(), num(get(&out, "EI2_B") - dd));
    let maxrun = |key: &str| {
        pts.iter()
            .flat_map(|p| p["runs"].as_array().unwrap().iter().map(|r| r[key].as_f64().unwrap_or(f64::NAN)).collect::<Vec<_>>())
            .fold(0.0, f64::max)
    };
    for (k, rk) in [("claim1_maxdiff_p", "maxdiff_p"), ("claim1_maxdiff_c", "maxdiff_c"), ("claim1_occ_maxdiff", "occ_maxdiff"),
                    ("claim1_rev_maxdiff", "rev_maxdiff"), ("claim1_surv_maxdiff", "surv_maxdiff"), ("claim1_dens_maxdiff", "dens_maxdiff")] {
        out.insert(k.into(), num(maxrun(rk)));
    }
    out.insert("nbad_max".into(), json!(maxrun("nbad") as u64));
    out.insert("raw_points".into(), Value::Array(pts.into_iter().map(Value::Object).collect()));
    let mut rg = Map::new();
    for &h in &[h1, h2] {
        for &nb in &[n1, n2] {
            rg.insert(format!("du={},Nb={}", py_float(h), nb), Value::Object(res[&(h.to_bits(), nb)].clone()));
        }
    }
    out.insert("resolvent_grid".into(), Value::Object(rg));
    out.insert("secs".into(), num(t0.elapsed().as_secs_f64()));
    out
}

/// Python's repr of a float for the keys used in the reference JSON (0.02 -> "0.02", 1.0 -> "1.0").
pub fn py_float(x: f64) -> String {
    let s = format!("{x:?}");
    s
}

//! Research grids (the reference `run_grid.py`, `run_variants.py`, `checks/speed_grid.py`,
//! `scan_crossings.py`, `check_unimodal.py` and the Nyquist certificates).

use super::{log, write_json, Paths};
use crate::episode::{run, speed, Clock};
use crate::stationary::{analysis, default_grid, Model, Params, ResetKind};
use anyhow::Result;
use rayon::prelude::*;
use serde_json::{json, Map, Value};
use std::time::Instant;

pub const THETAS: [f64; 3] = [0.35, 0.7, 1.0];
pub const ETANU: [(f64, f64); 4] = [(0.6, 1.0), (2.0, 1.0), (5.0, 1.0), (0.6, run::NU6)];
pub const LAMS: [f64; 4] = [0.0, 0.25, 1.0, 2.0];

/// The pausing-clock grid of Section 5: for every pull, 12 economies, written to grid_lam{lam}.json.
pub fn grid(p: &Paths, lams: &[f64]) -> Result<()> {
    for &lam in lams {
        let t0 = Instant::now();
        let pts: Vec<(f64, f64, f64)> = THETAS.iter().flat_map(|&th| ETANU.iter().map(move |&(e, n)| (th, e, n))).collect();
        let rows: Vec<Value> = pts
            .par_iter()
            .map(|&(theta, eta, nu)| {
                let o = run::full_point(lam, theta, eta, nu, &run::FullOpts::default());
                log(&format!(
                    "grid lam={lam} theta={theta} eta={eta} nu={nu:.4}: S_B={:.6} S_P={:.6} gap={:+.4}% ({:.0}s)",
                    run::get(&o, "SB_fd"), run::get(&o, "SP_fd"), 100.0 * run::get(&o, "gap_fd") / run::get(&o, "SB_fd"), run::get(&o, "secs")
                ));
                Value::Object(o)
            })
            .collect();
        write_json(&p.grid().join(format!("grid_lam{}.json", py_float(lam))), &Value::Array(rows))?;
        log(&format!("grid lam={lam} done in {:.0}s", t0.elapsed().as_secs_f64()));
    }
    Ok(())
}

fn py_float(x: f64) -> String {
    run::py_float(x)
}

/// History-dependent variants of Section 5: (file stem, clock, pull, etas).
pub fn variant_specs() -> Vec<(String, Clock, f64, Vec<f64>)> {
    vec![
        ("variant_dist_lam0.0".into(), Clock::Dist, 0.0, vec![5.0]),
        ("variant_dist_lam1.0".into(), Clock::Dist, 1.0, vec![2.0, 5.0, 20.0]),
        ("variant_dist_lam2.0".into(), Clock::Dist, 2.0, vec![5.0, 20.0]),
        ("variant_sq_lam1.0".into(), Clock::Sq, 1.0, vec![2.0, 5.0, 20.0]),
        ("variant_distthr_0.5_lam1.0".into(), Clock::DistThr(0.5), 1.0, vec![50.0]),
        ("variant_distthr_1.0_lam0.0".into(), Clock::DistThr(1.0), 0.0, vec![50.0]),
        ("variant_distthr_1.0_lam1.0".into(), Clock::DistThr(1.0), 1.0, vec![50.0]),
    ]
}

pub fn variants(p: &Paths) -> Result<()> {
    for (stem, clock, lam, etas) in variant_specs() {
        let rows: Vec<Value> = etas
            .par_iter()
            .map(|&eta| {
                let fo = run::FullOpts { clock, ..Default::default() };
                let mut o = run::full_point(lam, 0.0, eta, 1.0, &fo);
                o.insert("variant".into(), json!(clock.label()));
                log(&format!("{} lam={lam} eta={eta}: S_B={:.6} S_P={:.6}", clock.label(), run::get(&o, "SB_fd"), run::get(&o, "SP_fd")));
                Value::Object(o)
            })
            .collect();
        write_json(&p.grid().join(format!("{stem}.json")), &Value::Array(rows))?;
    }
    Ok(())
}

/// The mean-lag grid (speed_grid.json).
pub fn speed_grid(p: &Paths) -> Result<()> {
    let pts = speed::grid_points();
    let rows: Vec<Value> = pts
        .par_iter()
        .map(|&(lam, theta, eta, nu)| {
            let r = speed::point(lam, theta, eta, nu);
            log(&format!("speed lam={lam} theta={theta} eta={eta} nu={nu:.4}: gapL={:.3}%", run::get(&r, "gapL_pct")));
            Value::Object(r)
        })
        .collect();
    write_json(&p.grid().join("speed_grid.json"), &Value::Array(rows))
}

/// One point of the stationary scans (coarse grid h = 0.05, 80 processing cells).
pub fn scan_point(theta: f64, eta: f64, nu: f64, lam: f64, reset: ResetKind, gam: f64) -> Value {
    let (h, na) = (0.05, 80);
    let t0 = Instant::now();
    let p = Params::new(theta, eta, nu, lam, gam, reset);
    let g = default_grid(&p, h, na);
    let result = std::panic::catch_unwind(|| {
        let m = Model::new(p, g);
        let st = m.stats();
        let s = m.s_resolvent();
        let fc = analysis::first_crossing(&m, 60.0, 90);
        let mut rec = Map::new();
        for (k, v) in [("theta", theta), ("eta", eta), ("nu", nu), ("lam", lam), ("gam", gam)] {
            rec.insert(k.into(), json!(v));
        }
        rec.insert("reset".into(), json!(reset.code()));
        rec.insert("c".into(), json!(p.c));
        rec.insert("h".into(), json!(h));
        rec.insert("Na_f".into(), json!(na));
        rec.insert("Umax".into(), json!(g.umax));
        rec.insert("N_odd".into(), json!(m.odd.nn));
        for (k, v) in [("S", s), ("invS", 1.0 / s), ("var", st.var), ("F", st.f), ("kurt6F", st.kurt_over_6f),
                       ("share_nonzero", st.share_nonzero), ("lump_mass", st.lump_mass), ("edge_mass", st.edge_mass)] {
            rec.insert(k.into(), if v.is_finite() { json!(v) } else { Value::Null });
        }
        rec.insert("fc_type".into(), json!(fc.kind));
        rec.insert("g_first".into(), json!(fc.g));
        rec.insert("omega_first".into(), json!(fc.omega));
        rec.insert("crossings".into(), json!(fc.crossings));
        rec.insert("min_re_khat".into(), json!(fc.min_re_khat));
        rec.insert("max_abs_khat_wpos".into(), json!(fc.max_abs_khat_w_pos));
        rec.insert("nevals".into(), json!(fc.nevals));
        let (ks, _, _) = analysis::kernel_stats(&m, None, 0.01, 4.0, 0.05, None);
        for (k, v) in [("kmin", ks.kmin), ("tmin", ks.tmin), ("neg_mass", ks.neg_mass), ("int_k", ks.int_k), ("T_kernel", ks.t_end)] {
            rec.insert(k.into(), json!(v));
        }
        rec.insert("first_neg_t".into(), json!(ks.first_neg_t));
        rec
    });
    match result {
        Ok(mut rec) => {
            rec.insert("seconds".into(), json!(t0.elapsed().as_secs_f64()));
            Value::Object(rec)
        }
        Err(e) => {
            let msg = e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default();
            json!({"theta": theta, "eta": eta, "nu": nu, "lam": lam, "reset": reset.code(), "gam": gam, "error": msg})
        }
    }
}

pub fn scan_points(phase: &str) -> Vec<(f64, f64, f64, f64, ResetKind, f64)> {
    let etas = [0.0, 0.6, 2.0, 5.0, 10.0, 20.0];
    let mut pts = vec![];
    if phase == "main" {
        for lam in [1.0, 0.25] {
            for nu in [1.0, 0.2] {
                for eta in etas {
                    pts.push((0.0, eta, nu, lam, ResetKind::Zero, 1.0));
                }
            }
        }
        for theta in [0.35, 0.7, 1.0, 1.5] {
            for lam in [1.0, 0.25] {
                for nu in [1.0, 0.2] {
                    for eta in etas {
                        for r in [ResetKind::Pl, ResetKind::S, ResetKind::Zero] {
                            pts.push((theta, eta, nu, lam, r, 1.0));
                        }
                    }
                }
            }
        }
    } else {
        for theta in [0.7, 1.5, 3.0] {
            for eta in [50.0, 100.0] {
                for nu in [0.2, 0.05] {
                    for gam in [1.0, 10.0] {
                        for r in [ResetKind::Pl, ResetKind::Zero] {
                            pts.push((theta, eta, nu, 1.0, r, gam));
                        }
                    }
                }
            }
        }
    }
    pts
}

/// Scans of the first crossing and of the kernel sign (scan_main.jsonl, scan_extreme.jsonl).
pub fn scans(p: &Paths) -> Result<()> {
    for phase in ["main", "extreme"] {
        let pts = scan_points(phase);
        let rows: Vec<Value> = pts
            .par_iter()
            .map(|&(theta, eta, nu, lam, r, gam)| {
                let v = scan_point(theta, eta, nu, lam, r, gam);
                if let Some(e) = v.get("error") {
                    log(&format!("scan ERR {theta} {eta} {nu} {lam} {} {gam}: {e}", r.code()));
                }
                v
            })
            .collect();
        let s: Vec<String> = rows.iter().map(|r| serde_json::to_string(r).unwrap()).collect();
        std::fs::write(p.grid().join(format!("scan_{phase}.jsonl")), s.join("\n") + "\n")?;
        log(&format!("scan {phase}: {} points", rows.len()));
    }
    Ok(())
}

/// Slice unimodality of the stationary law: for every processing slice, is pi(., a) nonincreasing
/// in |u|? (unimodality.json)
pub fn unimodality(p: &Paths) -> Result<()> {
    let mut pts = vec![];
    for th in [0.35, 0.7, 1.0, 1.5] {
        for eta in [0.6, 5.0, 20.0] {
            for nu in [1.0, 0.2] {
                for lam in [1.0, 0.25] {
                    for r in [ResetKind::Pl, ResetKind::S, ResetKind::Zero] {
                        pts.push((th, eta, nu, lam, r));
                    }
                }
            }
        }
    }
    let rows: Vec<Value> = pts
        .par_iter()
        .map(|&(th, eta, nu, lam, r)| {
            let prm = Params::new(th, eta, nu, lam, 1.0, r);
            let m = Model::new(prm, default_grid(&prm, 0.04, 80));
            let n = m.d.n_u;
            let pos: Vec<usize> = (0..n).filter(|&i| m.d.uc[i] > 0.0).collect();
            let tot: f64 = m.pi.iter().sum();
            let (mut worst, mut badmass): (f64, f64) = (0.0, 0.0);
            for j in 0..m.d.na {
                let row: Vec<f64> = pos.iter().map(|&i| m.pi[j * n + i]).collect();
                let rs: f64 = row.iter().sum();
                if rs < 1e-13 * tot {
                    continue;
                }
                let rmax = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let inc = row.windows(2).map(|w| (w[1] - w[0]).max(0.0)).fold(0.0, f64::max) / rmax;
                if inc > 1e-10 {
                    badmass += 2.0 * rs / tot;
                }
                worst = worst.max(inc);
            }
            json!({"theta": th, "eta": eta, "nu": nu, "lam": lam, "reset": r.code(), "worst_rel_increase": worst, "mass_in_nonmonotone_slices": badmass})
        })
        .collect();
    let nbad = rows.iter().filter(|r| r["worst_rel_increase"].as_f64().unwrap() > 1e-10).count();
    log(&format!("unimodality: {} points, {} with a slice non-monotone in |u|", rows.len(), nbad));
    write_json(&p.grid().join("unimodality.json"), &Value::Array(rows))
}

/// Argument-principle certificates at gS = 0.98 and 1.02 for three economies with partial resets
/// (nyquist_certificates.json).
pub fn certificates(p: &Paths) -> Result<()> {
    let cases = [(0.7, 0.6, run::NU6, 1.0, ResetKind::Pl), (0.35, 20.0, 0.2, 1.0, ResetKind::Pl), (1.5, 20.0, 1.0, 0.25, ResetKind::S)];
    let mut out = vec![];
    for (theta, eta, nu, lam, r) in cases {
        let prm = Params::new(theta, eta, nu, lam, 1.0, r);
        let m = Model::new(prm, default_grid(&prm, 0.02, 160));
        let s = m.s_resolvent();
        let (ks, _, _) = analysis::kernel_stats(&m, None, 0.01, 4.0, 0.05, None);
        for gs in [0.98, 1.02] {
            let c = analysis::certify_count(&m, gs / s, &ks, None, 2.0, 20000).map_err(anyhow::Error::msg)?;
            out.push(json!({"theta": theta, "eta": eta, "nu": nu, "lam": lam, "reset": r.code(), "S": s, "gS": gs, "N": c.n,
                            "N_round": c.n_round, "min_absF": c.min_abs_f, "nsteps": c.nsteps}));
        }
    }
    write_json(&p.grid().join("nyquist_certificates.json"), &Value::Array(out))
}

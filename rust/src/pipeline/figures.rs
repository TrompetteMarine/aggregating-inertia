//! Every number and every data file used by the figures and the text (port of `make_data.py`).
//! Writes data/*.csv (figure data), data/records.json (all quoted numbers with provenance) and the
//! TeX macros tex/values.tex and tex/tipping_values.tex.

use super::{f, log, py_g, read_json, write_csv, write_json, Paths};
use crate::episode::{self, speed, Model as EModel, Store};
use crate::ghf::Ghf;
use crate::stationary::{analysis, default_grid, Model as FvModel, Params, ResetKind, Shifted};
use crate::util::rng::NumpyPcg64;
use crate::util::{
    self, argmin, cumulative_trapezoid, default_l, interp::{interp, Pchip}, linalg::lstsq, linspace, max_of, odd,
    rich_b, rich_t, roots::{brentq, BrentOpts}, tex_sci, time_grid, trapezoid,
};
use anyhow::Result;
use nalgebra::DMatrix;
use rayon::prelude::*;
use serde_json::{json, Map, Value};

// The running economy (Example 1): time in units of the pull, full reset.
pub const LAM: f64 = 1.0;
pub const GAM: f64 = 1.0;
pub const NU: f64 = 1.0;
pub const THETA: f64 = 0.7;
pub const ETA: f64 = 5.0;

fn act(u: f64) -> f64 {
    util::activation(u, THETA)
}

fn hazard(u: f64, a: f64) -> f64 {
    NU + ETA * act(u) * (-(-GAM * a).exp_m1())
}

/// Figure 1: one episode (a prescribed path; processing and hazard computed exactly from it).
pub fn episode(p: &Paths) -> Result<Value> {
    let tc = 8.0;
    let kb = (vec![0.0, 0.6, 1.4, 2.2, 3.1, 3.8, 4.8, 5.4, 6.4, 7.2, 8.0], vec![0.0, 0.38, 0.72, 1.55, 1.3, 0.45, 0.35, 0.9, 1.65, 1.9, 2.15]);
    let ka = (vec![8.0, 8.7, 9.5, 10.2], vec![0.0, 0.22, 0.52, 0.40]);
    let (fb, fa) = (Pchip::new(&kb.0, &kb.1), Pchip::new(&ka.0, &ka.1));
    let t1 = linspace(0.0, tc, 1601);
    let u1: Vec<f64> = t1.iter().map(|&t| fb.eval(t)).collect();
    let t2 = linspace(tc, 10.2, 441);
    let u2: Vec<f64> = t2.iter().map(|&t| fa.eval(t)).collect();
    let a1 = cumulative_trapezoid(&u1.iter().map(|&u| act(u)).collect::<Vec<_>>(), &t1);
    let a2 = cumulative_trapezoid(&u2.iter().map(|&u| act(u)).collect::<Vec<_>>(), &t2);
    let h1: Vec<f64> = u1.iter().zip(&a1).map(|(&u, &a)| hazard(u, a)).collect();
    let h2: Vec<f64> = u2.iter().zip(&a2).map(|(&u, &a)| hazard(u, a)).collect();
    write_csv(&p.data().join("episode_before.csv"), "time,assessment,processing,hazard", &[&t1, &u1, &a1, &h1], 6)?;
    write_csv(&p.data().join("episode_after.csv"), "time,assessment,processing,hazard", &[&t2, &u2, &a2, &h2], 6)?;
    let mut pt = t1.clone();
    let mut pu = u1.clone();
    pt.extend([tc, 0.0]);
    pu.extend([0.0, 0.0]);
    write_csv(&p.data().join("episode_area.csv"), "time,assessment", &[&pt, &pu], 6)?;
    let pause: Vec<f64> = t1.iter().zip(&u1).filter(|(t, u)| **u <= THETA && **t > 3.0 && **t < 6.0).map(|(t, _)| *t).collect();
    let ustar = 0.40;
    let tr1 = brentq(|s| fb.eval(s) - ustar, 3.8, 4.8, BrentOpts::default()).unwrap();
    let tr2 = brentq(|s| fa.eval(s) - ustar, 8.7, 9.5, BrentOpts::default()).unwrap();
    let marks = json!({
        "EpPauseLo": util::min_of(&pause), "EpPauseHi": util::max_of(&pause), "EpTc": tc, "EpUpre": u1[u1.len() - 1],
        "EpApre": a1[a1.len() - 1], "EpTrOne": tr1, "EpTrTwo": tr2, "EpUstar": ustar,
        "EpArOne": interp(tr1, &t1, &a1), "EpArTwo": interp(tr2, &t2, &a2), "EpHazPre": h1[h1.len() - 1],
        "EpHazRone": interp(tr1, &t1, &h1), "EpHazRtwo": interp(tr2, &t2, &h2), "EpArea": trapezoid(&u1, &t1),
    });
    assert!(marks["EpArTwo"].as_f64().unwrap() == 0.0 && marks["EpArOne"].as_f64().unwrap() > 0.5);
    Ok(marks)
}

/// Figure 5: one record, two economies (the running economy B and its projection P).
pub fn record(p: &Paths) -> Result<Value> {
    let l = default_l(LAM);
    let m = EModel::new(LAM, THETA, ETA, NU, l, odd(2.0 * l / 0.02), 200);
    let n = m.n_u;
    let ts = time_grid(40.0 / NU, 0.04, 1e-3, 1.1, 2);
    let tstar = 1.0;
    let nstar = argmin(&ts[1..].iter().map(|t| (t - tstar).abs()).collect::<Vec<_>>());
    let store = Store { p: true, c: true, cm2: true, ..Default::default() };
    let rb = episode::ts_b(&m, &ts, 0.0, 3, store, 1e-16, &[nstar]);
    let ts = rb.ts.clone();
    let (hb, nbad) = episode::make_hbar(&m, &rb.store.p, &rb.store.c, 1e-280);
    let rp = episode::ts_p(&m, &ts, &hb, 0.0, 3, store, Some((&rb.store.p, &rb.store.c)));
    let tt = &ts[1..];
    let du = m.du;
    let (db, dp) = (&rb.dens, &rp.dens);
    let rowsum = |a: &[f64], k: usize| a[k * n..(k + 1) * n].iter().sum::<f64>();
    let eb: Vec<f64> = (0..db.len()).map(|k| rowsum(&rb.store.cm2, k) / db[k].max(1e-300)).collect();
    let ep: Vec<f64> = (0..dp.len()).map(|k| rowsum(&rp.store.cm2, k) / dp[k].max(1e-300)).collect();
    let keep: Vec<usize> = (0..db.len()).filter(|&k| tt[k] <= 6.0).collect();
    let idx: Vec<usize> = keep.iter().step_by(3).cloned().collect();
    let pick = |v: &[f64]| idx.iter().map(|&k| v[k]).collect::<Vec<f64>>();
    write_csv(&p.data().join("record_duration.csv"), "t,durB,durP,areaB,areaP", &[&pick(tt), &pick(db), &pick(dp), &pick(&eb), &pick(&ep)], 7)?;
    let (sb_rev, sp_rev): (f64, f64) = (rb.rev.iter().sum(), rp.rev.iter().sum());
    let rev_b: Vec<f64> = rb.rev.iter().map(|x| x / (sb_rev * du)).collect();
    let rev_p: Vec<f64> = rp.rev.iter().map(|x| x / (sp_rev * du)).collect();
    let sel: Vec<usize> = (0..n).filter(|&i| m.u[i].abs() <= 4.0).collect();
    let neg_u: Vec<f64> = sel.iter().map(|&i| -m.u[i]).collect();
    write_csv(&p.data().join("record_revision.csv"), "delta,revB,revP",
              &[&neg_u, &sel.iter().map(|&i| rev_b[i]).collect::<Vec<_>>(), &sel.iter().map(|&i| rev_p[i]).collect::<Vec<_>>()], 7)?;
    // the hidden clock at age tstar: projected hazard and the spread of the true hazard among survivors
    let q = &rb.snaps.iter().find(|(k, _)| *k == nstar).unwrap().1.q;
    let qsum: f64 = q.iter().sum();
    let mut rows: Vec<[f64; 5]> = vec![];
    for i in 0..n {
        let w: Vec<f64> = (0..m.nb).map(|j| q[j * n + i]).collect();
        let ws: f64 = w.iter().sum();
        if ws < 1e-9 * qsum || m.u[i].abs() > 3.0 {
            continue;
        }
        let hcol: Vec<f64> = (0..m.nb).map(|j| m.h[j * n + i]).collect();
        let hbar: f64 = w.iter().zip(&hcol).map(|(a, b)| a * b).sum::<f64>() / ws;
        let mut order: Vec<usize> = (0..m.nb).collect();
        order.sort_by(|&a, &b| hcol[a].partial_cmp(&hcol[b]).unwrap());
        let mut cw = Vec::with_capacity(m.nb);
        let mut acc = 0.0;
        for &o in &order {
            acc += w[o];
            cw.push(acc / ws);
        }
        let qtile = |p: f64| hcol[order[cw.partition_point(|&c| c < p).min(m.nb - 1)]];
        rows.push([m.u[i], hbar, qtile(0.10), qtile(0.90), ws / du]);
    }
    let colr = |k: usize| rows.iter().map(|r| r[k]).collect::<Vec<f64>>();
    write_csv(&p.data().join("record_hazard.csv"), "u,hbar,q10,q90,surv", &[&colr(0), &colr(1), &colr(2), &colr(3), &colr(4)], 7)?;
    let mut bu = colr(0);
    let mut bh = colr(2);
    bu.extend(rows.iter().rev().map(|r| r[0]));
    bh.extend(rows.iter().rev().map(|r| r[3]));
    write_csv(&p.data().join("record_hazard_band.csv"), "u,h", &[&bu, &bh], 7)?;
    let sb = episode::stats_from(&m, &rb.source(3));
    let sp = episode::stats_from(&m, &rp.source(3));
    let maxabs = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
    Ok(json!({
        "tstar": tt[nstar], "nbad": nbad, "maxdiff_p": rp.maxdiff_p, "maxdiff_c": rp.maxdiff_c, "EI2_B": rb.ei2, "EI2_P": rp.ei2,
        "Etau": rb.etau, "S_B_ts": sb.s_formula, "S_P_ts": sp.s_formula, "F": sb.f, "VarU": sb.var_u, "Kurt": sb.kurt, "ALO": sb.alo,
        "dur_maxdiff": maxabs(db, dp) / max_of(db), "rev_maxdiff": maxabs(&rev_b, &rev_p) / max_of(&rev_b),
    }))
}

/// Figure 6, panels A and B: the grid economies placed by their pull.
pub fn magnitudes(p: &Paths) -> Result<Value> {
    struct Row {
        family: usize,
        lam: f64,
        sb: f64,
        sp: f64,
        alo: f64,
    }
    let mut rows = vec![];
    let mut files: Vec<_> = std::fs::read_dir(p.grid())?.filter_map(|e| e.ok()).map(|e| e.path()).collect();
    files.sort();
    for fname in &files {
        let name = fname.file_name().unwrap().to_string_lossy().to_string();
        if name.starts_with("grid_lam") && name.ends_with(".json") {
            for r in read_json(fname)?.as_array().unwrap() {
                rows.push(Row { family: 0, lam: f(r, "lam"), sb: f(r, "SB_fd"), sp: f(r, "SP_fd"), alo: f(r, "ALO") });
            }
        }
    }
    for fname in &files {
        let name = fname.file_name().unwrap().to_string_lossy().to_string();
        if name.starts_with("variant_") && name.ends_with(".json") {
            for r in read_json(fname)?.as_array().unwrap() {
                let v = r["variant"].as_str().unwrap();
                let fam = match v {
                    "dist" => 1,
                    "sq" => 2,
                    _ => 3,
                };
                rows.push(Row { family: fam, lam: f(r, "lam"), sb: f(r, "SB_fd"), sp: f(r, "SP_fd"), alo: f(r, "ALO") });
            }
        }
    }
    let xpos = |lam: f64| [0.0, 0.25, 1.0, 2.0].iter().position(|&x| x == lam).unwrap() as f64;
    let mut out: Vec<[f64; 5]> = vec![];
    // horizontal jitter for legibility, with the reference pipeline's generator (numpy's default_rng(3))
    let mut rng = NumpyPcg64::default_rng(3);
    for r in rows.iter() {
        let gap = 100.0 * (r.sb - r.sp) / r.sb;
        let kerr = 100.0 * (r.alo - r.sb) / r.sb;
        let x = xpos(r.lam) + if r.family != 0 { 0.18 } else { -0.12 } + 0.1 * (rng.random() - 0.5);
        out.push([x, r.lam, r.family as f64, gap, kerr]);
    }
    let c = |k: usize| out.iter().map(|r| r[k]).collect::<Vec<f64>>();
    write_csv(&p.data().join("magnitudes.csv"), "x,lam,family,gap,kerr", &[&c(0), &c(1), &c(2), &c(3), &c(4)], 6)?;
    let mut summ = Map::new();
    for lam in [0.0, 0.25, 1.0, 2.0] {
        let cl: Vec<&[f64; 5]> = out.iter().filter(|r| r[2] == 0.0 && r[1] == lam).collect();
        let g: Vec<f64> = cl.iter().map(|r| r[3]).collect();
        let k: Vec<f64> = cl.iter().map(|r| r[4]).collect();
        summ.insert(format!("{lam:?}"), json!({"n": cl.len(), "gap_min": util::min_of(&g), "gap_max": max_of(&g),
                                               "kerr_min": util::min_of(&k), "kerr_max": max_of(&k)}));
    }
    let other: Vec<&[f64; 5]> = out.iter().filter(|r| r[2] > 0.0).collect();
    let og: Vec<f64> = other.iter().filter(|r| r[1] > 0.0).map(|r| r[3]).collect();
    let o0: Vec<f64> = other.iter().filter(|r| r[1] == 0.0).map(|r| r[3].abs()).collect();
    summ.insert("other".into(), json!({"n": other.len(), "gap_min": util::min_of(&og), "gap_max": max_of(&og), "lam0_max_abs": max_of(&o0)}));
    summ.insert("all_lam_pos_max_abs_gap".into(), json!(out.iter().filter(|r| r[1] > 0.0).map(|r| r[3].abs()).fold(0.0, f64::max)));
    summ.insert("all_lam0_max_abs_gap".into(), json!(out.iter().filter(|r| r[1] == 0.0).map(|r| r[3].abs()).fold(0.0, f64::max)));
    summ.insert("n_total".into(), json!(out.len()));
    Ok(Value::Object(summ))
}

/// Figure 6, panel C, and the speed numbers, from data/grid/speed_grid.json.
pub fn speed_magnitudes(p: &Paths) -> Result<Value> {
    let rows = read_json(&p.grid().join("speed_grid.json"))?;
    let rows = rows.as_array().unwrap();
    let xpos = |lam: f64| [0.0, 0.25, 1.0, 2.0].iter().position(|&x| x == lam).unwrap() as f64;
    let mut rng = NumpyPcg64::default_rng(5);
    let out: Vec<[f64; 3]> = rows.iter().map(|r| [xpos(f(r, "lam")) - 0.12 + 0.1 * (rng.random() - 0.5), f(r, "lam"), f(r, "gapL_pct")]).collect();
    let c = |k: usize| out.iter().map(|r| r[k]).collect::<Vec<f64>>();
    write_csv(&p.data().join("magnitudes_lag.csv"), "x,lam,gapL", &[&c(0), &c(1), &c(2)], 6)?;
    let a: Vec<f64> = out.iter().map(|r| r[2].abs()).collect();
    let pick = |pred: &dyn Fn(f64) -> bool| -> Vec<f64> { out.iter().zip(&a).filter(|(r, _)| pred(r[1])).map(|(_, v)| *v).collect() };
    let (z, ps) = (pick(&|l| l == 0.0), pick(&|l| l > 0.0));
    let run = rows.iter().find(|r| f(r, "lam") == LAM && f(r, "theta") == THETA && f(r, "eta") == ETA && (f(r, "nu") - NU).abs() < 1e-12).unwrap();
    Ok(json!({
        "n": rows.len(), "zero": [util::min_of(&z), max_of(&z)], "pos": [util::min_of(&ps), max_of(&ps)], "all": [util::min_of(&a), max_of(&a)],
        "all_negative": out.iter().all(|r| r[2] < 0.0), "relerr": rows.iter().map(|r| f(r, "gapM1_relerr")).fold(0.0, f64::max),
        "run": {"LB": f(run, "LB"), "LP": f(run, "LP"), "gapL_pct": f(run, "gapL_pct"), "SB": f(run, "SB"), "SP": f(run, "SP")},
    }))
}

/// Figure 4, panel D: branches of stationary states m = M(g c tanh(m / c)) for two generalised hazards.
pub const GHF_NU: f64 = 0.5;
pub const GHF_ZETA: f64 = 3.0;
pub const GHF_ELL: f64 = 0.7;
pub const GHF_C: f64 = 1.5;
pub const GHF_L: f64 = 12.0;
pub const GHF_H: f64 = 0.02;

pub fn ghf_rule(name: &str) -> impl Fn(f64) -> f64 + Sync + Copy {
    let up = name == "up";
    move |u: f64| {
        let e = (-u * u / (2.0 * GHF_ELL * GHF_ELL)).exp();
        if up { GHF_NU + GHF_ZETA * (1.0 - e) } else { GHF_NU + GHF_ZETA * e }
    }
}

pub fn bifurcation(p: &Paths) -> Result<Value> {
    let c = GHF_C;
    let mut fs = util::geomspace(1e-3, 0.5, 30);
    fs.extend(linspace(0.52, 9.0, 120));
    let ts = linspace(0.0, 12.0, 241);
    let mut out = Map::new();
    for name in ["up", "down"] {
        let rule = ghf_rule(name);
        let m = Ghf::new(rule, 1.0, 1.0, GHF_L, GHF_H);
        let s = m.s_tilted(0.0);
        let ms: Vec<f64> = fs.par_iter().map(|&x| m.m(x)).collect();
        let dm: Vec<f64> = fs.par_iter().map(|&x| (m.m(x + 1e-5) - m.m(x - 1e-5)) / 2e-5).collect();
        let g: Vec<f64> = fs.iter().zip(&ms).map(|(x, mm)| x / (c * (mm / c).tanh())).collect();
        let slope: Vec<f64> = (0..fs.len()).map(|i| dm[i] * g[i] / (ms[i] / c).cosh().powi(2)).collect();
        let gs: Vec<f64> = g.iter().map(|x| x * s).collect();
        write_csv(&p.data().join(format!("tip_branch_{name}.csv")), "gS,m,slope", &[&gs, &ms, &slope], 7)?;
        // checks at stable branch points: kernel nonnegative and its integral equal to M'(f)
        let stable: Vec<usize> = (0..fs.len()).filter(|&i| slope[i] < 1.0).collect();
        let pick: Vec<usize> = if stable.is_empty() {
            vec![]
        } else {
            linspace(0.0, (stable.len() - 1) as f64, 6).iter().map(|x| stable[x.round() as usize]).collect()
        };
        let mc = Ghf::new(rule, 1.0, 1.0, GHF_L, 0.04);
        let mut kmin = f64::INFINITY;
        let mut intrel: f64 = 0.0;
        for &i in &pick {
            let k = mc.kernel(fs[i], &ts);
            kmin = kmin.min(util::min_of(&k));
            intrel = intrel.max((m.s_tilted(fs[i]) - dm[i]).abs() / dm[i]);
        }
        kmin = kmin.min(util::min_of(&mc.kernel(0.0, &ts)));
        let sel: Vec<usize> = (0..fs.len()).filter(|&i| fs[i] > 0.04 && fs[i] < 0.31).collect();
        let mut a = DMatrix::<f64>::zeros(sel.len(), 3);
        let mut y = vec![0.0; sel.len()];
        for (r, &i) in sel.iter().enumerate() {
            a[(r, 0)] = fs[i];
            a[(r, 1)] = fs[i].powi(3);
            a[(r, 2)] = fs[i].powi(5);
            y[r] = ms[i];
        }
        let coef = lstsq(&a, &y);
        let mut e = json!({"S": s, "M3": 6.0 * coef[1], "fold_gS": util::min_of(&gs), "kmin": kmin, "int_rel": intrel,
                           "slope_min": util::min_of(&slope), "slope_max": max_of(&slope)});
        if name == "down" {
            let i = argmin(&g);
            let upper: Vec<usize> = (i + 1..fs.len()).collect();
            let ux: Vec<f64> = upper.iter().map(|&k| gs[k]).collect();
            let uy: Vec<f64> = upper.iter().map(|&k| ms[k]).collect();
            e["fold_m"] = json!(ms[i]);
            e["jump_m"] = json!(interp(1.0, &ux, &uy));
        }
        out.insert(name.into(), e);
    }
    let calvo = Ghf::new(|_| GHF_NU, 1.0, 1.0, GHF_L, GHF_H);
    let err = [0.1, 1.0, 4.0].iter().map(|&x| (calvo.m(x) * (1.0 + GHF_NU) / x - 1.0).abs()).fold(0.0, f64::max);
    out.insert("calvo_rel_err".into(), json!(err));
    Ok(Value::Object(out))
}

fn fit_m3(fs: &[f64], ms: &[f64]) -> (f64, f64) {
    let mut a = DMatrix::<f64>::zeros(fs.len(), 3);
    for (r, &x) in fs.iter().enumerate() {
        a[(r, 0)] = x;
        a[(r, 1)] = x.powi(3);
        a[(r, 2)] = x.powi(5);
    }
    let c = lstsq(&a, ms);
    (c[0], 6.0 * c[1])
}

/// Third derivative of M at zero for B (time-integrated solve) and for P (Richardson in dt),
/// extrapolated in Nb; at lam = 0 also the cross-section and record formulas of Proposition 5.
pub fn shape(lam: f64) -> Value {
    let (nbs, refs) = ([100usize, 200], [1usize, 2, 4]);
    let fs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3];
    let per: Vec<Map<String, Value>> = nbs
        .par_iter()
        .map(|&nb| {
            let l = default_l(lam);
            let m = EModel::new(lam, THETA, ETA, NU, l, odd(2.0 * l / 0.02), nb);
            let mb: Vec<f64> = fs.iter().map(|&x| {
                let r = episode::resolvent_b(&m, x, 1);
                r.ei / r.etau
            }).collect();
            let (sb, m3b) = fit_m3(&fs, &mb);
            let mut vals: std::collections::BTreeMap<&str, Vec<f64>> = Default::default();
            for &rf in &refs {
                let t_end = if lam > 0.0 { 40.0 / NU } else { 55.0 / NU };
                let ts0 = time_grid(t_end, 0.04, 1e-3, 1.1, rf);
                let rb = episode::ts_b(&m, &ts0, 0.0, 1, Store::pc(), 1e-17, &[]);
                let ts = rb.ts.clone();
                let (hb, _) = episode::make_hbar(&m, &rb.store.p, &rb.store.c, 1e-280);
                let mp: Vec<f64> = fs.iter().map(|&x| {
                    let r = episode::ts_p(&m, &ts, &hb, x, 1, Store::none(), None);
                    r.ei / r.etau
                }).collect();
                let (spv, m3p) = fit_m3(&fs, &mp);
                vals.entry("M3P").or_default().push(m3p);
                vals.entry("SP").or_default().push(spv);
                if lam == 0.0 {
                    let n = m.n_u;
                    let (pp, cc, u) = (&rb.store.p, &rb.store.c, &m.u);
                    let rows = pp.len() / n;
                    let tt = &ts[1..rows + 1];
                    let dt: Vec<f64> = (0..rows).map(|k| ts[k + 1] - ts[k]).collect();
                    let wsum = |a: &[f64], k: usize, pw: i32| -> f64 { (0..n).map(|i| a[k * n + i] * u[i].powi(pw)).sum() };
                    let tot = |g: &dyn Fn(usize) -> f64| -> f64 { (0..rows).map(g).sum() };
                    let et = tot(&|k| dt[k] * wsum(pp, k, 0));
                    let v = tot(&|k| dt[k] * wsum(pp, k, 2)) / et;
                    let eu4 = tot(&|k| dt[k] * wsum(pp, k, 4)) / et;
                    let eau2 = tot(&|k| dt[k] * tt[k] * wsum(pp, k, 2)) / et;
                    let ea = tot(&|k| dt[k] * tt[k] * wsum(pp, k, 0)) / et;
                    vals.entry("cross").or_default().push(eu4 - 3.0 * eau2 - 3.0 * v * (v - ea));
                    let ms = tot(&|k| dt[k] * wsum(cc, k, 0));
                    let ed6 = tot(&|k| dt[k] * wsum(cc, k, 6)) / ms;
                    let ed4 = tot(&|k| dt[k] * wsum(cc, k, 4)) / ms;
                    let etd4 = tot(&|k| dt[k] * tt[k] * wsum(cc, k, 4)) / ms;
                    let et2 = tot(&|k| dt[k] * tt[k] * tt[k] * wsum(cc, k, 0)) / ms;
                    let etr = tot(&|k| dt[k] * tt[k] * wsum(cc, k, 0)) / ms;
                    let sr = ed4 / (6.0 * etr);
                    vals.entry("record").or_default().push((ed6 - 5.0 * etd4) / (10.0 * etr) - 3.0 * sr * (sr - et2 / (2.0 * etr)));
                }
            }
            let mut o = Map::new();
            o.insert("SB".into(), json!(sb));
            o.insert("M3B".into(), json!(m3b));
            for (k, v) in vals {
                o.insert(k.into(), json!(rich_t(&v, &refs).1.unwrap()));
            }
            o
        })
        .collect();
    let mut ex = Map::new();
    for k in per[1].keys() {
        ex.insert(k.clone(), json!(rich_b(per[0][k].as_f64().unwrap(), per[1][k].as_f64().unwrap())));
    }
    ex.insert("raw".into(), json!({"100": per[0], "200": per[1]}));
    Value::Object(ex)
}

pub fn tipping(p: &Paths) -> Result<Value> {
    let bif = bifurcation(p)?;
    log("tipping: bifurcation done");
    let shape_run = shape(LAM);
    log("tipping: shape (running economy) done");
    let shape0 = shape(0.0);
    log("tipping: shape (lam = 0) done");
    let l = default_l(LAM);
    let m = EModel::new(LAM, THETA, ETA, NU, l, odd(2.0 * l / 0.02), 200);
    let m1b = speed::s_m1(&speed::resolvent_moments(&m, 0.0), LAM, 1.0).1;
    Ok(json!({"bifurcation": bif, "shape_run": shape_run, "shape_lam0": shape0, "M1B_base": m1b}))
}

/// Figure 4: kernels, Nyquist curves and dominant roots for eta = 0, 1, 5, 20; certificates.
pub fn stability(p: &Paths) -> Result<Value> {
    let etas = [0.0, 1.0, 5.0, 20.0];
    struct Res {
        eta: f64,
        model: FvModel,
        s: f64,
        m1: f64,
        gap: f64,
        ks: analysis::KernelStats,
        t: Vec<f64>,
        k: Vec<f64>,
        rc: analysis::RootCurve,
        fc: analysis::FirstCrossing,
    }
    let results: Vec<Res> = etas
        .par_iter()
        .map(|&eta| {
            let prm = Params::new(THETA, eta, NU, LAM, GAM, ResetKind::Zero);
            let model = FvModel::new(prm, default_grid(&prm, 0.02, 160));
            let s = model.s_resolvent();
            let sh0 = Shifted::<f64>::new(&model.odd, 0.0, false);
            let x1 = sh0.solve(&model.odd, &model.b.iter().map(|v| -v).collect::<Vec<_>>());
            let x2 = sh0.solve(&model.odd, &x1.iter().map(|v| -v).collect::<Vec<_>>());
            let m1 = model.odd.inner_u(&x2);
            let gap = analysis::slowest_rate(&model);
            let (ks, t, k) = analysis::kernel_stats(&model, None, 0.01, 4.0, 0.05, Some(gap));
            let rc = analysis::dominant_root_curve(&model, s, m1, &linspace(0.6, 1.4, 41), gap);
            let fc = analysis::first_crossing(&model, 80.0, 90);
            log(&format!("stability eta={eta}: S={s:.4} Lbar={:.3} gap={gap:.3}", m1 / s));
            Res { eta, model, s, m1, gap, ks, t, k, rc, fc }
        })
        .collect();
    let mut res = Map::new();
    for r in &results {
        let (ws, vals) = &r.fc.curve;
        let _ = ws;
        res.insert(format!("eta{}", py_g(r.eta)), json!({
            "S": r.s, "M1": r.m1, "Lbar": r.m1 / r.s, "gap": r.gap, "lin_err": r.rc.lin_err, "kmin": r.ks.kmin, "first": r.fc.kind,
            "g_first": r.fc.g, "crossings": r.fc.crossings, "min_re": vals.iter().map(|v| v.re).fold(f64::INFINITY, f64::min) / r.s,
        }));
    }
    let tg = linspace(0.0, 6.0, 241);
    let kc: Vec<Vec<f64>> = results.iter().map(|r| tg.iter().map(|&t| interp(t, &r.t, &r.k)).collect()).collect();
    write_csv(&p.data().join("stab_kernels.csv"), "t,k0,k1,k5,k20", &[&tg, &kc[0], &kc[1], &kc[2], &kc[3]], 7)?;
    for r in &results {
        let (ws, vals) = &r.fc.curve;
        let ok: Vec<usize> = (0..ws.len()).filter(|&i| ws[i] <= 60.0).collect();
        let re: Vec<f64> = ok.iter().map(|&i| vals[i].re / r.s).collect();
        let im: Vec<f64> = ok.iter().map(|&i| vals[i].im / r.s).collect();
        write_csv(&p.data().join(format!("stab_nyquist_eta{}.csv", py_g(r.eta))), "re,im", &[&re, &im], 7)?;
    }
    let gs = &results[0].rc.gs;
    let mut cols: Vec<Vec<f64>> = vec![gs.clone()];
    for r in &results {
        cols.push(r.rc.x.clone());
    }
    for r in &results {
        cols.push(gs.iter().map(|g| (g - 1.0) / (r.rc.m1 / r.rc.s)).collect());
    }
    let refs: Vec<&[f64]> = cols.iter().map(|c| c.as_slice()).collect();
    write_csv(&p.data().join("tip_rate.csv"), "gS,x0,x1,x5,x20,l0,l1,l5,l20", &refs, 7)?;
    // certificates for the running economy: no root in Re z > 0 below g = 1/S, one above
    let run = results.iter().find(|r| r.eta == ETA).unwrap();
    let mut certs = Map::new();
    for fac in [0.98, 1.02] {
        let c = analysis::certify_count(&run.model, fac / run.s, &run.ks, None, 2.0, 20000).map_err(anyhow::Error::msg)?;
        certs.insert(format!("{fac:?}"), json!({"N": c.n, "min_absF": c.min_abs_f}));
    }
    let grid = read_json(&p.grid().join("grid_lam1.0.json"))?;
    let rg = grid.as_array().unwrap().iter().find(|r| f(r, "theta") == THETA && f(r, "eta") == ETA && (f(r, "nu") - NU).abs() < 1e-12).unwrap();
    let gstar = 0.5 * (1.0 / f(rg, "SB_fd") + 1.0 / f(rg, "SP_fd"));
    let c = analysis::certify_count(&run.model, gstar, &run.ks, None, 2.0, 20000).map_err(anyhow::Error::msg)?;
    certs.insert("gstar".into(), json!({"g": gstar, "N": c.n, "min_absF": c.min_abs_f, "gS_fv": gstar * run.s}));
    res.insert("certificates".into(), Value::Object(certs));
    res.insert("S_run_fv".into(), json!(run.s));
    Ok(Value::Object(res))
}

pub fn scan_summary(p: &Paths) -> Result<(Value, Value)> {
    let mut rows = vec![];
    for ph in ["main", "extreme"] {
        let s = std::fs::read_to_string(p.grid().join(format!("scan_{ph}.jsonl")))?;
        for l in s.lines().filter(|l| !l.trim().is_empty()) {
            rows.push(serde_json::from_str::<Value>(&l.replace("NaN", "null"))?);
        }
    }
    let nreal = rows.iter().filter(|r| r["fc_type"] == "real").count();
    let minkmin = rows.iter().filter_map(|r| r["kmin"].as_f64()).fold(f64::INFINITY, f64::min);
    let npos = rows.iter().filter(|r| r["crossings"].as_array().map(|c| c.iter().any(|x| x[0].as_f64().unwrap() > 0.0 && x[1].as_f64().unwrap() > 0.0)).unwrap_or(false)).count();
    let uni = read_json(&p.grid().join("unimodality.json"))?;
    let uni = uni.as_array().unwrap();
    let nok = uni.iter().filter(|u| f(u, "worst_rel_increase") <= 1e-10).count();
    Ok((json!({"n": rows.len(), "n_real": nreal, "min_kmin": minkmin, "n_pos_axis_crossings": npos}), json!({"n": uni.len(), "n_ok": nok})))
}

/// All of make_data.py: figure data, records.json and the two macro files.
pub fn run_all(p: &Paths) -> Result<()> {
    let mut rec = Map::new();
    rec.insert("episode".into(), episode(p)?);
    log("figures: episode done");
    rec.insert("magnitudes".into(), magnitudes(p)?);
    let (scan, uni) = scan_summary(p)?;
    rec.insert("scan".into(), scan);
    rec.insert("unimodal".into(), uni);
    rec.insert("record".into(), record(p)?);
    log("figures: record done");
    rec.insert("stability".into(), stability(p)?);
    log("figures: stability done");
    rec.insert("tipping".into(), tipping(p)?);
    let tp = rec["tipping"].clone();
    rec.insert("bifurcation".into(), tp["bifurcation"].clone());
    rec.insert("speed".into(), speed_magnitudes(p)?);
    write_values(p, &mut rec)?;
    write_tipping_values(p, &rec)?;
    write_json(&p.data().join("records.json"), &Value::Object(rec))?;
    Ok(())
}

fn fmt(x: f64, d: usize) -> String {
    format!("{:.*}", d, x)
}

pub fn write_values(p: &Paths, rec: &mut Map<String, Value>) -> Result<()> {
    let grid = read_json(&p.grid().join("grid_lam1.0.json"))?;
    let run = grid.as_array().unwrap().iter().find(|r| f(r, "theta") == THETA && f(r, "eta") == ETA && (f(r, "nu") - NU).abs() < 1e-12).unwrap().clone();
    let (sb, sp, var, ff, alo) = (f(&run, "SB_fd"), f(&run, "SP_fd"), f(&run, "VarU"), f(&run, "F"), f(&run, "ALO"));
    let gstar = 0.5 * (1.0 / sb + 1.0 / sp);
    rec.insert("running".into(), json!({"SB": sb, "SP": sp, "VarU": var, "F": ff, "ALO": alo, "gstar": gstar, "gSB": gstar * sb,
        "gSP": gstar * sp, "critB": 1.0 / sb, "critP": 1.0 / sp, "critALO": 1.0 / alo, "SB_formula": f(&run, "SB_formula"),
        "SP_formula": f(&run, "SP_formula")}));
    let (mag, st, ep, rc, sc, un) = (&rec["magnitudes"], &rec["stability"], &rec["episode"], &rec["record"], &rec["scan"], &rec["unimodal"]);
    let m = |lam: &str, k: &str| mag[lam][k].as_f64().unwrap();
    let cert = |k: &str| st["certificates"][k]["N"].as_f64().unwrap().round().abs() as i64;
    let v: Vec<(&str, String)> = vec![
        ("RunLam", py_g(LAM)), ("RunTheta", py_g(THETA)), ("RunEta", py_g(ETA)), ("RunNu", py_g(NU)),
        ("RunSB", fmt(sb, 4)), ("RunSP", fmt(sp, 4)), ("RunVar", fmt(var, 4)), ("RunF", fmt(ff, 3)),
        ("RunALO", fmt(alo, 4)), ("RunCritB", fmt(1.0 / sb, 3)), ("RunCritP", fmt(1.0 / sp, 3)),
        ("RunCritALO", fmt(1.0 / alo, 3)), ("RunGstar", fmt(gstar, 3)), ("RunGSB", fmt(gstar * sb, 4)),
        ("RunGSP", fmt(gstar * sp, 4)), ("RunGapPct", fmt(100.0 * (sb - sp) / sb, 2)), ("RunGapPctAbs", fmt(100.0 * (sb - sp).abs() / sb, 2)),
        ("RunALOPct", fmt(100.0 * (alo - sb) / sb, 0)), ("RunTstar", fmt(f(rc, "tstar"), 1)),
        ("RunDurDiff", tex_sci(f(rc, "dur_maxdiff").max(f(rc, "rev_maxdiff")))), ("RunEItwoB", fmt(f(rc, "EI2_B"), 4)),
        ("RunEItwoP", fmt(f(rc, "EI2_P"), 4)), ("RunSFV", fmt(f(st, "S_run_fv"), 4)),
        ("MagClockQGapMin", fmt(m("0.25", "gap_max").abs(), 2)), ("MagClockQGapMax", fmt(m("0.25", "gap_min").abs(), 2)),
        ("MagClockOneGapMin", fmt(m("1.0", "gap_max").abs(), 2)), ("MagClockOneGapMax", fmt(m("1.0", "gap_min").abs(), 2)),
        ("MagClockTwoGapMin", fmt(m("2.0", "gap_max").abs(), 3)), ("MagClockTwoGapMax", fmt(m("2.0", "gap_min").abs(), 2)),
        ("MagKerrOneMin", fmt(m("1.0", "kerr_min"), 0)), ("MagKerrOneMax", fmt(m("1.0", "kerr_max"), 0)),
        ("MagKerrTwoMin", fmt(m("2.0", "kerr_min"), 0)), ("MagKerrTwoMax", fmt(m("2.0", "kerr_max"), 0)),
        ("MagKerrQMin", fmt(m("0.25", "kerr_min"), 0)), ("MagKerrQMax", fmt(m("0.25", "kerr_max"), 0)),
        ("MagMaxGap", fmt(f(mag, "all_lam_pos_max_abs_gap"), 1)), ("MagN", format!("{}", mag["n_total"])),
        ("MagOtherGapMax", fmt(mag["other"]["gap_min"].as_f64().unwrap().abs(), 1)),
        ("ScanN", format!("{}", sc["n"])), ("ScanReal", format!("{}", sc["n_real"])), ("UniN", format!("{}", un["n"])), ("UniOK", format!("{}", un["n_ok"])),
        ("EpTau", py_g(f(ep, "EpTc"))),
        ("CertN", format!("{}", cert("gstar"))), ("CertMinF", fmt(st["certificates"]["gstar"]["min_absF"].as_f64().unwrap(), 3)),
        ("CertNlo", format!("{}", cert("0.98"))), ("CertNhi", format!("{}", cert("1.02"))),
        ("StabSzero", fmt(st["eta0"]["S"].as_f64().unwrap(), 4)), ("StabSone", fmt(st["eta1"]["S"].as_f64().unwrap(), 4)),
        ("StabSfive", fmt(st["eta5"]["S"].as_f64().unwrap(), 4)), ("StabStwenty", fmt(st["eta20"]["S"].as_f64().unwrap(), 4)),
    ];
    let mut lines = vec!["% Generated by inertia (Rust port of code/make_data.py); do not edit.".to_string()];
    for (k, s) in v {
        lines.push(format!("\\def\\{k}{{{s}}}"));
    }
    for k in ["EpPauseLo", "EpPauseHi", "EpTc", "EpUpre", "EpApre", "EpTrOne", "EpTrTwo", "EpUstar", "EpArOne", "EpArTwo", "EpHazPre", "EpHazRone", "EpHazRtwo", "EpArea"] {
        lines.push(format!("\\def\\{k}{{{:.6}}}", f(ep, k)));
    }
    std::fs::write(p.tex().join("values.tex"), lines.join("\n") + "\n")?;
    Ok(())
}

pub fn write_tipping_values(p: &Paths, rec: &Map<String, Value>) -> Result<()> {
    let (st, bf, tp, sp, run) = (&rec["stability"], &rec["bifurcation"], &rec["tipping"], &rec["speed"], &rec["running"]);
    let (sr, s0) = (&tp["shape_run"], &tp["shape_lam0"]);
    let (m3b, m3p) = (f(sr, "M3B"), f(sr, "M3P"));
    let amp = 100.0 * (((f(run, "SP") / f(run, "SB")).powi(3) * m3b / m3p).sqrt() - 1.0);
    let lam_nu = LAM + NU;
    let calvo_lag = (st["eta0"]["Lbar"].as_f64().unwrap() * lam_nu - 1.0).abs().max(st["eta0"]["lin_err"].as_f64().unwrap());
    let kneg = (-(bf["up"]["kmin"].as_f64().unwrap().min(bf["down"]["kmin"].as_f64().unwrap()))).max(0.0);
    let lbar = |k: &str| st[k]["Lbar"].as_f64().unwrap();
    let linerr = ["eta0", "eta1", "eta5", "eta20"].iter().map(|k| st[*k]["lin_err"].as_f64().unwrap()).fold(0.0, f64::max);
    let v: Vec<(&str, String)> = vec![
        ("TipLbarZero", fmt(lbar("eta0"), 3)), ("TipLbarOne", fmt(lbar("eta1"), 3)), ("TipLbarFive", fmt(lbar("eta5"), 3)),
        ("TipLbarTwenty", fmt(lbar("eta20"), 3)), ("TipLinErr", fmt(100.0 * linerr, 1)),
        ("TipC", py_g(GHF_C)), ("TipNu", py_g(GHF_NU)), ("TipZeta", py_g(GHF_ZETA)), ("TipEll", py_g(GHF_ELL)), ("TipDu", py_g(GHF_H)),
        ("TipDownFold", fmt(f(&bf["down"], "fold_gS"), 2)), ("TipUpMthree", fmt(f(&bf["up"], "M3"), 3)),
        ("TipDownFoldM", fmt(f(&bf["down"], "fold_m"), 3)), ("TipDownJumpM", fmt(f(&bf["down"], "jump_m"), 3)),
        ("TipDownMthree", fmt(f(&bf["down"], "M3"), 3)),
        ("TipKminTol", tex_sci(kneg.max(1e-16))),
        ("TipKintRel", tex_sci(f(&bf["up"], "int_rel").max(f(&bf["down"], "int_rel")).max(1e-16))),
        ("TipCalvoErr", tex_sci(f(bf, "calvo_rel_err").max(1e-16))), ("TipCalvoLagErr", tex_sci(calvo_lag.max(1e-16))),
        ("RunLbarB", fmt(f(&sp["run"], "LB"), 4)), ("RunLbarP", fmt(f(&sp["run"], "LP"), 4)),
        ("RunLbarGapPct", fmt(f(&sp["run"], "gapL_pct").abs(), 1)),
        ("RunMthreeB", fmt(m3b, 3)), ("RunMthreeP", fmt(m3p, 3)), ("RunMthreeGapPct", fmt(100.0 * (m3b - m3p).abs() / m3b.abs(), 0)),
        ("RunAmpGapPct", fmt(amp, 0)),
        ("RunMoneB", fmt(f(tp, "M1B_base"), 4)), ("RunMoneFV", fmt(st["eta5"]["M1"].as_f64().unwrap(), 4)),
        ("ValShapeB", fmt(f(s0, "M3B"), 4)), ("ValShapeP", fmt(f(s0, "M3P"), 4)), ("ValShapeCross", fmt(f(s0, "cross"), 4)),
        ("ValShapeRec", fmt(f(s0, "record"), 4)),
        ("MagLagZeroMin", fmt(sp["zero"][0].as_f64().unwrap(), 1)), ("MagLagZeroMax", fmt(sp["zero"][1].as_f64().unwrap(), 1)),
        ("MagLagPosMin", fmt(sp["pos"][0].as_f64().unwrap(), 2)), ("MagLagPosMax", fmt(sp["pos"][1].as_f64().unwrap(), 1)),
        ("MagLagAllMin", fmt(sp["all"][0].as_f64().unwrap(), 2)), ("MagLagAllMax", fmt(sp["all"][1].as_f64().unwrap(), 1)),
        ("GridLagRelErr", fmt(100.0 * f(sp, "relerr"), 1)),
    ];
    let mut lines = vec!["% Generated by inertia (Rust port of code/make_data.py); do not edit.".to_string()];
    for (k, s) in v {
        lines.push(format!("\\def\\{k}{{{s}}}"));
    }
    std::fs::write(p.tex().join("tipping_values.tex"), lines.join("\n") + "\n")?;
    Ok(())
}

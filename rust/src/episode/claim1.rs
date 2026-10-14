//! Numerical check of Theorem 3 (B and P have the same p(t, u), hence the same joint law of
//! (tau, U_tau-), duration law, revision law and stationary cross-section). Port of `run_claim1.py`.
//!  (1) same discretisation: P with hbar^{n+1} = c^{n+1} / p^{n+1} on B's grid (identity to round-off);
//!  (2) cross discretisation: P on a time grid twice finer with hbar interpolated linearly in time from
//!      the coarse B run, and P with hbar from a coarser clock grid (Nb / 2), compared with B.

use super::run::NU6;
use super::*;
use crate::util::{default_l, odd, time_grid};
use serde_json::{json, Map, Value};
use std::time::Instant;

struct Laws {
    dens: Vec<f64>,
    surv: Vec<f64>,
    revcdf: Vec<f64>,
    occcdf: Vec<f64>,
    etau: f64,
    ed2: f64,
    kurt: f64,
    var: f64,
}

fn laws(m: &Model, r: &Stepped) -> Laws {
    let (occ, rev, u) = (&r.occ, &r.rev, &m.u);
    let etau = np_sum(occ);
    let rs = np_sum(rev);
    let d2 = wsum2(u, rev) / rs;
    let d4: f64 = np_sum(&u.iter().zip(rev).map(|(x, w)| x.powi(4) * w).collect::<Vec<_>>()) / rs;
    let cum = |v: &[f64], tot: f64| {
        let mut acc = 0.0;
        v.iter().map(|x| {
            acc += x;
            acc / tot
        }).collect::<Vec<f64>>()
    };
    Laws { dens: r.dens.clone(), surv: r.surv.clone(), revcdf: cum(rev, rs), occcdf: cum(occ, etau), etau, ed2: d2,
           kurt: d4 / (d2 * d2), var: wsum2(u, occ) / etau }
}

fn maxabs_rows(a: &[f64], b: &[f64], ra: &[usize], rb: &[usize]) -> f64 {
    ra.iter().zip(rb).map(|(&i, &j)| (a[i] - b[j]).abs()).fold(0.0, f64::max)
}

#[allow(clippy::too_many_arguments)]
fn compare(a: &Laws, b: &Laws, ca: Option<&[f64]>, cb: Option<&[f64]>, n_u: usize, rows_a: Option<&[usize]>, rows_b: Option<&[usize]>) -> Map<String, Value> {
    let na = rows_a.map(|r| r.len()).unwrap_or(a.dens.len());
    let nb = rows_b.map(|r| r.len()).unwrap_or(b.dens.len());
    let n = na.min(nb);
    let ra: Vec<usize> = match rows_a { Some(r) => r[..n].to_vec(), None => (0..n).collect() };
    let rb: Vec<usize> = match rows_b { Some(r) => r[..n].to_vec(), None => (0..n).collect() };
    let bmax = util::max_of(&b.dens);
    let mut o = Map::new();
    o.insert("dur_density_rel".into(), json!(maxabs_rows(&a.dens, &b.dens, &ra, &rb) / bmax));
    o.insert("survival_abs".into(), json!(maxabs_rows(&a.surv, &b.surv, &ra, &rb)));
    let ks = |x: &[f64], y: &[f64]| x.iter().zip(y).map(|(p, q)| (p - q).abs()).fold(0.0, f64::max);
    o.insert("revision_KS".into(), json!(ks(&a.revcdf, &b.revcdf)));
    o.insert("stationary_KS".into(), json!(ks(&a.occcdf, &b.occcdf)));
    o.insert("Etau_rel".into(), json!((a.etau / b.etau - 1.0).abs()));
    o.insert("Ed2_rel".into(), json!((a.ed2 / b.ed2 - 1.0).abs()));
    o.insert("kurt_rel".into(), json!((a.kurt / b.kurt - 1.0).abs()));
    o.insert("var_rel".into(), json!((a.var / b.var - 1.0).abs()));
    if let (Some(ca), Some(cb)) = (ca, cb) {
        let cbmax = util::max_of(cb);
        let mut mx = 0.0f64;
        for (&i, &j) in ra.iter().zip(&rb) {
            for k in 0..n_u {
                mx = mx.max((ca[i * n_u + k] - cb[j * n_u + k]).abs());
            }
        }
        o.insert("joint_c_rel".into(), json!(mx / cbmax));
    }
    o
}

/// hbar at fine output times by linear interpolation in t of the coarse table (row 0: t = 0, hbar = nu).
fn interp_table(ts_coarse: &[f64], hb_coarse: &[f64], ts_fine: &[f64], nu: f64, n_u: usize) -> Vec<f64> {
    let rows = hb_coarse.len() / n_u;
    let tn = &ts_coarse[..rows + 1];
    let mut tab = vec![nu; n_u];
    tab.extend_from_slice(hb_coarse);
    let mut out = Vec::with_capacity((ts_fine.len() - 1) * n_u);
    for &tf in &ts_fine[1..] {
        let mut k = tn.partition_point(|&t| t <= tf) as i64 - 1;
        k = k.clamp(0, tn.len() as i64 - 2);
        let k = k as usize;
        let w = ((tf - tn[k]) / (tn[k + 1] - tn[k])).clamp(0.0, 1.0);
        for i in 0..n_u {
            out.push((1.0 - w) * tab[k * n_u + i] + w * tab[(k + 1) * n_u + i]);
        }
    }
    out
}

pub fn point(lam: f64, theta: f64, eta: f64, nu: f64) -> Map<String, Value> {
    let (du, nb, dtmax) = (0.02, 200usize, 0.04);
    let t0 = Instant::now();
    let l = default_l(lam);
    let t_end = if lam > 0.0 { 40.0 / nu } else { 55.0 / nu };
    let mb = Model::new(lam, theta, eta, nu, l, odd(2.0 * l / du), nb);
    let n_u = mb.n_u;
    let mut res = Map::new();
    for (k, v) in [("lam", lam), ("theta", theta), ("eta", eta), ("nu", nu), ("du", du), ("dtmax", dtmax)] {
        res.insert(k.into(), json!(v));
    }
    res.insert("Nb".into(), json!(nb));
    struct Run {
        ts: Vec<f64>,
        rb: Stepped,
        hb: Vec<f64>,
        rp: Stepped,
        lb: Laws,
        lp: Laws,
    }
    let mut runs: Vec<(usize, Run)> = vec![];
    for rf in [1usize, 2, 4] {
        let ts = time_grid(t_end, dtmax, 1e-3, 1.1, rf);
        let rb = ts_b(&mb, &ts, 0.0, 1, Store::pc(), 1e-16, &[]);
        let (hb, _) = make_hbar(&mb, &rb.store.p, &rb.store.c, 1e-280);
        let rp = ts_p(&mb, &rb.ts, &hb, 0.0, 1, Store::pc(), None);
        let lb = laws(&mb, &rb);
        let lp = laws(&mb, &rp);
        runs.push((rf, Run { ts: rb.ts.clone(), rb, hb, rp, lb, lp }));
    }
    let get = |rf: usize| &runs.iter().find(|(r, _)| *r == rf).unwrap().1;
    let mut ident = Map::new();
    for (rf, r) in &runs {
        let mut d = compare(&r.lp, &r.lb, Some(&r.rp.store.c), Some(&r.rb.store.c), n_u, None, None);
        let pmax = util::max_of(&r.rb.store.p);
        let jp = r.rp.store.p.iter().zip(&r.rb.store.p).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max) / pmax;
        d.insert("joint_p_rel".into(), json!(jp));
        ident.insert(format!("ref{rf}"), Value::Object(d));
    }
    res.insert("identity".into(), Value::Object(ident));
    let mut cross = Map::new();
    for k in [1usize, 2] {
        let (rc, rfn) = (get(k), get(2 * k));
        let hbi = interp_table(&rc.ts, &rc.hb, &rfn.ts, nu, n_u);
        let rpx = ts_p(&mb, &rfn.ts, &hbi, 0.0, 1, Store { c: true, ..Default::default() }, None);
        let lpx = laws(&mb, &rpx);
        let nco = rc.rb.dens.len().min(rfn.rb.dens.len() / 2);
        let rows_f: Vec<usize> = (0..nco).map(|i| 2 * i + 1).collect();
        let rows_c: Vec<usize> = (0..nco).collect();
        let mut e = Map::new();
        e.insert("P_fine_vs_B_coarse".into(), Value::Object(compare(&lpx, &rc.lb, Some(&rpx.store.c), Some(&rc.rb.store.c), n_u, Some(&rows_f), Some(&rows_c))));
        e.insert("P_fine_vs_B_fine".into(), Value::Object(compare(&lpx, &rfn.lb, Some(&rpx.store.c), Some(&rfn.rb.store.c), n_u, None, None)));
        e.insert("B_fine_vs_B_coarse".into(), Value::Object(compare(&rfn.lb, &rc.lb, Some(&rfn.rb.store.c), Some(&rc.rb.store.c), n_u, Some(&rows_f), Some(&rows_c))));
        cross.insert(format!("hbar_ref{}_P_ref{}", k, 2 * k), Value::Object(e));
    }
    res.insert("cross_time".into(), Value::Object(cross));
    let mh = Model::new(lam, theta, eta, nu, l, odd(2.0 * l / du), nb / 2);
    let r1 = get(1);
    let rbh = ts_b(&mh, &r1.ts, 0.0, 1, Store::pc(), 0.0, &[]);
    let (hbh, _) = make_hbar(&mh, &rbh.store.p, &rbh.store.c, 1e-280);
    let rph = ts_p(&mb, &r1.ts, &hbh, 0.0, 1, Store { c: true, ..Default::default() }, None);
    let lh = laws(&mb, &rph);
    let lbh = laws(&mh, &rbh);
    let mut cb = Map::new();
    cb.insert("P_hbarNbhalf_vs_B_Nb".into(), Value::Object(compare(&lh, &r1.lb, Some(&rph.store.c), Some(&r1.rb.store.c), n_u, None, None)));
    cb.insert("B_Nbhalf_vs_B_Nb".into(), Value::Object(compare(&lbh, &r1.lb, Some(&rbh.store.c), Some(&r1.rb.store.c), n_u, None, None)));
    res.insert("cross_b".into(), Value::Object(cb));
    res.insert("secs".into(), json!(t0.elapsed().as_secs_f64()));
    res
}

pub fn points() -> Vec<(f64, f64, f64, f64)> {
    vec![(1.0, 0.35, 5.0, 1.0), (1.0, 0.7, 0.6, NU6), (0.25, 1.0, 5.0, 1.0), (2.0, 0.35, 5.0, 1.0), (0.0, 0.7, 2.0, 1.0)]
}

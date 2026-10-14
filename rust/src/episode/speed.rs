//! Mean lags of the true economy B and of its Markovian projection P (Section 5, panel C of the
//! magnitudes figure). Port of the reference `checks/speed_grid.py`.
//!
//!   S      = (E[I U_-] + 2 lam E int U I dt) / (sig^2 E tau)       (the renewal form, eq. 14-15)
//!   S Lbar = (E int U I dt + lam E int U J dt) / (sig^2 E tau),  J = int I.
//! The moment fields are q, m1 = E[I; .] and m2 = E[J; .] (weights w1 = u, w2 = 1).

use super::run::NU6;
use super::*;
use crate::util::{default_l, odd, rich_b, rich_t, rich_u, time_grid};
use serde_json::{json, Map, Value};
use std::time::Instant;

#[derive(Clone, Copy, Debug, Default)]
pub struct Moments {
    pub etau: f64,
    pub eiu: f64,
    pub eui: f64,
    pub euj: f64,
}

/// Time-integrated episode solve of B with fields q, E[I; .], E[J; .].
pub fn resolvent_moments(m: &Model, f: f64) -> Moments {
    let (nb, n) = (m.nb, m.n_u);
    let mut q = vec![0.0; nb * n];
    q[m.i0] = 1.0;
    let mut m1 = vec![0.0; nb * n];
    let mut m2 = vec![0.0; nb * n];
    let ones = vec![1.0; n];
    let mut out = SweepOut::new(n);
    let coef = m.ucoef(f);
    m.sweep(&coef, 0.0, 1.0, nb, &m.r, &m.h, &m.w1, &ones, &mut q, Some(&mut m1), Some(&mut m2), 3, &mut out);
    let u = &m.u;
    Moments { etau: np_sum(&out.p), eiu: wsum(u, &out.cm1), eui: wsum(u, &out.pm1), euj: wsum(u, &out.pm2) }
}

/// Backward-Euler episode solve of P with the same three fields.
pub fn p_moments(m: &Model, ts: &[f64], hb: &[f64], f: f64) -> Moments {
    let n = m.n_u;
    let mut q = vec![0.0; n];
    q[m.i0] = 1.0;
    let mut m1 = vec![0.0; n];
    let mut m2 = vec![0.0; n];
    let r0 = vec![0.0; n];
    let ones = vec![1.0; n];
    let mut out = SweepOut::new(n);
    let coef = m.ucoef(f);
    let u = &m.u;
    let mut acc = Moments::default();
    for k in 0..hb.len() / n {
        let dt = ts[k + 1] - ts[k];
        let s = 1.0 / dt;
        m.sweep(&coef, s, s, 1, &r0, &hb[k * n..(k + 1) * n], &m.w1, &ones, &mut q, Some(&mut m1), Some(&mut m2), 3, &mut out);
        acc.etau += dt * np_sum(&out.p);
        acc.eiu += dt * wsum(u, &out.cm1);
        acc.eui += dt * wsum(u, &out.pm1);
        acc.euj += dt * wsum(u, &out.pm2);
    }
    acc
}

/// (S, M1) with M1 = int t k(t) dt = S Lbar.
pub fn s_m1(r: &Moments, lam: f64, sig: f64) -> (f64, f64) {
    let s = (r.eiu + 2.0 * lam * r.eui) / (sig * sig * r.etau);
    let m1 = (r.eui + lam * r.euj) / (sig * sig * r.etau);
    (s, m1)
}

/// One economy of the speed grid.
pub fn point(lam: f64, theta: f64, eta: f64, nu: f64) -> Map<String, Value> {
    let (du, nbs, refs, fine_du, fine_nb) = (0.02, [100usize, 200], [1usize, 2, 4], [0.02, 0.01], [800usize, 1600]);
    let t0 = Instant::now();
    let l = default_l(lam);
    let t_end = if lam > 0.0 { 40.0 / nu } else { 55.0 / nu };
    let mut gaps = Vec::new();
    let mut raw = Map::new();
    for &nb in &nbs {
        let m = Model::new(lam, theta, eta, nu, l, odd(2.0 * l / du), nb);
        let (sb, m1b) = s_m1(&resolvent_moments(&m, 0.0), lam, 1.0);
        let (mut gs, mut gm) = (vec![], vec![]);
        for &rf in &refs {
            let ts = time_grid(t_end, 0.04, 1e-3, 1.1, rf);
            let rb = ts_b(&m, &ts, 0.0, 1, Store::pc(), 1e-17, &[]);
            let (hb, _) = make_hbar(&m, &rb.store.p, &rb.store.c, 1e-280);
            let (sp, m1p) = s_m1(&p_moments(&m, &rb.ts, &hb, 0.0), lam, 1.0);
            gs.push(sb - sp);
            gm.push(m1b - m1p);
        }
        let (_, rs3) = rich_t(&gs, &refs);
        let (rm2, rm3) = rich_t(&gm, &refs);
        let g = (rs3.unwrap(), rm3.unwrap(), (rm3.unwrap() - rm2.unwrap()).abs(), sb, m1b);
        raw.insert(nb.to_string(), json!({"gapS": g.0, "gapM1": g.1, "gapM1_dterr": g.2, "SB": g.3, "M1B": g.4}));
        gaps.push(g);
    }
    let gap_s = rich_b(gaps[0].0, gaps[1].0);
    let gap_m1 = rich_b(gaps[0].1, gaps[1].1);
    let err_m1 = (gap_m1 - gaps[1].1).abs() + gaps[1].2;
    let mut fine = std::collections::HashMap::new();
    for &h in &fine_du {
        for &nb in &fine_nb {
            let m = Model::new(lam, theta, eta, nu, l, odd(2.0 * l / h), nb);
            let (s, m1) = s_m1(&resolvent_moments(&m, 0.0), lam, 1.0);
            fine.insert((h.to_bits(), nb), (m.du, s, m1));
        }
    }
    let f = |h: f64, nb: usize| fine[&(h.to_bits(), nb)];
    let ext = |j: usize| {
        let pick = |t: (f64, f64, f64)| if j == 1 { t.1 } else { t.2 };
        let x1 = rich_b(pick(f(fine_du[0], fine_nb[0])), pick(f(fine_du[0], fine_nb[1])));
        let x2 = rich_b(pick(f(fine_du[1], fine_nb[0])), pick(f(fine_du[1], fine_nb[1])));
        rich_u(x1, f(fine_du[0], fine_nb[1]).0, x2, f(fine_du[1], fine_nb[1]).0)
    };
    let (sb, m1b) = (ext(1), ext(2));
    let (sp, m1p) = (sb - gap_s, m1b - gap_m1);
    let (lb, lp) = (m1b / sb, m1p / sp);
    let mut o = Map::new();
    for (k, v) in [("lam", lam), ("theta", theta), ("eta", eta), ("nu", nu), ("SB", sb), ("SP", sp), ("M1B", m1b), ("M1P", m1p),
                   ("LB", lb), ("LP", lp), ("gapS_pct", 100.0 * (sb - sp) / sb), ("gapL_pct", 100.0 * (lb - lp) / lb),
                   ("gapM1_relerr", if gap_m1 != 0.0 { err_m1 / gap_m1.abs() } else { 0.0 })] {
        o.insert(k.into(), json!(v));
    }
    o.insert("raw".into(), Value::Object(raw));
    o.insert("secs".into(), json!(t0.elapsed().as_secs_f64()));
    o
}

/// The 48 pausing-clock economies of Section 5.
pub fn grid_points() -> Vec<(f64, f64, f64, f64)> {
    let mut v = vec![];
    for lam in [0.0, 0.25, 1.0, 2.0] {
        for theta in [0.35, 0.7, 1.0] {
            for (eta, nu) in [(0.6, NU6), (0.6, 1.0), (2.0, 1.0), (5.0, 1.0)] {
                v.push((lam, theta, eta, nu));
            }
        }
    }
    v
}

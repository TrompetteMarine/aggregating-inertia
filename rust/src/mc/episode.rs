//! Monte Carlo of economy B (true clock A) and economy P (tabulated hbar) with common random numbers.
//! One episode per path from (U, A) = (0, 0) with push f. U advances with the exact OU transition, A
//! and I = int U dt by the trapezoid rule, and the killing is integrated out along each path: with
//! survival weights S(t) = exp(-int_0^t hazard), per path
//!   tau = int S dt, U2 = int S U^2 dt, IU = int h S I U dt, I2 = int h S I^2 dt, I1 = int S U dt,
//! for B (hazard H(U, A)) and P (hazard hbar(t, U), bilinear in (t, u)). Histograms by batch: the
//! duration density, the revision law and the occupation, for B and P.

use super::mean_std;
use crate::episode::{make_hbar, run, ts_b, Model, Store};
use crate::util::{binned_from_cdf, default_l, odd, rng::Xoshiro, time_grid};
use rayon::prelude::*;
use serde_json::{json, Map, Value};
use std::time::Instant;

#[inline]
fn smoothact(u: f64, theta: f64) -> f64 {
    if theta <= 0.0 {
        return 1.0;
    }
    let z = (u.abs() - theta) / theta;
    if z <= 0.0 {
        0.0
    } else if z >= 1.0 {
        1.0
    } else {
        let a = (-1.0 / z).exp();
        let b = (-1.0 / (1.0 - z)).exp();
        a / (a + b)
    }
}

/// The projected hazard on a coarse time grid, extrapolated in dt (refinements 1, 2) and in Nb.
pub struct HbarTable {
    pub m: Model,
    pub tnodes: Vec<f64>,
    pub table: Vec<f64>,
}

pub fn hbar_table(lam: f64, theta: f64, eta: f64, nu: f64) -> HbarTable {
    let (du, nbs, refs, dtmax, dtmin, growth) = (0.02, [200usize, 400], [1usize, 2], 0.04, 1e-3, 1.1);
    let l = default_l(lam);
    let t_end = if lam > 0.0 { 40.0 / nu } else { 55.0 / nu };
    let mut tabs = std::collections::HashMap::new();
    let mut last = None;
    for &nb in &nbs {
        let m = Model::new(lam, theta, eta, nu, l, odd(2.0 * l / du), nb);
        for &rf in &refs {
            let ts = time_grid(t_end, dtmax, dtmin, growth, rf);
            let rb = ts_b(&m, &ts, 0.0, 1, Store::pc(), 1e-15, &[]);
            let (hb, _) = make_hbar(&m, &rb.store.p, &rb.store.c, 1e-280);
            tabs.insert((nb, rf), (rb.ts.clone(), hb));
        }
        last = Some(m);
    }
    let m = last.unwrap();
    let n_u = m.n_u;
    let tc = tabs[&(nbs[0], refs[0])].0.clone();
    let n_c = nbs.iter().map(|&nb| tabs[&(nb, refs[0])].0.len()).min().unwrap() - 1;
    let n_f = nbs.iter().map(|&nb| tabs[&(nb, refs[1])].0.len()).min().unwrap() - 1;
    let n = n_c.min(n_f / refs[1]);
    let coarse = |nb: usize, rf: usize| -> Vec<f64> {
        let hb = &tabs[&(nb, rf)].1;
        let k = rf / refs[0];
        let mut out = Vec::with_capacity(n * n_u);
        for r in 0..n {
            let row = k - 1 + r * k;
            out.extend_from_slice(&hb[row * n_u..(row + 1) * n_u]);
        }
        out
    };
    let h: Vec<Vec<f64>> = nbs
        .iter()
        .map(|&nb| {
            let (c1, c2) = (coarse(nb, refs[0]), coarse(nb, refs[1]));
            c2.iter().zip(&c1).map(|(a, b)| 2.0 * a - b).collect()
        })
        .collect();
    let mut table = vec![nu; n_u];
    for r in 0..n {
        for i in 0..n_u {
            let mut v = 2.0 * h[1][r * n_u + i] - h[0][r * n_u + i];
            v = v.max(m.nu).min(m.hmax[i]);
            if m.chi[i] == 0.0 {
                v = nu;
            }
            table.push(v);
        }
    }
    HbarTable { tnodes: tc[..n + 1].to_vec(), table, m }
}

/// Settings of one Monte Carlo run.
#[derive(Clone, Debug)]
pub struct McOpts {
    pub f: f64,
    pub npaths: usize,
    pub dt: f64,
    pub seed: u64,
    pub wtol: f64,
    pub nbatch: usize,
    pub tbin: f64,
    pub ubin: f64,
    pub ulim: f64,
}

impl Default for McOpts {
    fn default() -> Self {
        McOpts { f: 0.0, npaths: 200_000, dt: 0.002, seed: 12345, wtol: 1e-10, nbatch: 20, tbin: 0.05, ubin: 0.05, ulim: 5.0 }
    }
}

pub struct McRun {
    /// npaths x 10 per-path results: tauB, tauP, U2B, U2P, IUB, IUP, I2B, I2P, I1B, I1P
    pub out: Vec<[f64; 10]>,
    /// nbatch x nb histograms: dur B, dur P, rev B, rev P, occ B, occ P
    pub hists: [Vec<f64>; 6],
    pub nb_t: usize,
    pub nb_u: usize,
    pub secs: f64,
}

pub fn run_mc(tab: &HbarTable, lam: f64, theta: f64, eta: f64, nu: f64, o: &McOpts) -> McRun {
    let m = &tab.m;
    let tn = &tab.tnodes;
    let tmax = *tn.last().unwrap();
    let kmax = (tmax / o.dt) as usize;
    let mut krow = vec![0usize; kmax + 1];
    let mut kw = vec![0.0; kmax + 1];
    for k in 0..=kmax {
        let tk = k as f64 * o.dt;
        let mut r = tn.partition_point(|&x| x <= tk) as i64 - 1;
        r = r.clamp(0, tn.len() as i64 - 2);
        let r = r as usize;
        krow[k] = r;
        kw[k] = ((tk - tn[r]) / (tn[r + 1] - tn[r])).clamp(0.0, 1.0);
    }
    let nb_t = (10.0 / o.tbin).round() as usize;
    let nb_u = (2.0 * o.ulim / o.ubin).round() as usize;
    let (sig, gam, n_u, u0, du) = (m.sig, m.gam, m.n_u, m.u[0], m.du);
    let hb = &tab.table;
    let ea = (-lam * o.dt).exp();
    let sd = if lam > 0.0 { sig * (-(-2.0 * lam * o.dt).exp_m1() / (2.0 * lam)).sqrt() } else { sig * o.dt.sqrt() };
    let mf = if lam > 0.0 { o.f * (-(-lam * o.dt).exp_m1()) / lam } else { o.f * o.dt };
    let hbar_at = |row: usize, w: f64, u: f64| -> f64 {
        let x = (u - u0) / du;
        let mut i = x.floor() as i64;
        let a;
        if i < 0 {
            i = 0;
            a = 0.0;
        } else if i >= n_u as i64 - 1 {
            i = n_u as i64 - 2;
            a = 1.0;
        } else {
            a = x - i as f64;
        }
        let i = i as usize;
        let r0 = &hb[row * n_u..];
        let r1 = &hb[(row + 1) * n_u..];
        let h0 = (1.0 - a) * r0[i] + a * r0[i + 1];
        let h1 = (1.0 - a) * r1[i] + a * r1[i + 1];
        (1.0 - w) * h0 + w * h1
    };
    let t0 = Instant::now();
    let nbatch = o.nbatch;
    // batch bt holds paths n with floor(n nbatch / npaths) = bt
    let ranges: Vec<(usize, usize)> = (0..nbatch)
        .map(|bt| {
            let lo = (bt * o.npaths).div_ceil(nbatch);
            let hi = ((bt + 1) * o.npaths).div_ceil(nbatch);
            (lo, hi)
        })
        .collect();
    let results: Vec<(Vec<[f64; 10]>, [Vec<f64>; 6])> = ranges
        .par_iter()
        .map(|&(lo, hi)| {
            let mut hist: [Vec<f64>; 6] = [vec![0.0; nb_t], vec![0.0; nb_t], vec![0.0; nb_u], vec![0.0; nb_u], vec![0.0; nb_u], vec![0.0; nb_u]];
            let mut outs = Vec::with_capacity(hi - lo);
            for n in lo..hi {
                let mut r = Xoshiro::for_path(o.seed, n as u64);
                let (mut u, mut a, mut ii, mut al) = (0.0, 0.0, 0.0, smoothact(0.0, theta));
                let (mut sb, mut sp) = (1.0, 1.0);
                let mut hbv = nu + eta * al * (-(-gam * a).exp_m1());
                let mut hpv = hbar_at(krow[0], kw[0], u);
                let mut acc = [0.0f64; 10];
                for k in 0..kmax {
                    let un = u * ea + mf + sd * r.gauss();
                    let aln = smoothact(un, theta);
                    let an = a + 0.5 * (al + aln) * o.dt;
                    let inn = ii + 0.5 * (u + un) * o.dt;
                    let hbn = nu + eta * aln * (-(-gam * an).exp_m1());
                    let hpn = hbar_at(krow[k + 1], kw[k + 1], un);
                    let sbn = sb * (-0.5 * (hbv + hbn) * o.dt).exp();
                    let spn = sp * (-0.5 * (hpv + hpn) * o.dt).exp();
                    let h = 0.5 * o.dt;
                    acc[0] += h * (sb + sbn);
                    acc[1] += h * (sp + spn);
                    acc[2] += h * (sb * u * u + sbn * un * un);
                    acc[3] += h * (sp * u * u + spn * un * un);
                    acc[8] += h * (sb * u + sbn * un);
                    acc[9] += h * (sp * u + spn * un);
                    acc[4] += h * (hbv * sb * ii * u + hbn * sbn * inn * un);
                    acc[5] += h * (hpv * sp * ii * u + hpn * spn * inn * un);
                    acc[6] += h * (hbv * sb * ii * ii + hbn * sbn * inn * inn);
                    acc[7] += h * (hpv * sp * ii * ii + hpn * spn * inn * inn);
                    let tnow = (k + 1) as f64 * o.dt;
                    let it = (tnow / o.tbin) as usize;
                    let iu = ((un - (-o.ulim)) / o.ubin).floor();
                    if it < nb_t {
                        hist[0][it] += hbn * sbn * o.dt;
                        hist[1][it] += hpn * spn * o.dt;
                    }
                    if iu >= 0.0 && (iu as usize) < nb_u {
                        let iu = iu as usize;
                        hist[2][iu] += hbn * sbn * o.dt;
                        hist[3][iu] += hpn * spn * o.dt;
                        hist[4][iu] += sbn * o.dt;
                        hist[5][iu] += spn * o.dt;
                    }
                    u = un;
                    a = an;
                    ii = inn;
                    al = aln;
                    hbv = hbn;
                    hpv = hpn;
                    sb = sbn;
                    sp = spn;
                    if sb < o.wtol && sp < o.wtol {
                        break;
                    }
                }
                outs.push(acc);
            }
            (outs, hist)
        })
        .collect();
    let mut out = Vec::with_capacity(o.npaths);
    let mut hists: [Vec<f64>; 6] = [vec![], vec![], vec![], vec![], vec![], vec![]];
    for (outs, h) in results {
        out.extend(outs);
        for k in 0..6 {
            hists[k].extend_from_slice(&h[k]);
        }
    }
    McRun { out, hists, nb_t, nb_u, secs: t0.elapsed().as_secs_f64() }
}

fn col(out: &[[f64; 10]], k: usize) -> Vec<f64> {
    out.iter().map(|r| r[k]).collect()
}

/// S = mean X / mean Y with the delta-method standard error; also the per-path linearisation.
fn ratio_stats(x: &[f64], y: &[f64]) -> (f64, f64, Vec<f64>) {
    let (mx, _) = mean_std(x, 1);
    let (my, _) = mean_std(y, 1);
    let s = mx / my;
    let lin: Vec<f64> = x.iter().zip(y).map(|(a, b)| (a - s * b) / my).collect();
    let (_, sd) = mean_std(&lin, 1);
    (s, sd / (x.len() as f64).sqrt(), lin)
}

pub fn analyse(out: &[[f64; 10]], lam: f64, sig: f64) -> Map<String, Value> {
    let c = |k| col(out, k);
    let (taub, taup, u2b, u2p, iub, iup, i2b, i2p) = (c(0), c(1), c(2), c(3), c(4), c(5), c(6), c(7));
    let n = out.len() as f64;
    let mut r = Map::new();
    let comb = |a: &[f64], b: &[f64], w: f64| a.iter().zip(b).map(|(x, y)| x + w * y).collect::<Vec<f64>>();
    let scale = |a: &[f64]| a.iter().map(|x| sig * sig * x).collect::<Vec<f64>>();
    let (sb, seb, linb) = ratio_stats(&comb(&iub, &i2b, lam), &scale(&taub));
    let (spv, sep, linp) = ratio_stats(&comb(&iup, &i2p, lam), &scale(&taup));
    let d: Vec<f64> = linb.iter().zip(&linp).map(|(a, b)| a - b).collect();
    let (_, sdd) = mean_std(&d, 1);
    for (k, v) in [("S_B", sb), ("se_S_B", seb), ("S_P", spv), ("se_S_P", sep), ("gap", sb - spv), ("se_gap", sdd / n.sqrt())] {
        r.insert(k.into(), json!(v));
    }
    let (sb2, seb2, lb2) = ratio_stats(&comb(&u2b, &i2b, 0.5 * lam), &scale(&taub));
    let (sp2, sep2, lp2) = ratio_stats(&comb(&u2p, &i2p, 0.5 * lam), &scale(&taup));
    let d2: Vec<f64> = lb2.iter().zip(&lp2).map(|(a, b)| a - b).collect();
    let (_, sd2) = mean_std(&d2, 1);
    for (k, v) in [("S_B_split", sb2), ("se_S_B_split", seb2), ("S_P_split", sp2), ("se_S_P_split", sep2),
                   ("gap_split", sb2 - sp2), ("se_gap_split", sd2 / n.sqrt())] {
        r.insert(k.into(), json!(v));
    }
    for (name, x) in [("Etau_B", &taub), ("Etau_P", &taup), ("EI2_B", &i2b), ("EI2_P", &i2p), ("EIU_B", &iub), ("EIU_P", &iup),
                      ("EU2int_B", &u2b), ("EU2int_P", &u2p)] {
        let (m, s) = mean_std(x, 1);
        r.insert(name.into(), json!(m));
        r.insert(format!("se_{name}"), json!(s / n.sqrt()));
    }
    for (name, a, b) in [("EI2_diff", &i2b, &i2p), ("Etau_diff", &taub, &taup), ("EU2int_diff", &u2b, &u2p)] {
        let d: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();
        let (m, s) = mean_std(&d, 1);
        r.insert(name.into(), json!(m));
        r.insert(format!("se_{name}"), json!(s / n.sqrt()));
    }
    r
}

/// PDE laws of B (time-stepped, Nb = 400, refinement 2) binned as the Monte Carlo histograms. Unlike
/// the reference, the mass of every PDE time step is split between the bins it overlaps (the reference
/// assigned whole steps of up to 0.02 to bins of width 0.05, which made the PDE histogram uneven by
/// about 20% from bin to bin).
pub struct PdeRef {
    pub dur: Vec<f64>,
    pub rev: Vec<f64>,
    pub occ: Vec<f64>,
    pub etau: f64,
}

pub fn pde_reference(lam: f64, theta: f64, eta: f64, nu: f64, nb_t: usize, tbin: f64, nb_u: usize, ubin: f64, ulim: f64) -> PdeRef {
    let l = default_l(lam);
    let m = Model::new(lam, theta, eta, nu, l, odd(2.0 * l / 0.02), 400);
    let t_end = if lam > 0.0 { 40.0 / nu } else { 55.0 / nu };
    let ts = time_grid(t_end, 0.04, 1e-3, 1.1, 2);
    let rb = ts_b(&m, &ts, 0.0, 1, Store { c: true, ..Default::default() }, 1e-15, &[]);
    let t = &rb.ts;
    let mut durh = vec![0.0; nb_t];
    for (k, &dens) in rb.dens.iter().enumerate() {
        let (a, b) = (t[k], t[k + 1]);
        let rate = dens; // completions per unit time over the step (backward Euler: value at t_{k+1})
        let mut lo = a;
        while lo < b {
            let bin = (lo / tbin).floor() as usize;
            let edge = ((bin + 1) as f64 * tbin).min(b);
            if bin < nb_t {
                durh[bin] += rate * (edge - lo);
            }
            if edge <= lo {
                break;
            }
            lo = edge;
        }
    }
    let mut faces = vec![m.u[0] - m.du / 2.0];
    faces.extend(m.u.iter().map(|x| x + m.du / 2.0));
    let edges: Vec<f64> = (0..=nb_u).map(|k| -ulim + ubin * k as f64).collect();
    let binned = |mass: &[f64]| binned_from_cdf(mass, &faces, &edges);
    PdeRef { dur: durh, rev: binned(&rb.rev), occ: binned(&rb.occ), etau: rb.etau }
}

/// The Monte Carlo benchmark of the paper (Supplement S1.4): per-path estimates and finite-difference
/// responses with common random numbers, and the laws of B and P.
pub fn benchmark(lam: f64, theta: f64, eta: f64, nu: f64, npaths: usize, dts: &[f64]) -> Map<String, Value> {
    let t0 = Instant::now();
    let tab = hbar_table(lam, theta, eta, nu);
    let fp = run::full_point(lam, theta, eta, nu, &run::FullOpts::default());
    let mut res = Map::new();
    for (k, v) in [("lam", lam), ("theta", theta), ("eta", eta), ("nu", nu)] {
        res.insert(k.into(), json!(v));
    }
    res.insert("npaths".into(), json!(npaths));
    let mut pde = Map::new();
    for k in ["SB_formula", "SB_fd", "SP_formula", "SP_fd", "gap_formula", "gap_fd", "EI2_B", "EI2_P", "Etau", "VarU", "F"] {
        pde.insert(k.into(), fp[k].clone());
    }
    res.insert("pde".into(), Value::Object(pde));
    let mut runs = vec![];
    let mut reference: Option<PdeRef> = None;
    for &dt in dts {
        let o = McOpts { npaths, dt, ..Default::default() };
        let mc = run_mc(&tab, lam, theta, eta, nu, &o);
        let mut a = analyse(&mc.out, lam, 1.0);
        let dlt = 0.05;
        let nfd = (npaths / 2).max(1000);
        let op = run_mc(&tab, lam, theta, eta, nu, &McOpts { f: dlt, npaths: nfd, dt, seed: 777, ..Default::default() });
        let om = run_mc(&tab, lam, theta, eta, nu, &McOpts { f: -dlt, npaths: nfd, dt, seed: 777, ..Default::default() });
        let lin = |o: &[[f64; 10]], ci: usize, ti: usize| -> (f64, Vec<f64>) {
            let (mc_, _) = mean_std(&col(o, ci), 1);
            let (mt, _) = mean_std(&col(o, ti), 1);
            let mm = mc_ / mt;
            (mm, o.iter().map(|r| (r[ci] - mm * r[ti]) / mt).collect())
        };
        let mut fd = Map::new();
        let mut lins = std::collections::HashMap::new();
        for (nm, ci, ti) in [("B", 8usize, 0usize), ("P", 9, 1)] {
            let (mp, lp) = lin(&op.out, ci, ti);
            let (mm, lm) = lin(&om.out, ci, ti);
            let d: Vec<f64> = lp.iter().zip(&lm).map(|(a, b)| a - b).collect();
            let (_, s) = mean_std(&d, 1);
            fd.insert(format!("S_fd_{nm}"), json!((mp - mm) / (2.0 * dlt)));
            fd.insert(format!("se_S_fd_{nm}"), json!(s / (nfd as f64).sqrt() / (2.0 * dlt)));
            lins.insert(nm, d);
        }
        let gd: Vec<f64> = lins["B"].iter().zip(&lins["P"]).map(|(a, b)| a - b).collect();
        let (_, sg) = mean_std(&gd, 1);
        let sfb = fd["S_fd_B"].as_f64().unwrap();
        let sfp = fd["S_fd_P"].as_f64().unwrap();
        fd.insert("gap_fd".into(), json!(sfb - sfp));
        fd.insert("se_gap_fd".into(), json!(sg / (nfd as f64).sqrt() / (2.0 * dlt)));
        fd.insert("delta".into(), json!(dlt));
        a.extend(fd);
        if reference.is_none() {
            reference = Some(pde_reference(lam, theta, eta, nu, mc.nb_t, o.tbin, mc.nb_u, o.ubin, o.ulim));
        }
        let rf = reference.as_ref().unwrap();
        let mut dist = Map::new();
        let nbatch = o.nbatch;
        for (name, kb, kp, nb) in [("dur", 0usize, 1usize, mc.nb_t), ("rev", 2, 3, mc.nb_u), ("occ", 4, 5, mc.nb_u)] {
            let (hb, hp) = (&mc.hists[kb], &mc.hists[kp]);
            let mut b = vec![0.0; nb];
            let mut p = vec![0.0; nb];
            let mut seb = vec![0.0; nb];
            let mut sed = vec![0.0; nb];
            for i in 0..nb {
                let colb: Vec<f64> = (0..nbatch).map(|bt| hb[bt * nb + i]).collect();
                let colp: Vec<f64> = (0..nbatch).map(|bt| hp[bt * nb + i]).collect();
                b[i] = crate::util::np_sum(&colb) / npaths as f64;
                p[i] = crate::util::np_sum(&colp) / npaths as f64;
                let (_, sdb) = mean_std(&colb, 1);
                seb[i] = sdb * nbatch as f64 / npaths as f64 / (nbatch as f64).sqrt();
                let diff: Vec<f64> = colb.iter().zip(&colp).map(|(x, y)| (x - y) * nbatch as f64 / npaths as f64).collect();
                let (_, sdd) = mean_std(&diff, 1);
                sed[i] = sdd / (nbatch as f64).sqrt();
            }
            let pde: Vec<f64> = if name == "occ" {
                let (eb, ep) = (a["Etau_B"].as_f64().unwrap(), a["Etau_P"].as_f64().unwrap());
                for i in 0..nb {
                    b[i] /= eb;
                    p[i] /= ep;
                    seb[i] /= eb;
                    sed[i] /= eb;
                }
                rf.occ.iter().map(|x| x / rf.etau).collect()
            } else if name == "dur" {
                rf.dur.clone()
            } else {
                rf.rev.clone()
            };
            let maxv = |v: Vec<f64>| v.into_iter().fold(0.0f64, f64::max);
            dist.insert(name.into(), json!({
                "max_abs_BminusP": maxv((0..nb).map(|i| (b[i] - p[i]).abs()).collect()),
                "max_z_BminusP": maxv((0..nb).filter(|&i| sed[i] > 0.0).map(|i| (b[i] - p[i]).abs() / sed[i]).collect()),
                "max_abs_MC_B_minus_PDE": maxv((0..nb).map(|i| (b[i] - pde[i]).abs()).collect()),
                "max_z_MC_B_minus_PDE": maxv((0..nb).filter(|&i| seb[i] > 0.0).map(|i| (b[i] - pde[i]).abs() / seb[i]).collect()),
                "max_abs_MC_P_minus_PDE": maxv((0..nb).map(|i| (p[i] - pde[i]).abs()).collect()),
                "peak": maxv(pde.clone()),
            }));
        }
        a.insert("dist".into(), Value::Object(dist));
        a.insert("dt".into(), json!(dt));
        a.insert("secs".into(), json!(mc.secs));
        runs.push(Value::Object(a));
    }
    res.insert("runs".into(), Value::Array(runs));
    res.insert("secs".into(), json!(t0.elapsed().as_secs_f64()));
    res
}

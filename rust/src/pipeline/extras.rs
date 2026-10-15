//! The computations behind the paper's Figures 4 and 6 and the showcase plots of the repository: the joint
//! stationary law on (u, a), the propagation of a common push through the population, sample episodes, and
//! agent-based simulations of a population with social feedback near the tipping point (critical slowing down,
//! and hysteresis under a slow ramp of the social gain), with the numbers the paper quotes about them
//! (tex/extras_values.tex).

use super::{f, figures, log, py_g, py_thousands, read_csv, read_json, write_csv, write_csv_sig, write_json, Paths};
use crate::stationary::{default_grid, Model, Params, ResetKind, Shifted};
use crate::util::{activation, rng::Xoshiro};
use anyhow::{Context, Result};
use rayon::prelude::*;
use serde_json::json;

/// The share of agents with no processing at consensus under full resets, exactly. An episode starts at the
/// reference with no processing, which stays zero until the assessment first leaves the tolerance band or the
/// episode ends; inside the band the hazard is nu. By renewal-reward the share is F (1 - E[exp(-nu T)]) / nu,
/// with T the exit time of the Ornstein-Uhlenbeck process from (-theta, theta) started at zero and F the
/// completion rate; phi(u) = E_u[exp(-nu T)] solves (sigma^2/2) phi'' - lam u phi' - nu phi = 0 on the band,
/// with phi = 1 at its edges, here by central differences on 4001 interior nodes (the centre is a node).
pub fn no_processing_share(prm: &Params, completion_rate: f64) -> f64 {
    let n = 4001usize;
    let h = 2.0 * prm.theta / (n + 1) as f64;
    let s2 = 0.5 * prm.sig * prm.sig / (h * h);
    let (mut a, mut b, mut c, mut d) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let u = -prm.theta + (i + 1) as f64 * h;
        a[i] = s2 + prm.lam * u / (2.0 * h);
        b[i] = -2.0 * s2 - prm.nu;
        c[i] = s2 - prm.lam * u / (2.0 * h);
    }
    d[0] -= a[0];
    d[n - 1] -= c[n - 1];
    let phi = crate::util::linalg::thomas(&a, &b, &c, &d);
    completion_rate * (1.0 - phi[(n - 1) / 2]) / prm.nu
}

/// Joint stationary density on (u, a) and its u-marginal, for full and capped resets. The solver's first
/// processing cell, [0, 0.003), holds the agents with no processing together with a layer of agents with very
/// little: just outside the band processing advances very slowly, because the activation rises from zero with
/// all its derivatives. The joint density is written from the second cell on, the first cell separately.
pub fn stationary_density(p: &Paths) -> Result<()> {
    for (tag, reset) in [("full", ResetKind::Zero), ("capped", ResetKind::Pl)] {
        let prm = Params::new(figures::THETA, figures::ETA, figures::NU, figures::LAM, figures::GAM, reset);
        let m = Model::new(prm, default_grid(&prm, 0.02, 160));
        let (d, n, na) = (&m.d, m.d.n_u, m.d.na);
        let (mut uu, mut aa, mut dens) = (vec![], vec![], vec![]);
        for j in 0..na - 1 {
            let amid = 0.5 * (d.af[j] + d.af[j + 1]);
            if amid > 6.0 {
                break;
            }
            for i in 0..n {
                if d.uc[i].abs() > 3.0 {
                    continue;
                }
                uu.push(d.uc[i]);
                aa.push(amid);
                dens.push(m.pi[j * n + i] / (d.du[i] * d.da[j]));
            }
        }
        write_csv_sig(&p.extras().join(format!("stationary_joint_{tag}.csv")), "u,a,density", &[&uu, &aa, &dens], 9)?;
        let first: Vec<f64> = (0..n).map(|i| m.pi[i] / d.du[i]).collect();
        write_csv_sig(&p.extras().join(format!("stationary_first_cell_{tag}.csv")), "u,density", &[&d.uc, &first], 9)?;
        // shares of processing: in the first cell, below 0.1 (linear within the cell that contains it), and,
        // under full resets, exactly zero
        let cell: Vec<f64> = (0..na).map(|j| m.pi[j * n..(j + 1) * n].iter().sum()).collect();
        let below = |x: f64| -> f64 {
            let mut acc = 0.0;
            for j in 0..na {
                if d.af[j + 1] <= x {
                    acc += cell[j];
                } else {
                    acc += cell[j] * ((x - d.af[j]) / (d.af[j + 1] - d.af[j])).clamp(0.0, 1.0);
                    break;
                }
            }
            acc
        };
        let rate: f64 = (0..na * n).map(|k| d.h[k] * m.pi[k]).sum();
        let none = if matches!(reset, ResetKind::Zero) { json!(no_processing_share(&prm, rate)) } else { json!(null) };
        write_json(&p.extras().join(format!("stationary_meta_{tag}.json")),
                   &json!({"first_cell_share": cell[0], "first_cell_width": d.da[0], "share_below_0.1": below(0.1),
                           "no_processing_share": none, "completion_rate": rate, "theta": figures::THETA, "reset": reset.code()}))?;
        let marg: Vec<f64> = (0..n).map(|i| (0..na).map(|j| m.pi[j * n + i]).sum::<f64>() / d.du[i]).collect();
        let hz: Vec<f64> = (0..n).map(|i| (0..na).map(|j| d.h[j * n + i] * m.pi[j * n + i]).sum::<f64>() / d.du[i]).collect();
        write_csv_sig(&p.extras().join(format!("stationary_marginal_{tag}.csv")), "u,density,completion_rate", &[&d.uc, &marg, &hz], 9)?;
        log(&format!("extras: stationary density ({tag})"));
    }
    Ok(())
}

/// The u-marginal of e^{Lt} b over time: how a small common push propagates through the population.
pub fn response_field(p: &Paths) -> Result<()> {
    let prm = Params::new(figures::THETA, figures::ETA, figures::NU, figures::LAM, figures::GAM, ResetKind::Zero);
    let m = Model::new(prm, default_grid(&prm, 0.02, 160));
    let sec = &m.odd;
    let (n, na) = (sec.n, sec.na);
    let dt = 0.01;
    let z = 2.0 / dt;
    let sh = Shifted::<f64>::new(sec, z, false);
    let mut x = m.b.clone();
    let (mut tt, mut uu, mut vv) = (vec![], vec![], vec![]);
    let record = |x: &[f64], t: f64, tt: &mut Vec<f64>, uu: &mut Vec<f64>, vv: &mut Vec<f64>| {
        for i in 0..n {
            if sec.uc[i] > 3.0 {
                break;
            }
            let s: f64 = (0..na).map(|j| x[j * n + i]).sum::<f64>() / m.d.du[m.d.n_u / 2 + i];
            // odd sector: the density change is s at u and -s at -u
            for (u, v) in [(sec.uc[i], s), (-sec.uc[i], -s)] {
                tt.push(t);
                uu.push(u);
                vv.push(v);
            }
        }
    };
    record(&x, 0.0, &mut tt, &mut uu, &mut vv);
    let mut t = 0.0;
    for _ in 0..4 {
        let y: Vec<f64> = x.iter().map(|v| -z * v).collect();
        x = sh.solve(sec, &y);
        t += dt / 2.0;
        record(&x, t, &mut tt, &mut uu, &mut vv);
    }
    let mut step = 0;
    while t < 3.0 - 1e-9 {
        let lx = sec.matvec(&x);
        let y: Vec<f64> = x.iter().zip(&lx).map(|(a, b)| -(z * a + b)).collect();
        x = sh.solve(sec, &y);
        t += dt;
        step += 1;
        if t < 0.5 || step % 4 == 0 {
            record(&x, t, &mut tt, &mut uu, &mut vv);
        }
    }
    write_csv(&p.extras().join("response_field.csv"), "t,u,density_change", &[&tt, &uu, &vv], 8)?;
    log("extras: response field");
    Ok(())
}

/// A few episodes of the running economy (exact OU steps, dt = 0.002): assessment, processing,
/// hazard, and the completions.
pub fn sample_paths(p: &Paths) -> Result<()> {
    let (lam, nu, eta, theta, gam) = (figures::LAM, figures::NU, figures::ETA, figures::THETA, figures::GAM);
    let (dt, t_end, k) = (0.002, 14.0, 6usize);
    let ea = (-lam * dt).exp();
    let sd = (-(-2.0 * lam * dt).exp_m1() / (2.0 * lam)).sqrt();
    let (mut pid, mut ts, mut us, mut as_, mut hs) = (vec![], vec![], vec![], vec![], vec![]);
    let (mut epid, mut et, mut eu) = (vec![], vec![], vec![]);
    for path in 0..k {
        let mut r = Xoshiro::for_path(2026, path as u64);
        let (mut u, mut a) = (0.0, 0.0);
        let nsteps = (t_end / dt) as usize;
        for s in 0..=nsteps {
            let t = s as f64 * dt;
            let h = nu + eta * activation(u, theta) * (-(-gam * a).exp_m1());
            if s % 5 == 0 {
                pid.push(path as f64);
                ts.push(t);
                us.push(u);
                as_.push(a);
                hs.push(h);
            }
            if r.unif() < -(-h * dt).exp_m1() {
                epid.push(path as f64);
                et.push(t);
                eu.push(u);
                u = 0.0;
                a = 0.0;
            } else {
                let un = u * ea + sd * r.gauss();
                a += 0.5 * (activation(u, theta) + activation(un, theta)) * dt;
                u = un;
            }
        }
    }
    write_csv(&p.extras().join("sample_paths.csv"), "path,t,u,a,hazard", &[&pid, &ts, &us, &as_, &hs], 6)?;
    write_csv(&p.extras().join("sample_events.csv"), "path,t,u_before", &[&epid, &et, &eu], 6)?;
    log("extras: sample paths");
    Ok(())
}

/// Agent-based population with social feedback f_t = g c tanh(m_t / c) and a Markov hazard H(u).
struct Pop {
    lam: f64,
    rule: fn(f64) -> f64,
}

fn down_rule(u: f64) -> f64 {
    figures::ghf_rule("down")(u)
}
fn up_rule(u: f64) -> f64 {
    figures::ghf_rule("up")(u)
}

impl Pop {
    /// Simulates n agents from a population started at the stationary law of the reference (by a
    /// burn-in without feedback) and returns the average assessment on the recording grid.
    #[allow(clippy::too_many_arguments)]
    fn run(&self, n: usize, g_of_t: &(dyn Fn(f64) -> f64 + Sync), field: f64, t_end: f64, dt: f64, rec: f64, m0: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
        let c = figures::GHF_C;
        let ea = (-self.lam * dt).exp();
        let sd = (-(-2.0 * self.lam * dt).exp_m1() / (2.0 * self.lam)).sqrt();
        let drift = (-(-self.lam * dt).exp_m1()) / self.lam;
        let nchunk = 64usize;
        let size = n.div_ceil(nchunk);
        let mut rngs: Vec<Xoshiro> = (0..nchunk).map(|k| Xoshiro::for_path(seed, k as u64)).collect();
        let mut us: Vec<Vec<f64>> = (0..nchunk).map(|k| vec![0.0; size.min(n.saturating_sub(k * size))]).collect();
        // burn-in at f = 0, then a small common shift m0
        us.par_iter_mut().zip(rngs.par_iter_mut()).for_each(|(u, r)| {
            for _ in 0..(20.0 / dt) as usize {
                for x in u.iter_mut() {
                    if r.unif() < -(-(self.rule)(*x) * dt).exp_m1() {
                        *x = 0.0;
                    } else {
                        *x = *x * ea + sd * r.gauss();
                    }
                }
            }
            for x in u.iter_mut() {
                *x += m0;
            }
        });
        let mut m = us.iter().map(|u| u.iter().sum::<f64>()).sum::<f64>() / n as f64;
        let (mut tt, mut mm) = (vec![0.0], vec![m]);
        let every = (rec / dt).round() as usize;
        let nsteps = (t_end / dt).round() as usize;
        for s in 0..nsteps {
            let t = s as f64 * dt;
            let f = g_of_t(t) * c * (m / c).tanh() + field;
            let sums: Vec<f64> = us
                .par_iter_mut()
                .zip(rngs.par_iter_mut())
                .map(|(u, r)| {
                    let mut acc = 0.0;
                    for x in u.iter_mut() {
                        if r.unif() < -(-(self.rule)(*x) * dt).exp_m1() {
                            *x = 0.0;
                        } else {
                            *x = *x * ea + f * drift + sd * r.gauss();
                        }
                        acc += *x;
                    }
                    acc
                })
                .collect();
            m = sums.iter().sum::<f64>() / n as f64;
            if (s + 1) % every == 0 {
                tt.push((s + 1) as f64 * dt);
                mm.push(m);
            }
        }
        (tt, mm)
    }
}

/// Design of the population simulations: time step, burn-in without feedback, the gains held fixed after
/// a small common displacement, and the slow ramp of the gain up and back down.
pub const POP_DT: f64 = 0.01;
pub const POP_FIXED_GAINS: [f64; 4] = [0.8, 0.95, 1.05, 1.2];
pub const POP_FIXED_T: f64 = 60.0;
pub const POP_SHIFT: f64 = 0.05;
pub const POP_RAMP: (f64, f64, f64) = (0.6, 1.3, 300.0);

/// Critical slowing down and tipping in an agent-based population with the two Markov hazards of the
/// paper's bifurcation diagrams, at fixed gains and under a slow ramp of the gain up and down (hysteresis).
pub fn tipping_simulations(p: &Paths, n: usize) -> Result<()> {
    let s_down = crate::ghf::Ghf::new(down_rule, 1.0, 1.0, figures::GHF_L, figures::GHF_H).s_tilted(0.0);
    let s_up = crate::ghf::Ghf::new(up_rule, 1.0, 1.0, figures::GHF_L, figures::GHF_H).s_tilted(0.0);
    let dt = POP_DT;
    let mut out = serde_json::Map::new();
    for (name, rule, s) in [("down", down_rule as fn(f64) -> f64, s_down), ("up", up_rule as fn(f64) -> f64, s_up)] {
        let pop = Pop { lam: 1.0, rule };
        let mut cols: Vec<Vec<f64>> = vec![];
        let mut heads = vec!["t".to_string()];
        for (k, gs) in POP_FIXED_GAINS.iter().enumerate() {
            let g = gs / s;
            let (tt, mm) = pop.run(n, &|_| g, 0.0, POP_FIXED_T, dt, 0.1, POP_SHIFT, 100 + k as u64);
            if cols.is_empty() {
                cols.push(tt);
            }
            cols.push(mm);
            heads.push(format!("gS{gs}"));
        }
        let refs: Vec<&[f64]> = cols.iter().map(|c| c.as_slice()).collect();
        write_csv(&p.extras().join(format!("population_{name}.csv")), &heads.join(","), &refs, 6)?;
        // ramp gS from 0.6 to 1.3 and back over 2 x 300 time units. The model is symmetric under
        // u -> -u, so which branch the population selects is a fair coin; seeds are tried in order and
        // the first run that selects the upper branch is kept (runs that select the lower branch are
        // its mirror images in law, so the selection does not change the law of |m|).
        let ramp = move |t: f64| {
            let (lo, hi, half) = POP_RAMP;
            let x = if t < half { lo + (hi - lo) * t / half } else { hi - (hi - lo) * (t - half) / half };
            x / s
        };
        let mut chosen = None;
        for (tries, seed) in (300u64..320).enumerate() {
            let (tt, mm) = pop.run(n, &ramp, 0.0, 2.0 * POP_RAMP.2, dt, 0.5, 0.0, seed);
            let top = tt.iter().position(|&t| t >= POP_RAMP.2).unwrap_or(tt.len() - 1);
            if mm[top] > 0.0 {
                chosen = Some((tt, mm, seed, tries + 1));
                break;
            }
        }
        let (tt, mm, seed, tries) = chosen.expect("no run selected the upper branch in 20 seeds");
        let gsr: Vec<f64> = tt.iter().map(|&t| ramp(t) * s).collect();
        write_csv(&p.extras().join(format!("hysteresis_{name}.csv")), "t,gS,m", &[&tt, &gsr, &mm], 6)?;
        out.insert(name.into(), json!({"S": s, "agents": n, "dt": dt, "ramp_seed": seed, "ramp_runs": tries}));
        log(&format!("extras: population simulations ({name})"));
    }
    write_json(&p.extras().join("population_meta.json"), &serde_json::Value::Object(out))?;
    Ok(())
}

/// Column `name` of a CSV file.
fn column(path: &std::path::Path, name: &str) -> Result<Vec<f64>> {
    let (head, cols) = read_csv(path)?;
    let k = head.iter().position(|h| h == name).with_context(|| format!("no column {name} in {}", path.display()))?;
    Ok(cols[k].clone())
}

/// The stable stationary state with feedback m > 0 at the given gS, interpolated on the branch of
/// stationary states (data/tip_branch_{name}.csv, written by the `figures` stage); on the falling-hazard
/// branch only the part beyond the fold is stable.
fn stable_level(p: &Paths, name: &str, m_floor: f64, gs: f64) -> Result<f64> {
    let path = p.data().join(format!("tip_branch_{name}.csv"));
    let (g, m, sl) = (column(&path, "gS")?, column(&path, "m")?, column(&path, "slope")?);
    let mut pts: Vec<(f64, f64)> = (0..g.len()).filter(|&i| sl[i] < 1.0 && m[i] > m_floor).map(|i| (g[i], m[i])).collect();
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let (xs, ys): (Vec<f64>, Vec<f64>) = pts.into_iter().unzip();
    anyhow::ensure!(gs >= xs[0] && gs <= xs[xs.len() - 1], "gS = {gs} outside the stable branch of {name}");
    Ok(crate::util::interp::interp(gs, &xs, &ys))
}

/// The numbers the paper quotes about these computations, as TeX macros in tex/extras_values.tex: the
/// design of the population simulations, where the simulated population tips and drops back under the slow
/// ramp, how long it takes to tip at fixed gains, how close the settled averages come to the stationary
/// states, and the share of agents without processing in the running economy. Needs the `figures` stage.
pub fn values(p: &Paths) -> Result<()> {
    let rec = read_json(&p.data().join("records.json")).context("run `inertia figures` before `inertia extras`")?;
    let bif = &rec["bifurcation"]["down"];
    let (jump_m, fold_m) = (f(bif, "jump_m"), f(bif, "fold_m"));
    let meta = read_json(&p.extras().join("population_meta.json"))?;
    let agents = meta["down"]["agents"].as_u64().unwrap_or(0);
    anyhow::ensure!(meta["up"]["agents"].as_u64() == Some(agents), "the two populations differ in size");
    let mut v: Vec<(String, String)> = vec![];
    let mut push = |k: &str, s: String| v.push((k.to_string(), s));
    push("PopAgents", py_thousands(agents).replace(',', "{,}"));
    push("PopDt", py_g(POP_DT));
    push("PopRampLo", py_g(POP_RAMP.0));
    push("PopRampHi", py_g(POP_RAMP.1));
    push("PopRampT", py_g(POP_RAMP.2));
    push("PopShift", py_g(POP_SHIFT));
    push("PopFixedT", py_g(POP_FIXED_T));
    // under the slow ramp, the falling-hazard population tips when its average passes half-way to the
    // distant branch (at gS = 1) on the way up, and drops back when it passes half-way to consensus from
    // the fold on the way down
    let hp = p.extras().join("hysteresis_down.csv");
    let (t, g, m) = (column(&hp, "t")?, column(&hp, "gS")?, column(&hp, "m")?);
    let half = POP_RAMP.2;
    let up = (0..t.len()).find(|&i| t[i] <= half && m[i] > 0.5 * jump_m).context("the ramp never tipped")?;
    let dn = (0..t.len()).find(|&i| t[i] > half && m[i] < 0.5 * fold_m).context("the ramp never dropped back")?;
    push("PopTipUp", format!("{:.2}", g[up]));
    push("PopDropDown", format!("{:.2}", g[dn]));
    // at fixed gains: time to pass half-way to the stable tilted state, and the settled averages
    let mut gap: f64 = 0.0;
    for (name, floor) in [("down", fold_m), ("up", 0.0)] {
        let pp = p.extras().join(format!("population_{name}.csv"));
        let t = column(&pp, "t")?;
        for (gs, tag) in [(1.05, "A"), (1.2, "B")] {
            let level = stable_level(p, name, floor, gs)?;
            let m = column(&pp, &format!("gS{gs}"))?;
            let settled: Vec<f64> = (0..t.len()).filter(|&i| t[i] >= POP_FIXED_T - 20.0).map(|i| m[i]).collect();
            let mean = settled.iter().sum::<f64>() / settled.len() as f64;
            gap = gap.max((mean - level).abs() / level);
            let cap = if name == "down" { "Down" } else { "Up" };
            push(&format!("Pop{cap}Level{tag}"), format!("{level:.2}"));
            push(&format!("Pop{cap}Settled{tag}"), format!("{mean:.2}"));
            if name == "down" {
                let i = (0..t.len()).find(|&i| m[i] > 0.5 * level).context("no tip at a fixed gain")?;
                push(&format!("PopTipTime{tag}"), format!("{:.0}", t[i]));
            }
        }
    }
    push("PopLevelGap", format!("{:.0}", (100.0 * gap).ceil()));
    let st = read_json(&p.extras().join("stationary_meta_full.json"))?;
    push("HidNoneShare", format!("{:.0}", 100.0 * f(&st, "no_processing_share")));
    push("HidFirstCell", format!("{:.3}", f(&st, "first_cell_width")));
    let mut text = String::from("% Generated by inertia extras (agent-based simulations and the running economy's stationary law); do not edit.\n");
    for (k, s) in &v {
        text.push_str(&format!("\\def\\{k}{{{s}}}\n"));
    }
    std::fs::write(p.tex().join("extras_values.tex"), text)?;
    log("extras: values");
    Ok(())
}

pub fn run_all(p: &Paths, n_agents: usize) -> Result<()> {
    stationary_density(p)?;
    response_field(p)?;
    sample_paths(p)?;
    tipping_simulations(p, n_agents)?;
    values(p)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The share of agents with no processing in the running economy, against an independent solution of the
    /// same boundary-value problem (scipy's banded solver on 16,000 nodes gives 0.40882819) with the episode
    /// solver's completion rate F = 1.0553847428665462.
    #[test]
    fn no_processing_share_running_economy() {
        let prm = Params::new(figures::THETA, figures::ETA, figures::NU, figures::LAM, figures::GAM, ResetKind::Zero);
        let share = no_processing_share(&prm, 1.0553847428665462);
        assert!((share - 0.40882819).abs() < 1e-6, "{share}");
    }
}

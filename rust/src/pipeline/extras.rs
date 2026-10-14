//! Additional data for the showcase plots of the repository (not used by the paper's numbers):
//! the joint stationary law on (u, a), the propagation of a common push through the population,
//! sample episodes, and agent-based simulations of a population with social feedback near the
//! tipping point (critical slowing down, and hysteresis under a slow ramp of the social gain).

use super::{figures, log, write_csv, write_csv_sig, write_json, Paths};
use crate::stationary::{default_grid, Model, Params, ResetKind, Shifted};
use crate::util::{activation, rng::Xoshiro};
use anyhow::Result;
use rayon::prelude::*;
use serde_json::json;

/// Joint stationary density on (u, a) and its u-marginal, for full and capped resets.
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
        // the first processing cell holds the agents that have not left the tolerance band since their
        // last revision (processing exactly zero): an atom at a = 0, reported separately
        let atom: Vec<f64> = (0..n).map(|i| m.pi[i] / d.du[i]).collect();
        write_csv_sig(&p.extras().join(format!("stationary_atom_{tag}.csv")), "u,density", &[&d.uc, &atom], 9)?;
        let share: f64 = m.pi[..n].iter().sum();
        write_json(&p.extras().join(format!("stationary_meta_{tag}.json")),
                   &json!({"atom_share": share, "theta": figures::THETA, "first_cell_width": d.da[0], "reset": reset.code()}))?;
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

/// Critical slowing down and tipping in an agent-based population (falling hazard, Figure 4D's
/// "tip" economy), and a slow ramp of the gain up and down (hysteresis).
pub fn tipping_simulations(p: &Paths, n: usize) -> Result<()> {
    let s_down = crate::ghf::Ghf::new(down_rule, 1.0, 1.0, figures::GHF_L, figures::GHF_H).s_tilted(0.0);
    let s_up = crate::ghf::Ghf::new(up_rule, 1.0, 1.0, figures::GHF_L, figures::GHF_H).s_tilted(0.0);
    let (dt, t_end) = (0.01, 60.0);
    let mut out = serde_json::Map::new();
    for (name, rule, s) in [("down", down_rule as fn(f64) -> f64, s_down), ("up", up_rule as fn(f64) -> f64, s_up)] {
        let pop = Pop { lam: 1.0, rule };
        let mut cols: Vec<Vec<f64>> = vec![];
        let mut heads = vec!["t".to_string()];
        for (k, gs) in [0.8, 0.95, 1.05, 1.2].iter().enumerate() {
            let g = gs / s;
            let (tt, mm) = pop.run(n, &|_| g, 0.0, t_end, dt, 0.1, 0.05, 100 + k as u64);
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
        // the first run that selects the upper branch is kept (equivalently, a run reflected to it).
        let ramp = move |t: f64| {
            let (lo, hi, half) = (0.6, 1.3, 300.0);
            let x = if t < half { lo + (hi - lo) * t / half } else { hi - (hi - lo) * (t - half) / half };
            x / s
        };
        let mut chosen = None;
        for (tries, seed) in (300u64..320).enumerate() {
            let (tt, mm) = pop.run(n, &ramp, 0.0, 600.0, dt, 0.5, 0.0, seed);
            let top = tt.iter().position(|&t| t >= 300.0).unwrap_or(tt.len() - 1);
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

pub fn run_all(p: &Paths, n_agents: usize) -> Result<()> {
    stationary_density(p)?;
    response_field(p)?;
    sample_paths(p)?;
    tipping_simulations(p, n_agents)?;
    Ok(())
}

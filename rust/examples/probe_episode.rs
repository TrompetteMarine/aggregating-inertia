//! Prints episode-solver quantities for a few cases as JSON (compared with the reference by
//! tests_ref/compare_episode.py).
use inertia::episode::{self, run, Clock, Model, Store};
use inertia::util::{odd, time_grid};
use serde_json::json;

fn main() {
    let mut out = vec![];
    for (clock, lam, theta, eta, nu, nb, du) in [
        (Clock::Pausing, 1.0, 0.7, 5.0, 1.0, 50usize, 0.05),
        (Clock::Pausing, 0.0, 0.35, 0.6, run::NU6, 40, 0.05),
        (Clock::Dist, 1.0, 0.0, 5.0, 1.0, 40, 0.05),
        (Clock::Sq, 2.0, 0.0, 2.0, 1.0, 40, 0.05),
        (Clock::DistThr(1.0), 1.0, 0.0, 50.0, 1.0, 40, 0.05),
    ] {
        let l = inertia::util::default_l(lam);
        let m = Model::with_clock(clock, lam, theta, eta, nu, l, odd(2.0 * l / du), nb, 1.0, 1.0);
        let r = episode::resolvent_b(&m, 0.0, 3);
        let s = episode::stats_from(&m, &r.source(3));
        let (sfd, _, _) = episode::sb_fd(&m, 0.002);
        let ts = time_grid(if lam > 0.0 { 40.0 / nu } else { 55.0 / nu }, 0.04, 1e-3, 1.1, 1);
        let rb = episode::ts_b(&m, &ts, 0.0, 3, Store::pc(), 1e-17, &[]);
        let (hb, nbad) = episode::make_hbar(&m, &rb.store.p, &rb.store.c, 1e-280);
        let rp = episode::ts_p(&m, &rb.ts, &hb, 0.0, 3, Store::none(), Some((&rb.store.p, &rb.store.c)));
        let sp = episode::stats_from(&m, &rp.source(3));
        let (spfd, _, _) = episode::sp_fd(&m, &rb.ts, &hb, 0.002);
        let sbts = episode::stats_from(&m, &rb.source(3));
        out.push(json!({
            "clock": clock.label(), "lam": lam, "theta": theta, "eta": eta, "nu": nu, "Nb": nb, "du": du,
            "Etau": r.etau, "EI": r.ei, "EU2int": r.eu2int, "EIU": r.eiu, "EI2": r.ei2, "EI2_alt": r.ei2_alt,
            "S_formula": s.s_formula, "Kurt": s.kurt, "VarU": s.var_u, "SB_fd": sfd,
            "nsteps": rb.nsteps, "Etau_ts": rb.etau, "S_formula_ts": sbts.s_formula, "nbad": nbad,
            "SP_formula": sp.s_formula, "SP_fd": spfd, "EI2_P": rp.ei2, "maxdiff_p": rp.maxdiff_p,
            "hb_sum": hb.iter().sum::<f64>(), "hb_mid": hb[(hb.len() / m.n_u / 2) * m.n_u + m.i0 + 20],
        }));
    }
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}

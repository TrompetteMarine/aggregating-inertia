//! Prints stationary-solver quantities for a few cases as JSON (compared with the reference by
//! tests_ref/compare_stationary.py).
use inertia::stationary::{analysis, default_grid, Model, Params, ResetKind};
use num_complex::Complex64;
use serde_json::json;

fn main() {
    let cases = [
        (0.7, 5.0, 1.0, 1.0, ResetKind::Zero, 0.05, 40usize),
        (0.7, 0.6, 0.7136347625525, 1.0, ResetKind::Pl, 0.05, 40),
        (1.0, 2.0, 0.2, 0.25, ResetKind::S, 0.05, 40),
        (0.0, 0.6, 0.7136347625525, 1.0, ResetKind::Zero, 0.05, 40),
        (0.7, 0.0, 1.0, 1.0, ResetKind::Pl, 0.05, 40),
    ];
    let mut out = vec![];
    for (theta, eta, nu, lam, reset, h, na) in cases {
        let p = Params::new(theta, eta, nu, lam, 1.0, reset);
        let g = default_grid(&p, h, na);
        let m = Model::new(p, g);
        let st = m.stats();
        let s = m.s_resolvent();
        let sfd = m.s_fd(1e-3);
        let k1 = m.khat_real(0.7);
        let k2 = m.khat(Complex64::new(0.0, 1.3));
        let gap = analysis::slowest_rate(&m);
        let (ks, t, k) = analysis::kernel_stats(&m, Some(12.0), 0.01, 4.0, 0.05, None);
        let fc = analysis::first_crossing(&m, 60.0, 90);
        let cert = analysis::certify_count(&m, 0.98 / s, &ks, None, 2.0, 20000);
        out.push(json!({
            "theta": theta, "eta": eta, "nu": nu, "lam": lam, "reset": reset.code(), "h": h, "Na": na,
            "N_odd": m.odd.nn, "nT_odd": m.odd.nt(), "nT_even": m.even.nt(),
            "var": st.var, "F": st.f, "kurt6F": st.kurt_over_6f, "share_nonzero": st.share_nonzero, "k0": st.k0,
            "lump": st.lump_mass, "S": s, "S_fd": sfd, "khat07": k1, "khat13i_re": k2.re, "khat13i_im": k2.im,
            "gap": gap, "kmin": ks.kmin, "tmin": ks.tmin, "int_k": ks.int_k, "TV": ks.tv, "int_t_absk": ks.int_t_absk,
            "k_at_1": k[t.iter().position(|&x| (x - 1.0).abs() < 1e-9).unwrap()],
            "fc_type": fc.kind, "g_first": fc.g, "nevals": fc.nevals, "ncross": fc.crossings.len(),
            "min_re_khat": fc.min_re_khat, "cert_N": cert.as_ref().map(|c| c.n).unwrap_or(f64::NAN),
            "cert_minF": cert.as_ref().map(|c| c.min_abs_f).unwrap_or(f64::NAN),
        }));
    }
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}

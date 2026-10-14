use inertia::mc::episode::{hbar_table, run_mc, McOpts};
use serde_json::json;
fn main() {
    let (lam, theta, eta, nu) = (1.0, 0.7, 0.6, 0.7136347625525);
    let tab = hbar_table(lam, theta, eta, nu);
    let o = McOpts { npaths: 2000, ..Default::default() };
    let r = run_mc(&tab, lam, theta, eta, nu, &o);
    let tsum: f64 = tab.table.iter().sum();
    let first: Vec<Vec<f64>> = r.out.iter().take(5).map(|a| a.to_vec()).collect();
    let means: Vec<f64> = (0..10).map(|k| r.out.iter().map(|a| a[k]).sum::<f64>() / r.out.len() as f64).collect();
    let hs: Vec<f64> = r.hists.iter().map(|h| h.iter().sum::<f64>()).collect();
    println!("{}", json!({"rows": tab.table.len() / tab.m.n_u, "tend": tab.tnodes.last(), "tsum": tsum, "first": first, "means": means, "hsums": hs, "secs": r.secs}));
}

use inertia::ghf::Ghf;
use serde_json::json;
fn main() {
    let (nu, z, l) = (0.5, 3.0, 0.7);
    let up = move |u: f64| nu + z * (1.0 - (-u * u / (2.0 * l * l)).exp());
    let down = move |u: f64| nu + z * (-u * u / (2.0 * l * l)).exp();
    let ts: Vec<f64> = (0..241).map(|i| i as f64 * 0.05).collect();
    let mut out = vec![];
    for (name, f) in [("up", Box::new(up) as Box<dyn Fn(f64) -> f64>), ("down", Box::new(down))] {
        let m = Ghf::new(&f, 1.0, 1.0, 12.0, 0.02);
        let mk = Ghf::new(&f, 1.0, 1.0, 12.0, 0.04);
        let k0 = mk.kernel(0.0, &ts);
        let k1 = mk.kernel(0.8, &ts);
        out.push(json!({"name": name, "M01": m.m(0.1), "M2": m.m(2.0), "S0": m.s_tilted(0.0), "S1": m.s_tilted(1.0),
                        "k0_1": k0[20], "k0_6": k0[120], "k0_12": k0[240], "k1_1": k1[20], "k1_12": k1[240],
                        "kmin0": k0.iter().cloned().fold(f64::INFINITY, f64::min)}));
    }
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}

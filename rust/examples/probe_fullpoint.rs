use inertia::episode::run;
fn main() {
    let t0 = std::time::Instant::now();
    let o = run::full_point(1.0, 0.7, 5.0, 1.0, &run::FullOpts::default());
    let keys = ["SB_formula", "SB_fd", "SP_formula", "SP_fd", "gap_fd", "gap_fd_err_b", "gap_fd_err_t", "Etau", "F", "VarU", "Kurt", "ALO", "EI2_B", "EI2_P", "claim1_maxdiff_p"];
    let mut m = serde_json::Map::new();
    for k in keys { m.insert(k.into(), o[k].clone()); }
    m.insert("secs".into(), serde_json::json!(t0.elapsed().as_secs_f64()));
    println!("{}", serde_json::Value::Object(m));
}

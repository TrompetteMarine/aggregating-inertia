"""Compare the Rust Monte Carlo (examples/probe_mc.rs) with the reference C code (mc.c via run_mc.py).

    python3 tests_ref/compare_mc.py reference/code/solvers rust
"""
import sys, os, json, subprocess, tempfile
import numpy as np

SOLVERS = os.path.abspath(sys.argv[1])
RESEARCH = os.path.abspath(os.path.join(SOLVERS, "..", "..", "research", "episode"))
tmp = tempfile.mkdtemp()
# run_mc.py loads libmc.so from its own folder: run it from a folder holding links to both
os.symlink(os.path.join(RESEARCH, "run_mc.py"), os.path.join(tmp, "run_mc.py"))
os.symlink(os.path.join(SOLVERS, "libmc.so"), os.path.join(tmp, "libmc.so"))
sys.path[:0] = [tmp, SOLVERS]
import run_mc as R

rust = json.loads(subprocess.run(["cargo", "run", "--release", "-q", "--example", "probe_mc"], capture_output=True, text=True,
                                 check=True, cwd=sys.argv[2]).stdout)
lam, theta, eta, nu = 1.0, 0.7, 0.6, 0.7136347625525
m, tnodes, table, hraw = R.hbar_table(lam, theta, eta, nu)
out, hists, secs, info = R.run_mc(m, tnodes, table, lam, theta, eta, nu, npaths=2000, dt=0.002)
rel = lambda a, b: np.max(np.abs(np.asarray(a) - np.asarray(b)) / np.maximum(np.abs(np.asarray(b)), 1e-300))
checks = {
    "projected-hazard table": rel(rust["tsum"], table.sum()),
    "first five paths": rel(rust["first"], out[:5]),
    "means over 2000 paths": rel(rust["means"], out.mean(0)),
    "histogram masses": rel(rust["hsums"], [h.sum() for h in hists]),
}
for k, v in checks.items():
    print(f"{k:24s} largest relative difference {v:.1e}")
print(f"rows of the table: reference {table.shape[0]}, Rust {rust['rows']}")
sys.exit(0 if max(checks.values()) < 1e-10 and table.shape[0] == rust["rows"] else 1)

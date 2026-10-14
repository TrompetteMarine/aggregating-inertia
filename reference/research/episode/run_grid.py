"""Task (e)/(f): the parameter grid.  Usage:  python3 run_grid.py LAM [LAM ...]
Writes results/grid_lam{LAM}.json (full detail, incremental) for each lambda."""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c9run import full_point, NU6

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

THETAS = (0.35, 0.7, 1.0)
ETANU = ((0.6, 1.0), (2.0, 1.0), (5.0, 1.0), (0.6, NU6))


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items() if not str(k).startswith("_")}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    return x


def main(lams):
    for lam in lams:
        fn = os.path.join(OUT, f"grid_lam{lam}.json")
        done = json.load(open(fn)) if os.path.exists(fn) else []
        have = {(d["theta"], d["eta"], d["nu"]) for d in done}
        for theta in THETAS:
            for (eta, nu) in ETANU:
                if (theta, eta, nu) in have:
                    continue
                t0 = time.time()
                o = full_point(lam, theta, eta, nu)
                done.append(clean(o))
                json.dump(done, open(fn, "w"), indent=1)
                print(f"lam={lam} theta={theta} eta={eta} nu={nu:.4f}: S_B={o['SB_fd']:.6f}/{o['SB_formula']:.6f} "
                      f"S_P={o['SP_fd']:.6f}/{o['SP_formula']:.6f} gap={o['gap_fd']:+.6f} "
                      f"rel={o['gap_fd']/o['SB_fd']:+.4%} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main([float(a) for a in sys.argv[1:]])

"""Grid refinement of the stationary solver at the running economy (Example 1).

Writes data/validation/running_refinement.json: S on a sequence of grids and the Richardson
extrapolation in the number of processing cells (first order), to compare with the episode solver.
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "solvers"))
from fvsolver import Params, Model, default_grid

p = Params(theta=0.7, eta=5.0, nu=1.0, lam=1.0, gam=1.0, reset='0')
rows = []
for h, na in ((0.02, 160), (0.01, 160), (0.02, 320), (0.01, 320), (0.005, 320), (0.005, 640)):
    m = Model(p, default_grid(p, h=h, Na_f=na))
    rows.append(dict(h=h, Na_f=na, S=float(m.S_resolvent())))
fine = {(r["h"], r["Na_f"]): r["S"] for r in rows}
extrap = 2 * fine[(0.005, 640)] - fine[(0.005, 320)]
out = dict(rows=rows, S_extrap_Na=extrap, note="first-order Richardson in 1/Na_f at h = 0.005")
(ROOT / "data" / "validation").mkdir(parents=True, exist_ok=True)
json.dump(out, open(ROOT / "data" / "validation" / "running_refinement.json", "w"), indent=1)
print(json.dumps(out, indent=1))

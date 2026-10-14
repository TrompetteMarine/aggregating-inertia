"""Tasks (g) and (h): grid refinement.  For each point:
  (A) resolvent of B on du in {0.04, 0.02, 0.01, 0.005} x Nb in {200, ..., 3200}: S_B (fd, formula), Var U/sig^2,
      F, Kurt/(6F); Richardson in Nb (first order) and du (second order).
  (B) time-stepped B -> hbar -> P on (du, Nb) in {(0.04,100),(0.04,200),(0.02,100),(0.02,200),(0.02,400),(0.01,200)}
      with dt refinement 1,2,4: S_P (fd, formula) and the gap, raw and extrapolated.
Usage: python3 run_refine.py LAM THETA ETA NU [quick]  -> results/refine_<tag>.json"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c9run import resolvent_point, run_point, rich_b, rich_u, NU6

HERE = os.path.dirname(os.path.abspath(__file__))
KEYS = ("SB_formula", "SB_fd", "VarU", "F", "ALO", "Kurt", "Etau", "EI2_B")


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


if __name__ == "__main__":
    lam, theta, eta, nu = [float(a) for a in sys.argv[1:5]]
    quick = len(sys.argv) > 5 and sys.argv[5] == "quick"
    tag = f"lam{lam}_th{theta}_eta{eta}_nu{nu:.4f}"
    t0 = time.time()
    out = dict(lam=lam, theta=theta, eta=eta, nu=nu)
    dus = (0.04, 0.02, 0.01) if quick else (0.04, 0.02, 0.01, 0.005)
    Nbs = (200, 400, 800, 1600) if quick else (200, 400, 800, 1600, 3200)
    A = {}
    for du in dus:
        for Nb in Nbs:
            A[(du, Nb)] = resolvent_point(lam, theta, eta, nu, du, Nb)
    out["resolvent"] = [dict(du_target=k[0], **v) for k, v in A.items()]
    # Richardson tables
    ext = {}
    for key in KEYS:
        rows = {}
        for du in dus:
            seq = [A[(du, Nb)][key] for Nb in Nbs]
            rb = [rich_b(seq[i], seq[i + 1]) for i in range(len(seq) - 1)]
            rows[du] = dict(raw=seq, rich_b=rb, du=A[(du, Nbs[-1])]["du"])
        # du-Richardson of the finest b-extrapolated values
        fin = [(rows[du]["rich_b"][-1], rows[du]["du"]) for du in dus]
        ru = [rich_u(fin[i][0], fin[i][1], fin[i + 1][0], fin[i + 1][1]) for i in range(len(fin) - 1)]
        best = ru[-1]
        err = abs(ru[-1] - ru[-2]) + abs(rows[dus[-1]]["rich_b"][-1] - rows[dus[-1]]["rich_b"][-2])
        ext[key] = dict(by_du=rows, rich_u=ru, best=best, err=err)
    out["resolvent_extrap"] = ext
    # (B) time-stepped P
    combos = [(0.04, 100), (0.04, 200), (0.02, 100), (0.02, 200), (0.02, 400), (0.01, 200)]
    if quick:
        combos = [(0.04, 100), (0.04, 200), (0.02, 100), (0.02, 200)]
    B = []
    for du, Nb in combos:
        r = run_point(lam, theta, eta, nu, du=du, Nb=Nb, refs=(1, 2, 4))
        B.append(clean(r))
        print(f"du={r['du']:.4f} Nb={Nb}: S_B fd={r['SB_fd']:.7f} form={r['SB_formula']:.7f} | S_P fd={r['SP_fd']:.7f} "
              f"form={r['SP_formula']:.7f} (R2 {r['SP_formula_R2']:.7f}, raw {r['SP_formula_raw']:.7f}) | "
              f"gap fd={r['gap_fd']:+.7f} form={r['gap_formula']:+.7f} ({r['secs']:.0f}s)", flush=True)
    out["timestepped"] = B
    for key in KEYS:
        e = ext[key]
        print(f"{key:12s} best={e['best']:.7f} err~{e['err']:.1e}   du-Richardson seq {['%.7f' % x for x in e['rich_u']]}")
    out["secs"] = time.time() - t0
    json.dump(clean(out), open(os.path.join(HERE, "results", f"refine_{tag}.json"), "w"), indent=1)
    print(f"done {time.time()-t0:.0f}s")

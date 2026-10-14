"""Scan of the first-crossing type over (theta, eta, nu, lambda, reset), plus an extreme phase.
Coarse grid (h = 0.05, Na_f = 80) for the scan; selected points are re-run on refined grids elsewhere.
For each point: stationary statistics, S, the Nyquist curve khat(i w) on w >= 0 (first crossing:
real at g = 1/S unless khat(i w) crosses the positive real axis beyond S), and the time kernel k(t)
(its minimum and negative mass). Output: scan/scan_<phase>.jsonl (one JSON object per line)."""
import sys, os, json, time, itertools
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from fvsolver import Params, Model, default_grid
from analysis import first_crossing, kernel_stats

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scan")

def point(theta, eta, nu, lam, reset, gam=1.0, h=0.05, Na=80, do_kernel=True):
    p = Params(theta=theta, eta=eta, nu=nu, lam=lam, gam=gam, reset=reset)
    g = default_grid(p, h=h, Na_f=Na)
    t0 = time.time()
    m = Model(p, g)
    st = m.stats(); S = m.S_resolvent()
    fc = first_crossing(m)
    rec = dict(theta=theta, eta=eta, nu=nu, lam=lam, gam=gam, reset=reset, c=p.c, h=h, Na_f=Na,
               Umax=g.Umax, N_odd=m.odd.N, S=S, invS=1/S, var=st['var'], F=st['F'],
               kurt6F=st['kurt_over_6F'], share_nonzero=st['share_nonzero'], lump_mass=st['lump_mass'],
               edge_mass=st['edge_mass'], fc_type=fc['type'], g_first=fc['g'], omega_first=fc['omega'],
               crossings=[(float(w), float(x)) for (w, x) in fc['crossings']],
               min_re_khat=fc['min_re_khat'], max_abs_khat_wpos=fc['max_abs_khat_w_pos'], nevals=fc['nevals'])
    if do_kernel:
        ks, _ = kernel_stats(m)
        rec.update(kmin=ks['kmin'], tmin=ks['tmin'], neg_mass=ks['neg_mass'], first_neg_t=ks['first_neg_t'],
                   int_k=ks['int_k'], T_kernel=ks['T'])
    rec['seconds'] = time.time() - t0
    return rec

def run(phase, points):
    os.makedirs(OUT, exist_ok=True)
    fn = os.path.join(OUT, f"scan_{phase}.jsonl")
    done = set()
    if os.path.exists(fn):
        for line in open(fn):
            r = json.loads(line); done.add((r['theta'], r['eta'], r['nu'], r['lam'], r['reset'], r['gam']))
    with open(fn, "a") as fh:
        for (theta, eta, nu, lam, reset, gam) in points:
            if (theta, eta, nu, lam, reset, gam) in done:
                continue
            try:
                r = point(theta, eta, nu, lam, reset, gam)
            except Exception as e:
                r = dict(theta=theta, eta=eta, nu=nu, lam=lam, reset=reset, gam=gam, error=repr(e))
            fh.write(json.dumps(r) + "\n"); fh.flush()
            if 'error' in r:
                print(f"ERR {theta} {eta} {nu} {lam} {reset} {gam}: {r['error']}", flush=True)
            else:
                print(f"th={theta:<4} eta={eta:<5} nu={nu:<4} lam={lam:<4} R={reset} gam={gam}: S={r['S']:.5f} "
                      f"first={r['fc_type']} g*={r['g_first']:.4f} w={r['omega_first']:.3f} "
                      f"kmin={r.get('kmin', float('nan')):+.2e}@{r.get('tmin', float('nan')):.2f} "
                      f"negmass={r.get('neg_mass', float('nan')):+.2e} ({r['seconds']:.0f}s)", flush=True)

if __name__ == "__main__":
    phase = sys.argv[1]
    if phase == "main":
        pts = []
        for lam, nu, eta in itertools.product((1.0, 0.25), (1.0, 0.2), (0.0, 0.6, 2.0, 5.0, 10.0, 20.0)):
            pts.append((0.0, eta, nu, lam, '0', 1.0))
        for theta, lam, nu, eta, R in itertools.product((0.35, 0.7, 1.0, 1.5), (1.0, 0.25), (1.0, 0.2),
                                                        (0.0, 0.6, 2.0, 5.0, 10.0, 20.0), ('pl', 's', '0')):
            pts.append((theta, eta, nu, lam, R, 1.0))
        run("main", pts)
    elif phase == "extreme":
        pts = []
        for theta, eta, nu, gam, R in itertools.product((0.7, 1.5, 3.0), (50.0, 100.0), (0.2, 0.05),
                                                        (1.0, 10.0), ('pl', '0')):
            pts.append((theta, eta, nu, 1.0, R, gam))
        run("extreme", pts)

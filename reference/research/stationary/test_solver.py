"""Consistency tests of the marching + Woodbury solver on small grids."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spl, time
from fvsolver import *

def check(p, g):
    d = Disc(p, g)
    full = Sector(d, None)
    L = full.L
    cs = np.abs(np.asarray(L.sum(0)).ravel()).max()
    print(f"  full N={full.N} nT={full.nT}  max|column sum|={cs:.2e}")
    # shifted solves vs spsolve in each sector
    rng = np.random.default_rng(0)
    for sec, name in ((full, 'full'), (Sector(d, +1), 'even'), (Sector(d, -1), 'odd')):
        for z in (0.7, 0.3 + 2.1j):
            if name != 'odd' and z == 0.0:
                continue
            y = rng.standard_normal(sec.N) + (1j * rng.standard_normal(sec.N) if np.iscomplexobj(z) else 0)
            sh = Shifted(sec, z)
            x = sh.solve(y)
            A = (sec.L - z * sp.identity(sec.N)).tocsc()
            xr = spl.spsolve(A, y)
            print(f"  {name:4s} z={z}: rel err vs spsolve {np.linalg.norm(x - xr) / np.linalg.norm(xr):.2e}, "
                  f"residual {np.linalg.norm(A @ x - y) / np.linalg.norm(y):.2e}")
    # odd sector at z=0
    odd = Sector(d, -1)
    y = rng.standard_normal(odd.N)
    x = Shifted(odd, 0.0).solve(y)
    print(f"  odd z=0 residual {np.linalg.norm(odd.L @ x - y) / np.linalg.norm(y):.2e}")
    # stationary: even sector vs full grid
    m = Model(p, g)
    pif, info = stationary(full)
    print(f"  stationary: |pi_even - pi_full|_1 = {np.abs(m.pi - pif).sum():.2e}, "
          f"residual |L pi| = {np.abs(L @ pif.ravel()).max():.2e}, P colsum err {info['P_colsum_err']:.1e}")
    st = m.stats()
    S2 = m.S_resolvent(); S1 = m.S_fd(1e-3)
    print(f"  S_fd={S1:.10f}  S_res={S2:.10f}  diff={S1 - S2:.2e}  <u,b>={st['k0']:.6f} var={st['var']:.6f} F={st['F']:.6f}")

for p in (Params(theta=0.7, eta=0.6, nu=0.7136347625525, reset='pl'),
          Params(theta=0.7, eta=2.0, nu=0.3, reset='s'),
          Params(theta=0.0, eta=0.6, nu=0.7136347625525, reset='0'),
          Params(theta=0.7, eta=0.6, nu=0.7136347625525, reset='0'),
          Params(theta=0.35, eta=0.0, nu=1.0, reset='pl', lam=0.25)):
    print(p)
    g = default_grid(p, h=0.2, Na_f=12)
    t0 = time.time()
    check(p, g)
    print(f"  ({time.time() - t0:.1f}s)")

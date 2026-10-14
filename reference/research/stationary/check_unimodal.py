"""Slice unimodality of the stationary law: for every processing slice a, is pi(., a) nonincreasing in |u|?
Together with sign preservation (u E_{(u,a)}[U_t] >= 0), this implies k >= 0 (see the theory note)."""
import sys, itertools, json; sys.path.insert(0, '.')
import numpy as np
from fvsolver import Params, Model, default_grid
out = []; nbad = 0
for th, eta, nu, lam, R in itertools.product((0.35, 0.7, 1.0, 1.5), (0.6, 5.0, 20.0), (1.0, 0.2), (1.0, 0.25), ('pl', 's', '0')):
    p = Params(theta=th, eta=eta, nu=nu, lam=lam, reset=R)
    m = Model(p, default_grid(p, h=0.04, Na_f=80))
    u = m.d.uc; pos = np.flatnonzero(u > 0); pi = m.pi; tot = pi.sum()
    worst = 0.0; badmass = 0.0
    for j in range(pi.shape[0]):
        row = pi[j, pos]
        if row.sum() < 1e-13 * tot: continue
        inc = np.clip(np.diff(row), 0, None).max() / row.max()
        if inc > 1e-10:
            badmass += 2 * row.sum() / tot
        worst = max(worst, inc)
    out.append(dict(theta=th, eta=eta, nu=nu, lam=lam, reset=R, worst_rel_increase=worst, mass_in_nonmonotone_slices=badmass))
    if worst > 1e-10:
        nbad += 1
        print('NON-MONOTONE', out[-1], flush=True)
json.dump(out, open('scan/unimodality.json', 'w'), indent=1)
print(f"{len(out)} points; slices non-monotone in |u| at {nbad} points; max relative increase {max(o['worst_rel_increase'] for o in out):.2e}")

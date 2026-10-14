"""Task (b): validation with grid refinement.

For every case: grid matrix h in HS (u spacing; c and theta fall on cell faces) x Na_f in NAS
(finite a-cells, exponentially stretched).  Reported: raw values on the finest grids, the change
between successive grids, and Richardson extrapolations (first order in a, second order in u).
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import json, time, sys
import numpy as np
from fvsolver import Params, Model, default_grid, renewal_reference
from albl_copy import matched_nu

NU06 = 0.7136347625525
NU2 = matched_nu(2.0)
HS = (0.025, 0.0125, 0.00625)
NAS = (200, 400, 800, 1600)
QTY = ('S_res', 'S_fd', 'var', 'F', 'kurt_over_6F')

CASES = [
    dict(name='V1 theta=0 R0 eta=0 nu=1', p=Params(theta=0.0, eta=0.0, nu=1.0, reset='0'),
         ref=dict(S=0.5, var=1/3, F=1.0, kurt_over_6F=0.6), ref_src='exact'),
    dict(name='V2 theta=0 R0 eta=0.6', p=Params(theta=0.0, eta=0.6, nu=NU06, reset='0'),
         ref=renewal_reference(0.6, NU06, 1.0), ref_src='renewal quadrature (task: S=0.47728)'),
    dict(name='V3 theta=0 R0 eta=2', p=Params(theta=0.0, eta=2.0, nu=NU2, reset='0'),
         ref=renewal_reference(2.0, NU2, 1.0), ref_src='renewal quadrature (task: S=0.43733)'),
    dict(name='V4 eta=0 theta=0.7 Rpl c=0.35 lam=1', p=Params(theta=0.7, eta=0.0, nu=1.0, reset='pl', lam=1.0),
         ref=dict(S=0.64119, var=0.36157, kurt_over_6F=1.18671, F=1.0), ref_src='independent 1D solver'),
    dict(name='V5 eta=0 theta=0.7 Rpl c=0.35 lam=0', p=Params(theta=0.7, eta=0.0, nu=1.0, reset='pl', lam=0.0),
         ref=dict(S=1.35647, var=1.09546, kurt_over_6F=1.49498, F=1.0), ref_src='independent 1D solver / closed form'),
    dict(name='V6 eta=0.6 theta=0.7 Rpl c=0.35 lam=1', p=Params(theta=0.7, eta=0.6, nu=NU06, reset='pl'),
         ref=dict(S=0.677, var=0.379, F=0.728), ref_src='Monte Carlo, Euler dt=0.01 (S biased low ~0.3%)'),
    dict(name='BENCH R0 theta=0.7 eta=0.6 lam=1', p=Params(theta=0.7, eta=0.6, nu=NU06, reset='0'),
         ref=None, ref_src='shared benchmark (no reference)'),
]


def run_case(case, hs=HS, nas=NAS):
    p = case['p']
    rows = []
    for h in hs:
        for Na_f in (nas if p.eta > 0 else (0,)):
            g = default_grid(p, h=h, Na_f=max(Na_f, 1))
            if p.eta == 0:
                g.Na_f = 0
            t0 = time.time()
            m = Model(p, g)
            st = m.stats()
            r = dict(h=h, Na_f=Na_f, Nu=g.Nu, Umax=g.Umax, Amax=g.Amax, a_s=g.a_s,
                     S_res=m.S_resolvent(), S_fd=m.S_fd(1e-3), **{k: st[k] for k in
                     ('var', 'F', 'kurt_over_6F', 'mean', 'k0', 'lump_mass', 'edge_mass', 'P_colsum_err')})
            r['secs'] = time.time() - t0
            rows.append(r)
            print(f"   h={h:<8} Na_f={Na_f:5d} S_res={r['S_res']:.8f} S_fd={r['S_fd']:.8f} var={r['var']:.8f} "
                  f"F={r['F']:.8f} K/6F={r['kurt_over_6F']:.8f} <u,b>={r['k0']:.6f} ({r['secs']:.1f}s)", flush=True)
    return rows


def extrapolate(rows, q, eta):
    """Richardson: first order in a (finest two a-grids at each h), then second order in u."""
    hs = sorted({r['h'] for r in rows}, reverse=True)
    get = lambda h, n: next(r[q] for r in rows if r['h'] == h and r['Na_f'] == n)
    out = {}
    if eta > 0:
        nas = sorted({r['Na_f'] for r in rows})
        Sa, Sa_prev = {}, {}
        for h in hs:
            Sa[h] = 2 * get(h, nas[-1]) - get(h, nas[-2])
            Sa_prev[h] = 2 * get(h, nas[-2]) - get(h, nas[-3])
        hf = hs[-1]
        out['raw_finest'] = get(hf, nas[-1])
        out['d_a_last'] = get(hf, nas[-1]) - get(hf, nas[-2])        # change on last a-doubling
        out['d_u_last'] = get(hf, nas[-1]) - get(hs[-2], nas[-1])     # change on last u-halving
        out['a_extrap_err'] = abs(Sa[hf] - Sa_prev[hf]) / 3           # O(da^2) remainder estimate
    else:
        Sa = {h: get(h, 0) for h in hs}
        hf = hs[-1]
        out['raw_finest'] = Sa[hf]
        out['d_a_last'] = 0.0
        out['d_u_last'] = Sa[hf] - Sa[hs[-2]]
        out['a_extrap_err'] = 0.0
    Su = (4 * Sa[hs[-1]] - Sa[hs[-2]]) / 3
    Su_prev = (4 * Sa[hs[-2]] - Sa[hs[-3]]) / 3
    out['extrap'] = Su
    out['u_extrap_err'] = abs(Su - Su_prev)
    out['err_est'] = out['a_extrap_err'] + max(out['u_extrap_err'], abs(Sa[hs[-1]] - Sa[hs[-2]]) / 3)
    out['raw_err_est'] = abs(Su - out['raw_finest'])
    return out


if __name__ == '__main__':
    which = sys.argv[1:] or [c['name'].split()[0] for c in CASES]
    results = []
    for case in CASES:
        if case['name'].split()[0] not in which:
            continue
        print(case['name'], case['p'], flush=True)
        rows = run_case(case)
        ex = {q: extrapolate(rows, q, case['p'].eta) for q in QTY}
        for q in QTY:
            e = ex[q]
            refkey = 'S' if q.startswith('S_') else q
            refv = case['ref'].get(refkey) if case['ref'] else None
            print(f"   {q:13s} finest={e['raw_finest']:.7f}  d(a)={e['d_a_last']:+.1e} d(u)={e['d_u_last']:+.1e}  "
                  f"extrap={e['extrap']:.7f} +- {e['err_est']:.1e}" + (f"  ref={refv}" if refv is not None else ''))
        results.append(dict(name=case['name'], params=case['p'].__dict__, ref=case['ref'], ref_src=case['ref_src'],
                            rows=rows, extrap=ex))
        tag = '_'.join(which) if sys.argv[1:] else 'all'
        json.dump(results, open(f'out/validation_{tag}.json', 'w'), indent=1, default=float)

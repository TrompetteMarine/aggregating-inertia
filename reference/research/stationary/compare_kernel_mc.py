"""Compare the finite-volume k(t) with the coupled Monte Carlo k(t) (out/mc_kernel.json)."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import json, numpy as np
from fvsolver import Params, Model, default_grid
from analysis import kernel_stats

mc = json.load(open('out/mc_kernel.json'))
res = {}
for name, d in mc.items():
    p = Params(**d['params'])
    m = Model(p, default_grid(p, h=0.0125, Na_f=800))
    ks, (t, k) = kernel_stats(m, T=40.0, dt1=0.005, dt2=0.025)
    tm, km, se = map(np.array, (d['t'], d['k'], d['se']))
    kfv = np.interp(tm, t, k)
    z = (km[1:] - kfv[1:]) / se[1:]
    res[name] = dict(max_abs_diff=float(np.max(np.abs(km - kfv))), max_abs_z=float(np.max(np.abs(z))),
                     rms_z=float(np.sqrt(np.mean(z**2))), fv_kmin=ks['kmin'], fv_tmin=ks['tmin'],
                     fv_int_k=ks['int_k'], S=m.S_resolvent(), mc_int_k=float(np.trapezoid(km, tm)),
                     t=tm.tolist(), k_fv=kfv.tolist(), k_mc=km.tolist(), se=se.tolist())
    print(f"{name}: max|k_mc - k_fv| = {res[name]['max_abs_diff']:.4f}, max|z| = {res[name]['max_abs_z']:.2f}, "
          f"rms z = {res[name]['rms_z']:.2f}; FV kmin = {ks['kmin']:+.2e} at t = {ks['tmin']:.2f}; "
          f"int_0^5 k: MC {res[name]['mc_int_k']:.4f} vs FV S {res[name]['S']:.4f}")
    for i in (4, 10, 20, 40, 60, 100):
        print(f"   t={tm[i]:.2f}  FV {kfv[i]:+.5f}  MC {km[i]:+.5f} +- {se[i]:.5f}")
json.dump(res, open('out/kernel_mc_comparison.json', 'w'), indent=1)

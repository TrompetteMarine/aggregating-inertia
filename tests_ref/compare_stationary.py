"""Compare the Rust stationary solver (examples/probe_stationary.rs) with the reference fvsolver."""
import sys, json, subprocess
import numpy as np
sys.path.insert(0, sys.argv[1])
from fvsolver import Params, Model, default_grid, Shifted
from analysis import kernel_stats, slowest_rate, first_crossing, certify_count

rust = json.loads(subprocess.run(["cargo", "run", "--release", "-q", "--example", "probe_stationary"], capture_output=True,
                                 text=True, check=True, cwd=sys.argv[2]).stdout)
worst = 0.0
for r in rust:
    p = Params(theta=r["theta"], eta=r["eta"], nu=r["nu"], lam=r["lam"], gam=1.0, reset=r["reset"])
    m = Model(p, default_grid(p, h=r["h"], Na_f=r["Na"]))
    st = m.stats(); S = m.S_resolvent(); Sfd = m.S_fd(1e-3)
    k1 = m.khat(0.7); k2 = complex(m.khat(1.3j))
    gap = slowest_rate(m)
    ks, (t, k) = kernel_stats(m, T=12.0)
    fc = first_crossing(m)
    try:
        c = certify_count(m, 0.98 / S, ks); cN, cF = c["N"], c["min_absF"]
    except Exception as e:
        cN, cF = float("nan"), float("nan")
    ref = dict(N_odd=m.odd.N, nT_odd=m.odd.nT, nT_even=m.even.nT, var=st["var"], F=st["F"], kurt6F=st["kurt_over_6F"],
               share_nonzero=st["share_nonzero"], k0=st["k0"], S=S, S_fd=Sfd, khat07=float(k1), khat13i_re=k2.real,
               khat13i_im=k2.imag, gap=gap, kmin=ks["kmin"], tmin=ks["tmin"], int_k=ks["int_k"], TV=ks["TV"],
               int_t_absk=ks["int_t_absk"], k_at_1=float(k[np.argmin(np.abs(t - 1.0))]), g_first=fc["g"], nevals=fc["nevals"],
               ncross=len(fc["crossings"]), min_re_khat=fc["min_re_khat"], cert_N=cN, cert_minF=cF)
    print(f"theta={r['theta']} eta={r['eta']} nu={r['nu']:.4f} lam={r['lam']} reset={r['reset']}  type ref={fc['type']} rust={r['fc_type']}")
    for kk, v in ref.items():
        rv = r[kk]
        if rv is None: rv = float("nan")
        rel = abs(rv - v) / max(abs(v), 1e-300)
        ok = abs(rv - v) <= 1e-9 * max(abs(rv), abs(v)) + 1e-12
        tag = "" if ok else "   <-- differs"
        if not ok:
            worst = max(worst, rel)
        print(f"   {kk:14s} ref={v:+.12e} rust={rv:+.12e} rel={rel:.1e}{tag}")
print("WORST relative difference beyond tolerance:", worst)
sys.exit(0 if worst == 0.0 else 1)

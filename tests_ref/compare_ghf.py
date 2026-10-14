import sys, json, subprocess
import numpy as np
sys.path.insert(0, sys.argv[1])
from ghf1d import GHF
rust = json.loads(subprocess.run(["cargo", "run", "--release", "-q", "--example", "probe_ghf"], capture_output=True, text=True, check=True, cwd=sys.argv[2]).stdout)
nu, z, l = 0.5, 3.0, 0.7
rules = {"up": lambda u: nu + z * (1 - np.exp(-u ** 2 / (2 * l ** 2))), "down": lambda u: nu + z * np.exp(-u ** 2 / (2 * l ** 2))}
ts = np.linspace(0.0, 12.0, 241)
bad = 0
for r in rust:
    Lam = rules[r["name"]]
    m = GHF(Lam, lam=1.0, L=12.0, h=0.02); mk = GHF(Lam, lam=1.0, L=12.0, h=0.04)
    k0 = mk.kernel(0.0, ts); k1 = mk.kernel(0.8, ts)
    ref = dict(M01=m.M(0.1), M2=m.M(2.0), S0=m.S_tilted(0.0), S1=m.S_tilted(1.0), k0_1=k0[20], k0_6=k0[120], k0_12=k0[240],
               k1_1=k1[20], k1_12=k1[240], kmin0=k0.min())
    print(r["name"])
    for k, v in ref.items():
        ok = abs(r[k] - v) <= 1e-9 * max(abs(r[k]), abs(v)) + 1e-13
        bad += 0 if ok else 1
        print(f"   {k:6s} ref={v:+.12e} rust={r[k]:+.12e} rel={abs(r[k]-v)/max(abs(v),1e-300):.1e}" + ("" if ok else "  <-- differs"))
sys.exit(1 if bad else 0)

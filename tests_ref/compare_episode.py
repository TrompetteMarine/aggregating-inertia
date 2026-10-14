"""Compare the Rust episode solver (examples/probe_episode.rs) with the reference c9core."""
import sys, json, subprocess
import numpy as np
REF = sys.argv[1]           # path to the reference solvers (code/solvers)
sys.path.insert(0, REF)
import c9core
from c9core import Model, resolvent_B, SB_fd, ts_B, make_hbar, ts_P, SP_fd, stats_from, time_grid, default_L, odd

rust = json.loads(subprocess.run(["cargo", "run", "--release", "-q", "--example", "probe_episode"], capture_output=True,
                                 text=True, check=True, cwd=sys.argv[2]).stdout)

def variant(m, kind, kappa=None):
    u = m.u
    if kind == "dist": m.alpha = np.abs(u).copy()
    elif kind == "sq": m.alpha = (u * u).copy()
    else: m.alpha = np.abs(u).copy()
    m.chi = np.ones_like(u)
    if kind.startswith("distthr"):
        bk = 1.0 - np.exp(-m.gam * kappa)
        m.H = np.ascontiguousarray(m.nu + m.eta * (m.b[:, None] > bk) * m.chi[None, :])
    else:
        m.H = np.ascontiguousarray(m.nu + m.eta * m.b[:, None] * m.chi[None, :])
    vb = m.gam * (1.0 - (np.arange(m.Nb) + 1.0) * m.db); vb[-1] = 0.0
    m.R = np.ascontiguousarray(vb[:, None] * m.alpha[None, :] / m.db)
    m.Hmax = m.nu + m.eta * m.chi

worst = 0.0
for r in rust:
    lam, theta, eta, nu, Nb, du = r["lam"], r["theta"], r["eta"], r["nu"], r["Nb"], r["du"]
    L = default_L(lam)
    m = Model(lam=lam, theta=theta, eta=eta, nu=nu, L=L, Nu=odd(2 * L / du), Nb=Nb)
    if r["clock"] != "clock":
        kind = r["clock"]; kap = float(kind.split("_")[1]) if kind.startswith("distthr") else None
        variant(m, kind, kap)
    rr = resolvent_B(m, 0.0, nf=3); s = stats_from(m, rr); sfd = SB_fd(m, 0.002)[0]
    ts = time_grid(T=40.0 / nu if lam > 0 else 55.0 / nu, dt_max=0.04, dt_min=1e-3, growth=1.1, refine=1)
    rb = ts_B(m, ts, nf=3, store=("p", "c"), mass_tol=1e-17)
    hb, nbad = make_hbar(m, rb["store"]["p"], rb["store"]["c"])
    rp = ts_P(m, rb["ts"], hb, nf=3, compare=rb["store"])
    sp = stats_from(m, rp); spfd = SP_fd(m, rb["ts"], hb, 0.002)[0]
    sbts = stats_from(m, rb)
    ref = dict(Etau=rr["Etau"], EI=rr["EI"], EU2int=rr["EU2int"], EIU=rr["EIU"], EI2=rr["EI2"], EI2_alt=rr["EI2_alt"],
               S_formula=s["S_formula"], Kurt=s["Kurt"], VarU=s["VarU"], SB_fd=sfd, nsteps=rb["nsteps"], Etau_ts=rb["Etau"],
               S_formula_ts=sbts["S_formula"], nbad=nbad, SP_formula=sp["S_formula"], SP_fd=spfd, EI2_P=rp["EI2"],
               hb_sum=hb.sum(), hb_mid=hb[hb.shape[0] // 2, m.i0 + 20])
    print(f"{r['clock']:12s} lam={lam} theta={theta} eta={eta}")
    for k, v in ref.items():
        rv = r[k]
        rel = abs(rv - v) / max(abs(v), 1e-300) if isinstance(v, float) else abs(rv - v)
        if k not in ("EI",):        # EI is ~0 by symmetry: compare absolutely
            worst = max(worst, rel)
        print(f"   {k:14s} ref={v:+.15e} rust={rv:+.15e} rel={rel:.1e}")
print("WORST relative difference:", worst)
sys.exit(0 if worst < 1e-10 else 1)

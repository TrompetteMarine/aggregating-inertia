"""Mean lags of the true economy B and of its Markovian projection P on the magnitudes grid.

For the 48 pausing-clock economies of Section 5 (theta in {0.35, 0.7, 1}, (eta, nu) in
{(0.6, 0.7136), (0.6, 1), (2, 1), (5, 1)}, lambda in {0, 0.25, 1, 2}, full resets) this computes
    S       = (E[I U_-] + lambda E[I^2]) / (sigma^2 E tau)                        (eq. renewal)
    S Lbar  = (E int U I dt + lambda E int U J dt) / (sigma^2 E tau),  J = int I   (eq. lag)
for B (time-integrated episode solve, exact in time) and for P (backward Euler with the tabulated
projected hazard hbar = c_B / p_B, Richardson in dt over refinements 1, 2, 4).
The gaps B - P are extrapolated in the number of clock cells (Nb = 100, 200, first order); B's own
values come from fine resolvent grids extrapolated in du (second order) and Nb (first order), as in
code/solvers/c9run.py. Output: data/grid/speed_grid.json (about 40 minutes on two cores).
"""
import sys, json, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "solvers"))
from c9core import Model, ts_B, make_hbar, time_grid, default_L, odd
from c9run import rich_t, rich_b, rich_u

NU6 = 0.7136347625525


def resolvent_moments(m, f=0.0):
    """Time-integrated episode solve of B with fields q, m1 = E[I; .], m2 = E[J; .] (w2 = 1)."""
    Nb, Nu = m.Nb, m.Nu
    delta = np.zeros((Nb, Nu)); delta[0, m.i0] = 1.0
    Q = np.zeros((Nb, Nu)); M1 = np.zeros((Nb, Nu)); M2 = np.zeros((Nb, Nu))
    w2 = m.w2; m.w2 = np.ascontiguousarray(np.ones(Nu))
    try:
        p, c, pm1, cm1, pm2, cm2 = m.sweep(f, 0.0, 1.0, delta, None, None, Q, M1, M2, 3)
    finally:
        m.w2 = w2
    u = m.u
    return dict(Etau=p.sum(), EIU=(u * cm1).sum(), EUI=(u * pm1).sum(), EUJ=(u * pm2).sum())


def P_moments(m, ts, hb, f=0.0):
    """Backward-Euler episode solve of P with the same three fields."""
    Nu = m.Nu
    q = np.zeros(Nu); q[m.i0] = 1.0
    m1 = np.zeros(Nu); m2 = np.zeros(Nu)
    R0 = np.zeros(Nu); outs = [np.zeros(Nu) for _ in range(6)]
    w2 = m.w2; m.w2 = np.ascontiguousarray(np.ones(Nu))
    u = m.u
    acc = dict(Etau=0.0, EIU=0.0, EUI=0.0, EUJ=0.0)
    try:
        for n in range(hb.shape[0]):
            dt = ts[n + 1] - ts[n]; s = 1.0 / dt
            p, c, pm1, cm1, pm2, cm2 = m.sweep(f, s, s, q, m1, m2, q, m1, m2, 3, Nb=1, R=R0, H=hb[n], outs=outs)
            acc["Etau"] += dt * p.sum(); acc["EIU"] += dt * (u * cm1).sum()
            acc["EUI"] += dt * (u * pm1).sum(); acc["EUJ"] += dt * (u * pm2).sum()
    finally:
        m.w2 = w2
    return acc


def S_M1(r, lam, sig=1.0):
    S = (r["EIU"] + 2 * lam * r["EUI"]) / (sig ** 2 * r["Etau"])
    M1 = (r["EUI"] + lam * r["EUJ"]) / (sig ** 2 * r["Etau"])
    return S, M1


def point(lam, theta, eta, nu, du=0.02, Nbs=(100, 200), refs=(1, 2, 4), fine_du=(0.02, 0.01), fine_Nb=(800, 1600)):
    t0 = time.time()
    L = default_L(lam)
    T = 40.0 / nu if lam > 0 else 55.0 / nu
    gaps = {}
    for Nb in Nbs:
        m = Model(lam=lam, theta=theta, eta=eta, nu=nu, L=L, Nu=odd(2 * L / du), Nb=Nb)
        SB, M1B = S_M1(resolvent_moments(m), lam)
        gS, gM = [], []
        for ref in refs:
            ts = time_grid(T=T, dt_max=0.04, dt_min=1e-3, growth=1.1, refine=ref)
            rb = ts_B(m, ts, nf=1, store=("p", "c"), mass_tol=1e-17)
            hb, _ = make_hbar(m, rb["store"]["p"], rb["store"]["c"])
            SP, M1P = S_M1(P_moments(m, rb["ts"], hb), lam)
            gS.append(SB - SP); gM.append(M1B - M1P)
        RS, RM = rich_t(gS, list(refs)), rich_t(gM, list(refs))
        gaps[Nb] = dict(gapS=RS["R3"], gapM1=RM["R3"], gapM1_dterr=abs(RM["R3"] - RM["R2"]), SB=SB, M1B=M1B)
    n1, n2 = Nbs
    gapS = rich_b(gaps[n1]["gapS"], gaps[n2]["gapS"]); gapM1 = rich_b(gaps[n1]["gapM1"], gaps[n2]["gapM1"])
    errM1 = abs(gapM1 - gaps[n2]["gapM1"]) + gaps[n2]["gapM1_dterr"]
    fine = {}
    for h in fine_du:
        for Nb in fine_Nb:
            m = Model(lam=lam, theta=theta, eta=eta, nu=nu, L=L, Nu=odd(2 * L / h), Nb=Nb)
            S, M1 = S_M1(resolvent_moments(m), lam)
            fine[(h, Nb)] = (m.du, S, M1)
    out = {}
    for j, key in ((1, "SB"), (2, "M1B")):
        x1 = rich_b(fine[(fine_du[0], fine_Nb[0])][j], fine[(fine_du[0], fine_Nb[1])][j])
        x2 = rich_b(fine[(fine_du[1], fine_Nb[0])][j], fine[(fine_du[1], fine_Nb[1])][j])
        out[key] = rich_u(x1, fine[(fine_du[0], fine_Nb[1])][0], x2, fine[(fine_du[1], fine_Nb[1])][0])
    SB, M1B = out["SB"], out["M1B"]
    SP, M1P = SB - gapS, M1B - gapM1
    LB, LP = M1B / SB, M1P / SP
    return dict(lam=lam, theta=theta, eta=eta, nu=nu, SB=SB, SP=SP, M1B=M1B, M1P=M1P, LB=LB, LP=LP,
                gapS_pct=100 * (SB - SP) / SB, gapL_pct=100 * (LB - LP) / LB,
                gapM1_relerr=errM1 / abs(gapM1) if gapM1 != 0 else 0.0, raw=gaps, secs=time.time() - t0)


if __name__ == "__main__":
    out_path = ROOT / "data" / "grid" / "speed_grid.json"
    rows = []
    for lam in (0.0, 0.25, 1.0, 2.0):
        for theta in (0.35, 0.7, 1.0):
            for (eta, nu) in ((0.6, NU6), (0.6, 1.0), (2.0, 1.0), (5.0, 1.0)):
                r = point(lam, theta, eta, nu)
                rows.append(r)
                print(f"lam={lam} theta={theta} eta={eta} nu={nu:.4f}: gapS={r['gapS_pct']:.3f}% "
                      f"gapL={r['gapL_pct']:.3f}% LB={r['LB']:.4f} LP={r['LP']:.4f} relerr={r['gapM1_relerr']:.2e} "
                      f"({r['secs']:.0f}s)", flush=True)
                json.dump(rows, open(out_path, "w"), indent=1, default=float)

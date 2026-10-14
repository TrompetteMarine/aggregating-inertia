"""Task (f): which paths the clock selects.  Compares, cell by cell in (tau, U_tau-) and (t, U_t),
  E_B[I^2 | tau = t, U_tau- = u]  vs  E_P[I^2 | tau = t, U_tau- = u]    (completion-conditional)
  E_B[I_t^2 | tau > t, U_t = u]   vs  E_P[I_t^2 | tau > t, U_t = u]    (survivor-conditional)
and the clock A itself under both laws (for P the clock is tracked passively: same (u,b) system with the hazard
replaced by hbar(t,u), which does not depend on b).  Also decomposes E_B[I^2] - E_P[I^2] by duration.
Usage: python3 run_cells.py LAM THETA ETA NU  -> results/cells_<tag>.json"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c9core import Model, ts_B, ts_P, make_hbar, time_grid, default_L, odd

HERE = os.path.dirname(os.path.abspath(__file__))


def ts_P2D(m, ts, hbar, snaps=()):
    """Economy P on the (u,b) grid: the clock b is transported as in B but the hazard is hbar(t_{n+1}, u)."""
    Nb, Nu = m.Nb, m.Nu
    q = np.zeros((Nb, Nu)); q[0, m.i0] = 1.0
    m1 = np.zeros((Nb, Nu)); m2 = np.zeros((Nb, Nu))
    outs = [np.zeros(Nu) for _ in range(6)]
    N = hbar.shape[0]
    snapd = {}
    maxdiff = 0.0
    P = np.zeros((N, Nu))
    for n in range(N):
        dt = ts[n + 1] - ts[n]; s = 1.0 / dt
        H = np.ascontiguousarray(np.broadcast_to(hbar[n], (Nb, Nu)))
        p, c, pm1, cm1, pm2, cm2 = m.sweep(0.0, s, s, q, m1, m2, q, m1, m2, 3, H=H, outs=outs)
        P[n] = p
        if n in snaps:
            snapd[n] = dict(q=q.copy(), m2=m2.copy())
    return snapd, P


if __name__ == "__main__":
    lam, theta, eta, nu = [float(a) for a in sys.argv[1:5]]
    Nb = int(sys.argv[5]) if len(sys.argv) > 5 else 200
    tag = f"lam{lam}_th{theta}_eta{eta}_nu{nu:.4f}"
    t0 = time.time()
    L = default_L(lam)
    m = Model(lam=lam, theta=theta, eta=eta, nu=nu, L=L, Nu=odd(2 * L / 0.02), Nb=Nb)
    T = 40.0 / nu if lam > 0 else 55.0 / nu
    ts = time_grid(T=T, dt_max=0.04, dt_min=1e-3, growth=1.1, refine=2)
    tq = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0]
    snaps = [int(np.argmin(np.abs(ts[1:] - t))) for t in tq]
    rb = ts_B(m, ts, nf=3, store=("p", "c", "pm2", "cm2"), mass_tol=1e-16, snaps=snaps)
    ts = rb["ts"]
    hb, _ = make_hbar(m, rb["store"]["p"], rb["store"]["c"])
    rp = ts_P(m, ts, hb, nf=3, store=("p", "c", "pm2", "cm2"))
    snP, P2 = ts_P2D(m, ts, hb, snaps=set(snaps))
    out = dict(lam=lam, theta=theta, eta=eta, nu=nu, Nb=Nb, du=m.du,
               P2D_marginal_check=float(np.abs(P2 - rb["store"]["p"]).max() / rb["store"]["p"].max()))
    dts = np.diff(ts)
    sB, sP = rb["store"], rp["store"]
    # --- decomposition of E[I^2] by duration
    edges = [0, 0.5, 1, 2, 4, 8, np.inf]
    tt = ts[1:]
    dec = []
    for a, b in zip(edges[:-1], edges[1:]):
        k = (tt > a) & (tt <= b)
        eB = (dts[k] * sB["cm2"][k].sum(1)).sum(); eP = (dts[k] * sP["cm2"][k].sum(1)).sum()
        pr = (dts[k] * sB["c"][k].sum(1)).sum()
        dec.append(dict(tau_from=a, tau_to=b if np.isfinite(b) else "inf", prob=float(pr), EI2_B_part=float(eB),
                        EI2_P_part=float(eP), cond_B=float(eB / pr), cond_P=float(eP / pr)))
    out["by_duration"] = dec
    out["EI2_B"] = float((dts * sB["cm2"].sum(1)).sum()); out["EI2_P"] = float((dts * sP["cm2"].sum(1)).sum())
    # --- cells
    au = np.abs(m.u)
    ubins = [(0, 0.5 * theta), (0.5 * theta, theta), (theta, 1.5 * theta), (1.5 * theta, 2 * theta),
             (2 * theta, 3 * theta), (3 * theta, np.inf)]
    a_of_b = -np.log(1.0 - m.b) / m.gam
    cells = []
    for t, n in zip(tq, snaps):
        for (lo, hi) in ubins:
            k = (au >= lo) & (au < hi)
            cB = sB["c"][n][k].sum(); cP = sP["c"][n][k].sum()
            pB = sB["p"][n][k].sum()
            if cB < 1e-14:
                continue
            row = dict(t=float(ts[n + 1]), u_lo=lo, u_hi=hi if np.isfinite(hi) else "inf",
                       compl_rate=float(cB), compl_rate_P=float(cP),
                       EI2_compl_B=float(sB["cm2"][n][k].sum() / cB), EI2_compl_P=float(sP["cm2"][n][k].sum() / cP),
                       EI2_surv_B=float(sB["pm2"][n][k].sum() / pB), EI2_surv_P=float(sP["pm2"][n][k].sum() / pB))
            q = rb["snaps"][n]["q"]; m2 = rb["snaps"][n]["m2"]
            Hq = m.H * q
            row["EA_compl_B"] = float((a_of_b[:, None] * Hq)[:, k].sum() / Hq[:, k].sum())
            row["EA_surv_B"] = float((a_of_b[:, None] * q)[:, k].sum() / q[:, k].sum())
            qP = snP[n]["q"]; m2P = snP[n]["m2"]
            row["EA_surv_P"] = float((a_of_b[:, None] * qP)[:, k].sum() / qP[:, k].sum())   # = completion-conditional for P
            # within-cell covariance of the clock with I^2 among survivors
            for nm, qq, mm in (("B", q, m2), ("P", qP, m2P)):
                Z = qq[:, k].sum(); EA = (a_of_b[:, None] * qq)[:, k].sum() / Z
                EI2 = mm[:, k].sum() / Z; EAI2 = (a_of_b[:, None] * mm)[:, k].sum() / Z
                row["cov_A_I2_surv_" + nm] = float(EAI2 - EA * EI2)
            cells.append(row)
    out["cells"] = cells
    # --- conditional on duration only
    bydur = []
    for t, n in zip(tq, snaps):
        bydur.append(dict(t=float(ts[n + 1]), EI2_compl_B=float(sB["cm2"][n].sum() / sB["c"][n].sum()),
                          EI2_compl_P=float(sP["cm2"][n].sum() / sP["c"][n].sum()),
                          EI2_surv_B=float(sB["pm2"][n].sum() / sB["p"][n].sum()),
                          EI2_surv_P=float(sP["pm2"][n].sum() / sP["p"][n].sum())))
    out["by_t"] = bydur
    out["secs"] = time.time() - t0
    json.dump(out, open(os.path.join(HERE, "results", f"cells_{tag}.json"), "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))

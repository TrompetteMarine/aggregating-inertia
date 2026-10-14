"""Task (d): numerical check of claim 1 (B and P have the same p(t,u), hence the same joint law of (tau, U_tau-),
duration law, revision-size law and stationary U-law).
  (1) same discretisation: P with hbar^{n+1} = c^{n+1}/p^{n+1} on B's grid (discrete identity, round-off level);
  (2) cross discretisation: P on a time grid twice finer with hbar interpolated linearly in time from the coarse
      B run, compared with B on the coarse and on the fine grid; and P with hbar from a coarser b-grid (Nb/2)
      compared with B on Nb.  Differences must be at the level of B's own discretisation error and shrink with it.
Usage: python3 run_claim1.py   -> results/claim1.json"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c9core import Model, ts_B, ts_P, make_hbar, time_grid, default_L, odd
from c9run import NU6

HERE = os.path.dirname(os.path.abspath(__file__))


def laws(m, r, ts):
    """duration density and survival on the step grid, revision CDF and stationary CDF in u, moments."""
    occ, rev = r["occ"], r["rev"]
    u = m.u
    Etau = occ.sum()
    d2 = (u ** 2 * rev).sum() / rev.sum(); d4 = (u ** 4 * rev).sum() / rev.sum()
    return dict(dens=r["dens"], surv=r["surv"], revcdf=np.cumsum(rev) / rev.sum(), occcdf=np.cumsum(occ) / Etau,
                Etau=Etau, Ed2=d2, kurt=d4 / d2 ** 2, var=(u ** 2 * occ).sum() / Etau)


def compare(A, B, cA=None, cB=None, rowsA=None, rowsB=None):
    n = min(len(A["dens"]) if rowsA is None else len(rowsA), len(B["dens"]) if rowsB is None else len(rowsB))
    ra = np.arange(n) if rowsA is None else np.asarray(rowsA)[:n]
    rb = np.arange(n) if rowsB is None else np.asarray(rowsB)[:n]
    out = dict(dur_density_rel=float(np.abs(A["dens"][ra] - B["dens"][rb]).max() / B["dens"].max()),
               survival_abs=float(np.abs(A["surv"][ra] - B["surv"][rb]).max()),
               revision_KS=float(np.abs(A["revcdf"] - B["revcdf"]).max()),
               stationary_KS=float(np.abs(A["occcdf"] - B["occcdf"]).max()),
               Etau_rel=float(abs(A["Etau"] / B["Etau"] - 1)), Ed2_rel=float(abs(A["Ed2"] / B["Ed2"] - 1)),
               kurt_rel=float(abs(A["kurt"] / B["kurt"] - 1)), var_rel=float(abs(A["var"] / B["var"] - 1)))
    if cA is not None:
        out["joint_c_rel"] = float(np.abs(cA[ra] - cB[rb]).max() / cB.max())
    return out


def interp_table(ts_coarse, hb_coarse, ts_fine, nu):
    """hbar at fine output times ts_fine[1:] by linear interpolation in t of the coarse table (row 0: t=0, hbar=nu)."""
    tn = ts_coarse[:len(hb_coarse) + 1]
    tab = np.vstack([np.full((1, hb_coarse.shape[1]), nu), hb_coarse])
    tf = ts_fine[1:]
    k = np.clip(np.searchsorted(tn, tf, side="right") - 1, 0, len(tn) - 2)
    w = np.clip((tf - tn[k]) / (tn[k + 1] - tn[k]), 0, 1)[:, None]
    return np.ascontiguousarray((1 - w) * tab[k] + w * tab[k + 1])


def point(lam, theta, eta, nu, du=0.02, Nb=200, dtmax=0.04):
    L = default_L(lam)
    T = 40.0 / nu if lam > 0 else 55.0 / nu
    res = dict(lam=lam, theta=theta, eta=eta, nu=nu, du=du, Nb=Nb, dtmax=dtmax)
    mB = Model(lam=lam, theta=theta, eta=eta, nu=nu, L=L, Nu=odd(2 * L / du), Nb=Nb)
    runs = {}
    for ref in (1, 2, 4):
        ts = time_grid(T=T, dt_max=dtmax, dt_min=1e-3, growth=1.1, refine=ref)
        rb = ts_B(mB, ts, nf=1, store=("p", "c"), mass_tol=1e-16)
        hb, nbad = make_hbar(mB, rb["store"]["p"], rb["store"]["c"])
        rp = ts_P(mB, rb["ts"], hb, nf=1, store=("p", "c"))
        runs[ref] = dict(ts=rb["ts"], rb=rb, hb=hb, rp=rp, LB=laws(mB, rb, rb["ts"]), LP=laws(mB, rp, rb["ts"]))
    # (1) identity on the same grid
    res["identity"] = {}
    for ref, R in runs.items():
        d = compare(R["LP"], R["LB"], R["rp"]["store"]["c"], R["rb"]["store"]["c"])
        d["joint_p_rel"] = float(np.abs(R["rp"]["store"]["p"] - R["rb"]["store"]["p"]).max() / R["rb"]["store"]["p"].max())
        res["identity"][f"ref{ref}"] = d
    # (2a) cross discretisation in time: P on the 2x finer grid with interpolated coarse hbar
    res["cross_time"] = {}
    for k in (1, 2):
        Rc, Rf = runs[k], runs[2 * k]
        hbi = interp_table(Rc["ts"], Rc["hb"], Rf["ts"], nu)
        rpx = ts_P(mB, Rf["ts"], hbi, nf=1, store=("c",))
        LPx = laws(mB, rpx, Rf["ts"])
        nco = min(len(Rc["rb"]["dens"]), len(Rf["rb"]["dens"]) // 2)
        rows_f = 2 * np.arange(nco) + 1          # fine rows at the coarse output times
        rows_c = np.arange(nco)
        res["cross_time"][f"hbar_ref{k}_P_ref{2*k}"] = dict(
            P_fine_vs_B_coarse=compare(LPx, Rc["LB"], rpx["store"]["c"], Rc["rb"]["store"]["c"], rows_f, rows_c),
            P_fine_vs_B_fine=compare(LPx, Rf["LB"], rpx["store"]["c"], Rf["rb"]["store"]["c"]),
            B_fine_vs_B_coarse=compare(Rf["LB"], Rc["LB"], Rf["rb"]["store"]["c"], Rc["rb"]["store"]["c"], rows_f, rows_c))
    # (2b) cross discretisation in b: hbar from a coarser clock grid (Nb/2), P vs B on Nb
    mH = Model(lam=lam, theta=theta, eta=eta, nu=nu, L=L, Nu=odd(2 * L / du), Nb=Nb // 2)
    ts = runs[1]["ts"]
    rbH = ts_B(mH, ts, nf=1, store=("p", "c"))
    hbH, _ = make_hbar(mH, rbH["store"]["p"], rbH["store"]["c"])
    rpH = ts_P(mB, ts, hbH, nf=1, store=("c",))
    LH = laws(mB, rpH, ts); LBH = laws(mH, rbH, ts)
    res["cross_b"] = dict(P_hbarNbhalf_vs_B_Nb=compare(LH, runs[1]["LB"], rpH["store"]["c"], runs[1]["rb"]["store"]["c"]),
                          B_Nbhalf_vs_B_Nb=compare(LBH, runs[1]["LB"], rbH["store"]["c"], runs[1]["rb"]["store"]["c"]))
    return res


if __name__ == "__main__":
    pts = [(1.0, 0.35, 5.0, 1.0), (1.0, 0.7, 0.6, NU6), (0.25, 1.0, 5.0, 1.0), (2.0, 0.35, 5.0, 1.0),
           (0.0, 0.7, 2.0, 1.0)]
    allres = []
    for p in pts:
        t0 = time.time()
        r = point(*p)
        r["secs"] = time.time() - t0
        allres.append(r)
        print(json.dumps(r, indent=None, default=float)[:3000], flush=True)
        json.dump(allres, open(os.path.join(HERE, "results", "claim1.json"), "w"), indent=1, default=float)

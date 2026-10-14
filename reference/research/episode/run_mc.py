"""Task (c) independent check: Monte Carlo of economy P with the tabulated hbar (and of economy B with the
true clock), common random numbers, killing integrated out along each path (Rao-Blackwell).
Usage: python3 run_mc.py LAM THETA ETA NU [NPATHS] [DT_MC]
Writes results/mc_<tag>.json."""
import sys, os, json, time, ctypes
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c9core import Model, ts_B, make_hbar, time_grid, default_L, odd, ts_P, stats_from
from c9run import full_point, run_point, NU6

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
_lib = ctypes.CDLL(os.path.join(HERE, "libmc.so"))
_dp = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
_ip = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
_lib.mc.restype = ctypes.c_int
_lib.mc.argtypes = [ctypes.c_long, ctypes.c_uint64] + [ctypes.c_double] * 8 + [ctypes.c_int, ctypes.c_double,
                    _dp, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, _ip, _dp, _dp,
                    ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_int,
                    _dp, _dp, _dp, _dp, _dp, _dp]


def hbar_table(lam, theta, eta, nu, du=0.02, Nbs=(200, 400), refs=(1, 2), dtmax=0.04, dtmin=1e-3, growth=1.1):
    """hbar on the coarse time grid, Richardson-extrapolated in dt (refs 1,2) and in Nb (Nbs).
    Row 0 is t = 0 where hbar = nu exactly (A_0 = 0)."""
    L = default_L(lam)
    T = 40.0 / nu if lam > 0 else 55.0 / nu
    tabs = {}
    for Nb in Nbs:
        m = Model(lam=lam, theta=theta, eta=eta, nu=nu, L=L, Nu=odd(2 * L / du), Nb=Nb)
        for ref in refs:
            ts = time_grid(T=T, dt_max=dtmax, dt_min=dtmin, growth=growth, refine=ref)
            rb = ts_B(m, ts, nf=1, store=("p", "c"), mass_tol=1e-15)
            hb, _ = make_hbar(m, rb["store"]["p"], rb["store"]["c"])
            tabs[(Nb, ref)] = (rb["ts"], hb)
    # common coarse nodes
    tc = tabs[(Nbs[0], refs[0])][0]
    nC = min(len(tabs[(Nb, refs[0])][0]) for Nb in Nbs) - 1
    nF = min(len(tabs[(Nb, refs[1])][0]) for Nb in Nbs) - 1
    n = min(nC, nF // refs[1])
    def coarse(Nb, ref):
        ts, hb = tabs[(Nb, ref)]
        k = ref // refs[0]
        return hb[k - 1::k][:n]          # rows at t = tc[1..n]
    H = {}
    for Nb in Nbs:
        H[Nb] = 2 * coarse(Nb, refs[1]) - coarse(Nb, refs[0])
    hstar = 2 * H[Nbs[1]] - H[Nbs[0]]
    hstar = np.clip(hstar, m.nu, m.Hmax[None, :])
    hstar[:, m.chi == 0] = m.nu
    table = np.vstack([np.full((1, m.Nu), nu), hstar])
    tnodes = tc[:n + 1]
    hraw = np.vstack([np.full((1, m.Nu), nu), coarse(Nbs[1], refs[1])])
    return m, tnodes, np.ascontiguousarray(table), np.ascontiguousarray(hraw)


def run_mc(m, tnodes, table, lam, theta, eta, nu, f=0.0, npaths=200000, dt=0.002, seed=12345, tmax=None,
           wtol=1e-10, nbatch=20, tbin=0.05, ubin=0.05, ulim=5.0):
    tmax = tnodes[-1] if tmax is None else min(tmax, tnodes[-1])
    kmax = int(tmax / dt)
    tk = np.arange(kmax + 1) * dt
    row = np.searchsorted(tnodes, tk, side="right") - 1
    row = np.clip(row, 0, len(tnodes) - 2)
    w = (tk - tnodes[row]) / (tnodes[row + 1] - tnodes[row])
    w = np.clip(w, 0.0, 1.0)
    nb_t = int(round(10.0 / tbin)); nb_u = int(round(2 * ulim / ubin))
    out = np.zeros((npaths, 10))
    hists = [np.zeros((nbatch, nb_t)), np.zeros((nbatch, nb_t)), np.zeros((nbatch, nb_u)),
             np.zeros((nbatch, nb_u)), np.zeros((nbatch, nb_u)), np.zeros((nbatch, nb_u))]
    t0 = time.time()
    _lib.mc(npaths, seed, lam, m.sig, f, theta, eta, nu, m.gam, dt, kmax, wtol, table, table.shape[0], m.Nu,
            float(m.u[0]), float(m.du), row.astype(np.int32), np.ascontiguousarray(w), out,
            nb_t, tbin, nb_u, -ulim, ubin, nbatch, *hists)
    return out, hists, time.time() - t0, dict(nb_t=nb_t, tbin=tbin, nb_u=nb_u, ubin=ubin, ulim=ulim)


def ratio_stats(X, Y):
    """S = mean X / mean Y with delta-method standard error; also per-path linearisation."""
    mx, my = X.mean(), Y.mean()
    S = mx / my
    lin = (X - S * Y) / my
    return S, lin.std(ddof=1) / np.sqrt(len(X)), lin


def analyse(out, lam, sig=1.0):
    tauB, tauP, U2B, U2P, IUB, IUP, I2B, I2P, I1B, I1P = out.T
    r = {}
    SB, seB, linB = ratio_stats(IUB + lam * I2B, sig ** 2 * tauB)
    SP, seP, linP = ratio_stats(IUP + lam * I2P, sig ** 2 * tauP)
    d = linB - linP
    r.update(S_B=SB, se_S_B=seB, S_P=SP, se_S_P=seP, gap=SB - SP, se_gap=d.std(ddof=1) / np.sqrt(len(d)))
    # Girsanov-split form: S = (E int U^2 + lam E I^2 / 2) / (sig^2 E tau)
    SB2, seB2, lB2 = ratio_stats(U2B + 0.5 * lam * I2B, sig ** 2 * tauB)
    SP2, seP2, lP2 = ratio_stats(U2P + 0.5 * lam * I2P, sig ** 2 * tauP)
    d2 = lB2 - lP2
    r.update(S_B_split=SB2, se_S_B_split=seB2, S_P_split=SP2, se_S_P_split=seP2, gap_split=SB2 - SP2,
             se_gap_split=d2.std(ddof=1) / np.sqrt(len(d2)))
    n = len(tauB)
    for name, x in [("Etau_B", tauB), ("Etau_P", tauP), ("EI2_B", I2B), ("EI2_P", I2P), ("EIU_B", IUB),
                    ("EIU_P", IUP), ("EU2int_B", U2B), ("EU2int_P", U2P)]:
        r[name] = x.mean(); r["se_" + name] = x.std(ddof=1) / np.sqrt(n)
    dd = I2B - I2P
    r["EI2_diff"] = dd.mean(); r["se_EI2_diff"] = dd.std(ddof=1) / np.sqrt(n)
    dt_ = tauB - tauP
    r["Etau_diff"] = dt_.mean(); r["se_Etau_diff"] = dt_.std(ddof=1) / np.sqrt(n)
    du2 = U2B - U2P
    r["EU2int_diff"] = du2.mean(); r["se_EU2int_diff"] = du2.std(ddof=1) / np.sqrt(n)
    return r


def pde_reference(lam, theta, eta, nu, info):
    """PDE histograms of B (time-stepped, Nb=400, refine 2) for the duration density, revision law and occupation."""
    L = default_L(lam)
    m = Model(lam=lam, theta=theta, eta=eta, nu=nu, L=L, Nu=odd(2 * L / 0.02), Nb=400)
    T = 40.0 / nu if lam > 0 else 55.0 / nu
    ts = time_grid(T=T, dt_max=0.04, dt_min=1e-3, growth=1.1, refine=2)
    rb = ts_B(m, ts, nf=1, store=("c",), mass_tol=1e-15)
    t = rb["ts"]; dts = np.diff(t)
    dens = rb["dens"]                       # sum_u c at t_{n+1}
    nb_t, tbin = info["nb_t"], info["tbin"]
    durh = np.zeros(nb_t)
    it = (t[1:] / tbin).astype(int)
    ok = it < nb_t
    np.add.at(durh, it[ok], (dens * dts)[ok])
    # revision and occupation laws: CDF at cell faces, interpolated at bin edges
    faces = np.concatenate([[m.u[0] - m.du / 2], m.u + m.du / 2])
    edges = -info["ulim"] + info["ubin"] * np.arange(info["nb_u"] + 1)
    def binned(mass):
        cdf = np.concatenate([[0.0], np.cumsum(mass)])
        return np.diff(np.interp(edges, faces, cdf))
    return dict(dur=durh, rev=binned(rb["rev"]), occ=binned(rb["occ"]), Etau=rb["Etau"])


if __name__ == "__main__":
    lam, theta, eta, nu = [float(a) for a in sys.argv[1:5]]
    npaths = int(sys.argv[5]) if len(sys.argv) > 5 else 200000
    dts = [float(a) for a in sys.argv[6].split(",")] if len(sys.argv) > 6 else [0.002]
    tag = f"lam{lam}_th{theta}_eta{eta}_nu{nu:.4f}"
    t0 = time.time()
    m, tnodes, table, hraw = hbar_table(lam, theta, eta, nu)
    print(f"hbar table: {table.shape}, t_end={tnodes[-1]:.1f} ({time.time()-t0:.0f}s)", flush=True)
    res = dict(lam=lam, theta=theta, eta=eta, nu=nu, npaths=npaths, runs=[])
    # PDE values (extrapolated) for comparison
    fp = full_point(lam, theta, eta, nu)
    res["pde"] = {k: float(fp[k]) for k in ("SB_formula", "SB_fd", "SP_formula", "SP_fd", "gap_formula", "gap_fd",
                                              "EI2_B", "EI2_P", "Etau", "VarU", "F")}
    print("PDE:", res["pde"], flush=True)
    ref = None
    for dt in dts:
        out, hists, secs, info = run_mc(m, tnodes, table, lam, theta, eta, nu, npaths=npaths, dt=dt)
        a = analyse(out, lam)
        # finite-difference response with common random numbers (f = +-delta), fewer paths
        dlt = 0.05
        nfd = max(npaths // 2, 1000)
        op, _, _, _ = run_mc(m, tnodes, table, lam, theta, eta, nu, f=+dlt, npaths=nfd, dt=dt, seed=777)
        om, _, _, _ = run_mc(m, tnodes, table, lam, theta, eta, nu, f=-dlt, npaths=nfd, dt=dt, seed=777)
        fd = {}
        for nm, (ci, ti) in {"B": (8, 0), "P": (9, 1)}.items():
            Mp = op[:, ci].mean() / op[:, ti].mean(); Mm = om[:, ci].mean() / om[:, ti].mean()
            lp = (op[:, ci] - Mp * op[:, ti]) / op[:, ti].mean(); lm = (om[:, ci] - Mm * om[:, ti]) / om[:, ti].mean()
            fd["S_fd_" + nm] = (Mp - Mm) / (2 * dlt)
            fd["se_S_fd_" + nm] = (lp - lm).std(ddof=1) / np.sqrt(nfd) / (2 * dlt)
        # gap of FD estimates with CRN
        lpB = (op[:, 8] - op[:, 8].mean() / op[:, 0].mean() * op[:, 0]) / op[:, 0].mean()
        lmB = (om[:, 8] - om[:, 8].mean() / om[:, 0].mean() * om[:, 0]) / om[:, 0].mean()
        lpP = (op[:, 9] - op[:, 9].mean() / op[:, 1].mean() * op[:, 1]) / op[:, 1].mean()
        lmP = (om[:, 9] - om[:, 9].mean() / om[:, 1].mean() * om[:, 1]) / om[:, 1].mean()
        fd["gap_fd"] = fd["S_fd_B"] - fd["S_fd_P"]
        fd["se_gap_fd"] = ((lpB - lmB) - (lpP - lmP)).std(ddof=1) / np.sqrt(nfd) / (2 * dlt)
        fd["delta"] = dlt
        a.update(fd)
        # distributions: batch means and standard errors; B vs P (claim 1) and MC vs PDE
        if ref is None:
            ref = pde_reference(lam, theta, eta, nu, info)
        nbatch = hists[0].shape[0]
        dist = {}
        for name, hb_, hp_ in [("dur", hists[0], hists[1]), ("rev", hists[2], hists[3]), ("occ", hists[4], hists[5])]:
            B = hb_.sum(0) / npaths; P = hp_.sum(0) / npaths
            seB = hb_.std(0, ddof=1) * nbatch / npaths / np.sqrt(nbatch)
            diff_b = (hb_ - hp_) * nbatch / npaths
            seD = diff_b.std(0, ddof=1) / np.sqrt(nbatch)
            if name == "occ":
                B = B / a["Etau_B"]; P = P / a["Etau_P"]; seB = seB / a["Etau_B"]; seD = seD / a["Etau_B"]
                pde = ref["occ"] / ref["Etau"]
            else:
                pde = ref[name]
            mask = seB > 0
            dist[name] = dict(max_abs_BminusP=float(np.abs(B - P).max()),
                              max_z_BminusP=float(np.max(np.abs(B - P)[seD > 0] / seD[seD > 0])),
                              max_abs_MC_B_minus_PDE=float(np.abs(B - pde).max()),
                              max_z_MC_B_minus_PDE=float(np.max(np.abs(B - pde)[mask] / seB[mask])),
                              max_abs_MC_P_minus_PDE=float(np.abs(P - pde).max()),
                              peak=float(pde.max()))
        a["dist"] = dist
        a.update(dt=dt, secs=secs)
        res["runs"].append({k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in a.items()})
        print(f"dt={dt}: S_B={a['S_B']:.5f}+-{a['se_S_B']:.5f} S_P={a['S_P']:.5f}+-{a['se_S_P']:.5f} "
              f"gap={a['gap']:+.5f}+-{a['se_gap']:.5f}  FD: S_B={a['S_fd_B']:.5f}+-{a['se_S_fd_B']:.5f} "
              f"S_P={a['S_fd_P']:.5f}+-{a['se_S_fd_P']:.5f} gap={a['gap_fd']:+.5f}+-{a['se_gap_fd']:.5f} ({secs:.0f}s)",
              flush=True)
        print("   dist:", json.dumps(dist), flush=True)
    json.dump(res, open(os.path.join(OUT, f"mc_{tag}.json"), "w"), indent=1, default=float)
    print(f"total {time.time()-t0:.0f}s")

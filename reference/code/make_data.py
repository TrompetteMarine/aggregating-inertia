"""Generate every number and every data file used by the figures and the text of Paper E.

Run from any directory:  python3 code/make_data.py
Writes: data/*.csv (figure data), data/records.json (all quoted numbers with provenance),
figures/tikz/values.tex (TeX macros for every number quoted in the text).

Solvers (code/solvers):
  c9core/c9run  episode solver on (u, b), b = 1 - exp(-gamma A); the true economy B, its Markovian
                projection P, and the area moments E[I^2], E[I U] (finite volumes, backward Euler,
                Richardson extrapolation in all steps).
  fvsolver      stationary forward generator on (u, a) with resets; S, the kernel k(t), its Laplace
                transform, rightmost eigenvalues and argument-principle certificates.
Grid results for the magnitudes figure were produced by the same solvers (research runs of
8-11 October 2026) and are read from data/grid/*.json.
"""
import sys, os, json, glob, time
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.integrate import cumulative_trapezoid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE / "solvers"))
DATA = ROOT / "data"; DATA.mkdir(exist_ok=True)
TIKZ = ROOT / "figures" / "tikz"

import c9core
from c9core import ts_B, ts_P, make_hbar, time_grid, default_L, odd, stats_from
from fvsolver import Params, Model as FVModel, default_grid
from analysis import first_crossing, kernel_stats, certify_count, slowest_rate
from fvsolver import Shifted
from scipy.optimize import brentq
from ghf1d import GHF

# The running economy (Example 1): time in units of the pull, full reset.
LAM, GAM, SIG, NU, THETA, ETA = 1.0, 1.0, 1.0, 1.0, 0.7, 5.0
REC = {}


def act(u, theta=THETA):
    z = np.clip((np.abs(np.asarray(u, float)) - theta) / theta, 0, 1)
    out = (z >= 1).astype(float)
    m = (z > 0) & (z < 1)
    a = np.exp(-1 / z[m]); b = np.exp(-1 / (1 - z[m]))
    out[m] = a / (a + b)
    return out


def hazard(u, a):
    return NU + ETA * act(u) * (-np.expm1(-GAM * np.asarray(a, float)))


# --------------------------------------------------------------------------- Figure 1: one episode
def episode():
    tc = 8.0
    kb = ([0, .6, 1.4, 2.2, 3.1, 3.8, 4.8, 5.4, 6.4, 7.2, 8.0],
          [0.0, .38, .72, 1.55, 1.3, .45, .35, .9, 1.65, 1.9, 2.15])
    ka = ([8.0, 8.7, 9.5, 10.2], [0.0, 0.22, 0.52, 0.40])
    t1 = np.linspace(0, tc, 1601); u1 = PchipInterpolator(*kb)(t1)
    t2 = np.linspace(tc, 10.2, 441); u2 = PchipInterpolator(*ka)(t2)
    a1 = cumulative_trapezoid(act(u1), t1, initial=0); a2 = cumulative_trapezoid(act(u2), t2, initial=0)
    h1 = hazard(u1, a1); h2 = hazard(u2, a2)
    for name, t, u, a, h in (("before", t1, u1, a1, h1), ("after", t2, u2, a2, h2)):
        np.savetxt(DATA / f"episode_{name}.csv", np.column_stack([t, u, a, h]), delimiter=",",
                   header="time,assessment,processing,hazard", comments="", fmt="%.6f")
    poly = np.vstack([np.column_stack([t1, u1]), [[tc, 0.0], [0.0, 0.0]]])
    np.savetxt(DATA / "episode_area.csv", poly, delimiter=",", header="time,assessment", comments="", fmt="%.6f")
    inside = (u1 <= THETA) & (t1 > 3.0) & (t1 < 6.0)
    pause = t1[inside]
    ustar = 0.40
    f_b, f_a = PchipInterpolator(*kb), PchipInterpolator(*ka)
    from scipy.optimize import brentq
    tr1 = brentq(lambda s: f_b(s) - ustar, 3.8, 4.8); tr2 = brentq(lambda s: f_a(s) - ustar, 8.7, 9.5)
    marks = dict(EpPauseLo=float(pause.min()), EpPauseHi=float(pause.max()), EpTc=tc,
                 EpUpre=float(u1[-1]), EpApre=float(a1[-1]), EpTrOne=tr1, EpTrTwo=tr2, EpUstar=ustar,
                 EpArOne=float(np.interp(tr1, t1, a1)), EpArTwo=float(np.interp(tr2, t2, a2)),
                 EpHazPre=float(h1[-1]), EpHazRone=float(np.interp(tr1, t1, h1)),
                 EpHazRtwo=float(np.interp(tr2, t2, h2)), EpArea=float(np.trapezoid(u1, t1)))
    assert marks["EpArTwo"] == 0.0 and marks["EpArOne"] > 0.5
    REC["episode"] = marks
    return marks


# --------------------------------------------------------------------------- Figure 2: one record
def record():
    L = default_L(LAM)
    m = c9core.Model(lam=LAM, theta=THETA, eta=ETA, nu=NU, L=L, Nu=odd(2 * L / 0.02), Nb=200)
    ts = time_grid(T=40.0 / NU, dt_max=0.04, dt_min=1e-3, growth=1.1, refine=2)
    tstar = 1.0
    nstar = int(np.argmin(np.abs(ts[1:] - tstar)))
    rb = ts_B(m, ts, nf=3, store=("p", "c", "cm2"), mass_tol=1e-16, snaps=[nstar])
    ts = rb["ts"]
    hb, nbad = make_hbar(m, rb["store"]["p"], rb["store"]["c"])
    rp = ts_P(m, ts, hb, nf=3, store=("p", "c", "cm2"), compare=rb["store"])
    tt = ts[1:]
    du = m.du
    # duration densities and conditional squared area at completion
    dB, dP = rb["dens"], rp["dens"]
    eB = rb["store"]["cm2"].sum(1) / np.maximum(dB, 1e-300)
    eP = rp["store"]["cm2"].sum(1) / np.maximum(dP, 1e-300)
    keep = (tt <= 6.0)
    idx = np.flatnonzero(keep)[::3]
    np.savetxt(DATA / "record_duration.csv", np.column_stack([tt[idx], dB[idx], dP[idx], eB[idx], eP[idx]]),
               delimiter=",", header="t,durB,durP,areaB,areaP", comments="", fmt="%.7f")
    # revision-size densities (Delta = -U_{tau-}; symmetric)
    revB = rb["rev"] / (rb["rev"].sum() * du); revP = rp["rev"] / (rp["rev"].sum() * du)
    sel = np.abs(m.u) <= 4.0
    np.savetxt(DATA / "record_revision.csv", np.column_stack([-m.u[sel], revB[sel], revP[sel]]),
               delimiter=",", header="delta,revB,revP", comments="", fmt="%.7f")
    # the hidden clock at age tstar: projected hazard and the spread of the true hazard among survivors
    q = rb["snaps"][nstar]["q"]          # (Nb, Nu) survivor masses on (b, u)
    H = m.H                               # (Nb, Nu)
    rows = []
    for i in range(m.Nu):
        w = q[:, i]
        if w.sum() < 1e-9 * q.sum() or abs(m.u[i]) > 3.0:
            continue
        hbar = (w * H[:, i]).sum() / w.sum()
        order = np.argsort(H[:, i]); cw = np.cumsum(w[order]) / w.sum()
        q10 = H[order, i][np.searchsorted(cw, 0.10)]; q90 = H[order, i][np.searchsorted(cw, 0.90)]
        rows.append((m.u[i], hbar, q10, q90, w.sum() / du))
    rows = np.array(rows)
    np.savetxt(DATA / "record_hazard.csv", rows, delimiter=",", header="u,hbar,q10,q90,surv",
               comments="", fmt="%.7f")
    band = np.vstack([rows[:, [0, 2]], rows[::-1][:, [0, 3]]])
    np.savetxt(DATA / "record_hazard_band.csv", band, delimiter=",", header="u,h", comments="", fmt="%.7f")
    sB, sP = stats_from(m, rb), stats_from(m, rp)
    REC["record"] = dict(tstar=float(tt[nstar]), nbad=int(nbad),
                         maxdiff_p=float(rp["maxdiff"]["p"]), maxdiff_c=float(rp["maxdiff"]["c"]),
                         EI2_B=float(rb["EI2"]), EI2_P=float(rp["EI2"]), Etau=float(rb["Etau"]),
                         S_B_ts=sB["S_formula"], S_P_ts=sP["S_formula"], F=sB["F"], VarU=sB["VarU"],
                         Kurt=sB["Kurt"], ALO=sB["ALO"],
                         dur_maxdiff=float(np.abs(dB - dP).max() / dB.max()),
                         rev_maxdiff=float(np.abs(revB - revP).max() / revB.max()))
    return REC["record"]


# --------------------------------------------------------------------------- Figure 3: magnitudes
def magnitudes():
    rows = []
    for fn in sorted(glob.glob(str(DATA / "grid" / "grid_lam*.json"))):
        for r in json.load(open(fn)):
            rows.append(dict(family="clock", lam=r["lam"], theta=r["theta"], eta=r["eta"], nu=r["nu"],
                             SB=r["SB_fd"], SP=r["SP_fd"], VarU=r["VarU"], F=r["F"], ALO=r["ALO"]))
    for fn in sorted(glob.glob(str(DATA / "grid" / "variant_*.json"))):
        for r in json.load(open(fn)):
            fam = {"dist": "dist", "sq": "sq"}.get(r["variant"], "thr")
            rows.append(dict(family=fam, lam=r["lam"], theta=None, eta=r["eta"], nu=r["nu"],
                             SB=r["SB_fd"], SP=r["SP_fd"], VarU=r["VarU"], F=r["F"], ALO=r["ALO"]))
    out = []
    rng = np.random.default_rng(3)
    xpos = {0.0: 0, 0.25: 1, 1.0: 2, 2.0: 3}
    for r in rows:
        gap = 100 * (r["SB"] - r["SP"]) / r["SB"]
        kerr = 100 * (r["ALO"] - r["SB"]) / r["SB"]
        x = xpos[r["lam"]] + (0.18 if r["family"] != "clock" else -0.12) + 0.1 * (rng.random() - 0.5)
        out.append((x, r["lam"], {"clock": 0, "dist": 1, "sq": 2, "thr": 3}[r["family"]], gap, kerr))
    out = np.array(out)
    np.savetxt(DATA / "magnitudes.csv", out, delimiter=",", header="x,lam,family,gap,kerr", comments="",
               fmt="%.6f")
    clock = out[out[:, 2] == 0]; other = out[out[:, 2] > 0]
    summ = {}
    for lam in (0.0, 0.25, 1.0, 2.0):
        c = clock[clock[:, 1] == lam]
        summ[str(lam)] = dict(n=int(len(c)), gap_min=float(c[:, 3].min()), gap_max=float(c[:, 3].max()),
                              kerr_min=float(c[:, 4].min()), kerr_max=float(c[:, 4].max()))
    summ["other"] = dict(n=int(len(other)), gap_min=float(other[other[:, 1] > 0][:, 3].min()),
                         gap_max=float(other[other[:, 1] > 0][:, 3].max()),
                         lam0_max_abs=float(np.abs(other[other[:, 1] == 0][:, 3]).max()))
    summ["all_lam_pos_max_abs_gap"] = float(np.abs(out[out[:, 1] > 0][:, 3]).max())
    summ["all_lam0_max_abs_gap"] = float(np.abs(out[out[:, 1] == 0][:, 3]).max())
    summ["n_total"] = int(len(out))
    REC["magnitudes"] = summ
    return summ


def dominant_root_curve(mod, S, M1, gS=np.linspace(0.6, 1.4, 41), gap=None):
    """x_g solving g khat(x) = 1 on (-gap, inf) for g = gS/S (Theorem 2(b)); khat is real and decreasing,
    so the root is bracketed around the slowing-down approximation and refined by Brent's method."""
    gap = slowest_rate(mod) if gap is None else gap
    kh = lambda x: float(np.real(mod.khat(x)))
    xs = []
    for r in gS:
        g = r / S
        F = lambda x: g * kh(x) - 1.0
        if abs(r - 1.0) < 1e-12:
            xs.append(0.0); continue
        x0 = (r - 1.0) * S / M1
        w = 0.1 * abs(x0) + 1e-3
        floor = -gap * (1 - 1e-6)
        lo, hi = max(x0 - w, floor), x0 + w
        while F(hi) > 0:
            hi += 2 * w
        while F(lo) < 0 and lo > floor:
            lo = max(lo - 2 * w, floor)
        xs.append(brentq(F, lo, hi, xtol=1e-11) if F(lo) >= 0 else np.nan)
    xs = np.array(xs)
    lin = (gS - 1) * S / M1
    near = (np.abs(gS - 1) <= 0.1 + 1e-12) & (np.abs(gS - 1) > 1e-9)
    lin_err = float(np.max(np.abs(xs[near] - lin[near]) / np.abs(lin[near])))
    return dict(gS=gS, x=xs, S=S, M1=M1, gap=gap, lin_err=lin_err)


# --------------------------------------------------------------------------- Figure 2, panel D
GHF_PAR = dict(lam=1.0, nu=0.5, zeta=3.0, ell=0.7, c=1.5, L=12.0, h=0.02)


def ghf_rules():
    nu, z, l = GHF_PAR["nu"], GHF_PAR["zeta"], GHF_PAR["ell"]
    return {"up": lambda u: nu + z * (1 - np.exp(-u ** 2 / (2 * l ** 2))),
            "down": lambda u: nu + z * np.exp(-u ** 2 / (2 * l ** 2))}


def bifurcation():
    """Branches of stationary states m = M(g c tanh(m/c)) for two generalised hazard functions."""
    c = GHF_PAR["c"]
    out = {}
    fs = np.concatenate([np.geomspace(1e-3, 0.5, 30), np.linspace(0.52, 9.0, 120)])
    ts = np.linspace(0.0, 12.0, 241)
    for name, Lam in ghf_rules().items():
        m = GHF(Lam, lam=GHF_PAR["lam"], L=GHF_PAR["L"], h=GHF_PAR["h"])
        S = m.S_tilted(0.0)
        Ms = np.array([m.M(f) for f in fs])
        dM = np.array([(m.M(f + 1e-5) - m.M(f - 1e-5)) / 2e-5 for f in fs])
        g = fs / (c * np.tanh(Ms / c))
        slope = dM * g / np.cosh(Ms / c) ** 2
        np.savetxt(DATA / f"tip_branch_{name}.csv", np.column_stack([g * S, Ms, slope]), delimiter=",",
                   header="gS,m,slope", comments="", fmt="%.7f")
        # checks at stable branch points: kernel nonnegative and its integral equal to M'(f)
        stable = np.flatnonzero(slope < 1)
        pick = stable[np.linspace(0, len(stable) - 1, 6).round().astype(int)] if len(stable) else []
        kmin, intrel = np.inf, 0.0
        mc = GHF(Lam, lam=GHF_PAR["lam"], L=GHF_PAR["L"], h=0.04)
        for i in pick:
            k = mc.kernel(fs[i], ts)
            kmin = min(kmin, float(k.min()))
            intrel = max(intrel, abs(m.S_tilted(fs[i]) - dM[i]) / dM[i])
        kmin = min(kmin, float(mc.kernel(0.0, ts).min()))
        sel = (fs > 0.04) & (fs < 0.31)
        M3 = 6 * np.linalg.lstsq(np.vstack([fs[sel], fs[sel] ** 3, fs[sel] ** 5]).T, Ms[sel], rcond=None)[0][1]
        out[name] = dict(S=S, M3=float(M3), fold_gS=float((g * S).min()), kmin=kmin, int_rel=intrel,
                         slope_min=float(slope.min()), slope_max=float(slope.max()))
        if name == "down":
            i = int(np.argmin(g)); upper = np.arange(len(fs)) > i
            out[name]["fold_m"] = float(Ms[i])
            out[name]["jump_m"] = float(np.interp(1.0, (g * S)[upper], Ms[upper]))
    calvo = GHF(lambda u: GHF_PAR["nu"] + 0 * u, lam=GHF_PAR["lam"], L=GHF_PAR["L"], h=GHF_PAR["h"])
    out["calvo_rel_err"] = max(abs(calvo.M(f) * (GHF_PAR["lam"] + GHF_PAR["nu"]) / f - 1) for f in (0.1, 1.0, 4.0))
    REC["bifurcation"] = out
    return out


# --------------------------------------------------------------------------- shape in the running economy
def _fit_M3(fs, Ms):
    c = np.linalg.lstsq(np.vstack([fs, fs ** 3, fs ** 5]).T, Ms, rcond=None)[0]
    return float(c[0]), float(6 * c[1])


def shape(lam, theta=THETA, eta=ETA, nu=NU, Nbs=(100, 200), refs=(1, 2, 4)):
    """Third derivative of M at zero for B (time-integrated solve) and for P (Richardson in dt),
    extrapolated in Nb; at lam = 0 also the cross-section and record formulas of Proposition 5."""
    from c9run import rich_t, rich_b
    fs = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    per = {}
    for Nb in Nbs:
        L = default_L(lam)
        m = c9core.Model(lam=lam, theta=theta, eta=eta, nu=nu, L=L, Nu=odd(2 * L / 0.02), Nb=Nb)
        MB = []
        for f in fs:
            r = c9core.resolvent_B(m, f, nf=1); MB.append(r["EI"] / r["Etau"])
        SB, M3B = _fit_M3(fs, np.array(MB))
        vals = {"M3P": [], "SP": [], "cross": [], "record": []}
        for ref in refs:
            ts = time_grid(T=40.0 / nu if lam > 0 else 55.0 / nu, dt_max=0.04, dt_min=1e-3, growth=1.1, refine=ref)
            rb = ts_B(m, ts, nf=1, store=("p", "c"), mass_tol=1e-17)
            ts = rb["ts"]; hb, _ = make_hbar(m, rb["store"]["p"], rb["store"]["c"])
            MP = []
            for f in fs:
                rp = ts_P(m, ts, hb, f, nf=1); MP.append(rp["EI"] / rp["Etau"])
            SP, M3P = _fit_M3(fs, np.array(MP))
            vals["M3P"].append(M3P); vals["SP"].append(SP)
            if lam == 0:
                p, cc, u = rb["store"]["p"], rb["store"]["c"], m.u
                tt = ts[1:len(p) + 1]; dt = np.diff(ts)[:len(tt)]
                Et = (dt * p.sum(1)).sum()
                V = (dt * (p * u ** 2).sum(1)).sum() / Et
                EU4 = (dt * (p * u ** 4).sum(1)).sum() / Et
                EaU2 = (dt * tt * (p * u ** 2).sum(1)).sum() / Et
                Ea = (dt * tt * p.sum(1)).sum() / Et
                vals["cross"].append(EU4 - 3 * EaU2 - 3 * V * (V - Ea))
                ms = (dt * cc.sum(1)).sum()
                Ed6 = (dt * (cc * u ** 6).sum(1)).sum() / ms; Ed4 = (dt * (cc * u ** 4).sum(1)).sum() / ms
                Etd4 = (dt * tt * (cc * u ** 4).sum(1)).sum() / ms; Et2 = (dt * tt ** 2 * cc.sum(1)).sum() / ms
                Etr = (dt * tt * cc.sum(1)).sum() / ms
                Sr = Ed4 / (6 * Etr)
                vals["record"].append((Ed6 - 5 * Etd4) / (10 * Etr) - 3 * Sr * (Sr - Et2 / (2 * Etr)))
        R = {k: rich_t(v, list(refs))["R3"] for k, v in vals.items() if v}
        per[Nb] = dict(SB=SB, M3B=M3B, **R)
    n1, n2 = Nbs
    ex = {k: rich_b(per[n1][k], per[n2][k]) for k in per[n2]}
    ex["raw"] = per
    return ex


def tipping():
    rec = {"bifurcation": bifurcation()}
    rec["shape_run"] = shape(LAM)
    rec["shape_lam0"] = shape(0.0)
    # S Lbar of the running economy from the episode solver (base resolution), for the solver comparison
    from importlib import util
    spec = util.spec_from_file_location("speed_grid", str(HERE / "checks" / "speed_grid.py"))
    sg = util.module_from_spec(spec); spec.loader.exec_module(sg)
    L = default_L(LAM)
    m = c9core.Model(lam=LAM, theta=THETA, eta=ETA, nu=NU, L=L, Nu=odd(2 * L / 0.02), Nb=200)
    rec["M1B_base"] = sg.S_M1(sg.resolvent_moments(m), LAM)[1]
    REC["tipping"] = rec
    return rec


# --------------------------------------------------------------------------- Figure 4: stability
def stability(SB_ref):
    res = {}
    kern, nyq, rates = {}, {}, {}
    for eta in (0.0, 1.0, 5.0, 20.0):
        p = Params(theta=THETA, eta=eta, nu=NU, lam=LAM, gam=GAM, reset='0')
        mod = FVModel(p, default_grid(p, h=0.02, Na_f=160))
        S = mod.S_resolvent()
        sh0 = Shifted(mod.odd, 0.0)
        x1 = sh0.solve(-mod.b); x2 = sh0.solve(-x1)
        M1 = float(mod.odd.inner_u(x2))                      # int t k(t) dt = <u, L^{-2} b>
        gap = slowest_rate(mod)
        ks, (t, k) = kernel_stats(mod, gap=gap)
        rates[eta] = dominant_root_curve(mod, S, M1, gap=gap)
        fc = first_crossing(mod, w_max=80.0)
        ws, vals = fc["curve"]
        kern[eta] = (t, k, S); nyq[eta] = (ws, vals, S)
        res[f"eta{eta:g}"] = dict(S=S, M1=M1, Lbar=M1 / S, gap=rates[eta]["gap"], lin_err=rates[eta]["lin_err"],
                                 kmin=ks["kmin"], first=fc["type"], g_first=fc["g"],
                                 crossings=[(float(a), float(b)) for (a, b) in fc["crossings"]],
                                 min_re=float(np.min(vals.real) / S))
        if eta == ETA:
            mod_run, ks_run, S_run = mod, ks, S
    # kernels on a common time grid, normalised by S (area one)
    tg = np.linspace(0, 6, 241)
    cols = [tg] + [np.interp(tg, kern[e][0], kern[e][1]) for e in (0.0, 1.0, 5.0, 20.0)]
    np.savetxt(DATA / "stab_kernels.csv", np.column_stack(cols), delimiter=",",
               header="t,k0,k1,k5,k20", comments="", fmt="%.7f")
    for e in (0.0, 1.0, 5.0, 20.0):
        ws, vals, S = nyq[e]
        ok = ws <= 60
        np.savetxt(DATA / f"stab_nyquist_eta{e:g}.csv", np.column_stack([vals.real[ok] / S, vals.imag[ok] / S]),
                   delimiter=",", header="re,im", comments="", fmt="%.7f")
    gs = rates[0.0]["gS"]
    cols = [gs] + [rates[e]["x"] for e in (0.0, 1.0, 5.0, 20.0)] + \
           [(gs - 1) / (rates[e]["M1"] / rates[e]["S"]) for e in (0.0, 1.0, 5.0, 20.0)]
    np.savetxt(DATA / "tip_rate.csv", np.column_stack(cols), delimiter=",",
               header="gS,x0,x1,x5,x20,l0,l1,l5,l20", comments="", fmt="%.7f")
    # certificates for the running economy: no root in Re z > 0 below g = 1/S, one above
    certs = {}
    for fac in (0.98, 1.02):
        c = certify_count(mod_run, fac / S_run, ks_run)
        certs[str(fac)] = dict(N=c["N"], min_absF=c["min_absF"])
    g = json.load(open(DATA / "grid" / "grid_lam1.0.json"))
    run = [r for r in g if r["theta"] == THETA and r["eta"] == ETA and abs(r["nu"] - NU) < 1e-12][0]
    gstar = 0.5 * (1 / run["SB_fd"] + 1 / run["SP_fd"])
    c = certify_count(mod_run, gstar, ks_run)
    certs["gstar"] = dict(g=gstar, N=c["N"], min_absF=c["min_absF"], gS_fv=gstar * S_run)
    res["certificates"] = certs
    res["S_run_fv"] = S_run
    REC["stability"] = res
    return res


def scan_summary():
    rows = [json.loads(l) for l in open(DATA / "grid" / "scan_main.jsonl")] + \
           [json.loads(l) for l in open(DATA / "grid" / "scan_extreme.jsonl")]
    REC["scan"] = dict(n=len(rows), n_real=sum(r["fc_type"] == "real" for r in rows),
                       min_kmin=min(r["kmin"] for r in rows),
                       n_pos_axis_crossings=sum(any(w > 0 and x > 0 for (w, x) in r["crossings"]) for r in rows))
    uni = json.load(open(DATA / "grid" / "unimodality.json"))
    REC["unimodal"] = dict(n=len(uni), n_ok=sum(u["worst_rel_increase"] <= 1e-10 for u in uni))
    return REC["scan"]


def tex_sci(x):
    """3.3e-15 -> 3\\times10^{-15} (one significant digit)."""
    m, e = f"{x:.0e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def write_values():
    g = json.load(open(DATA / "grid" / "grid_lam1.0.json"))
    run = [r for r in g if r["theta"] == THETA and r["eta"] == ETA and abs(r["nu"] - NU) < 1e-12][0]
    SB, SP = run["SB_fd"], run["SP_fd"]
    VarU, F, ALO = run["VarU"], run["F"], run["ALO"]
    gstar = 0.5 * (1 / SB + 1 / SP)
    REC["running"] = dict(SB=SB, SP=SP, VarU=VarU, F=F, ALO=ALO, gstar=gstar, gSB=gstar * SB,
                          gSP=gstar * SP, critB=1 / SB, critP=1 / SP, critALO=1 / ALO,
                          SB_formula=run["SB_formula"], SP_formula=run["SP_formula"])
    mag = REC["magnitudes"]; st = REC["stability"]; ep = REC["episode"]; rc = REC["record"]
    sc = REC["scan"]; un = REC["unimodal"]
    v = {
        "RunLam": f"{LAM:g}", "RunTheta": f"{THETA:g}", "RunEta": f"{ETA:g}", "RunNu": f"{NU:g}",
        "RunSB": f"{SB:.4f}", "RunSP": f"{SP:.4f}", "RunVar": f"{VarU:.4f}", "RunF": f"{F:.3f}",
        "RunALO": f"{ALO:.4f}", "RunCritB": f"{1/SB:.3f}", "RunCritP": f"{1/SP:.3f}",
        "RunCritALO": f"{1/ALO:.3f}", "RunGstar": f"{gstar:.3f}", "RunGSB": f"{gstar*SB:.4f}",
        "RunGSP": f"{gstar*SP:.4f}", "RunGapPct": f"{100*(SB-SP)/SB:.2f}", "RunGapPctAbs": f"{100*abs(SB-SP)/SB:.2f}",
        "RunALOPct": f"{100*(ALO-SB)/SB:.0f}", "RunTstar": f"{rc['tstar']:.1f}",
        "RunDurDiff": tex_sci(max(rc['dur_maxdiff'], rc['rev_maxdiff'])), "RunEItwoB": f"{rc['EI2_B']:.4f}", "RunEItwoP": f"{rc['EI2_P']:.4f}",
        "RunSFV": f"{st['S_run_fv']:.4f}",
        "MagClockQGapMin": f"{abs(mag['0.25']['gap_max']):.2f}", "MagClockQGapMax": f"{abs(mag['0.25']['gap_min']):.2f}",
        "MagClockOneGapMin": f"{abs(mag['1.0']['gap_max']):.2f}", "MagClockOneGapMax": f"{abs(mag['1.0']['gap_min']):.2f}",
        "MagClockTwoGapMin": f"{abs(mag['2.0']['gap_max']):.3f}", "MagClockTwoGapMax": f"{abs(mag['2.0']['gap_min']):.2f}",
        "MagKerrOneMin": f"{mag['1.0']['kerr_min']:.0f}", "MagKerrOneMax": f"{mag['1.0']['kerr_max']:.0f}",
        "MagKerrTwoMin": f"{mag['2.0']['kerr_min']:.0f}", "MagKerrTwoMax": f"{mag['2.0']['kerr_max']:.0f}",
        "MagKerrQMin": f"{mag['0.25']['kerr_min']:.0f}", "MagKerrQMax": f"{mag['0.25']['kerr_max']:.0f}",
        "MagMaxGap": f"{mag['all_lam_pos_max_abs_gap']:.1f}", "MagN": f"{mag['n_total']}",
        "MagOtherGapMax": f"{abs(mag['other']['gap_min']):.1f}",
        "ScanN": f"{sc['n']}", "ScanReal": f"{sc['n_real']}", "UniN": f"{un['n']}", "UniOK": f"{un['n_ok']}",
        "EpTau": f"{ep['EpTc']:g}",
        "CertN": f"{abs(int(round(st['certificates']['gstar']['N']))):d}", "CertMinF": f"{st['certificates']['gstar']['min_absF']:.3f}",
        "CertNlo": f"{abs(int(round(st['certificates']['0.98']['N']))):d}", "CertNhi": f"{abs(int(round(st['certificates']['1.02']['N']))):d}",
        "StabSzero": f"{st['eta0']['S']:.4f}", "StabSone": f"{st['eta1']['S']:.4f}", "StabSfive": f"{st['eta5']['S']:.4f}",
        "StabStwenty": f"{st['eta20']['S']:.4f}",
    }
    lines = ["% Generated by code/make_data.py; do not edit."] + [r"\def\%s{%s}" % (k, s) for k, s in v.items()]
    for k, val in ep.items():
        lines.append(r"\def\%s{%.6f}" % (k, val))
    (TIKZ / "values.tex").write_text("\n".join(lines) + "\n")
    json.dump(REC, open(DATA / "records.json", "w"), indent=1, default=float)


def speed_magnitudes():
    """Figure 3, panel C, and the speed macros, from data/grid/speed_grid.json (code/checks/speed_grid.py)."""
    fn = DATA / "grid" / "speed_grid.json"
    rows = json.load(open(fn))
    rng = np.random.default_rng(5)
    xpos = {0.0: 0, 0.25: 1, 1.0: 2, 2.0: 3}
    out = np.array([(xpos[r["lam"]] - 0.12 + 0.1 * (rng.random() - 0.5), r["lam"], r["gapL_pct"]) for r in rows])
    np.savetxt(DATA / "magnitudes_lag.csv", out, delimiter=",", header="x,lam,gapL", comments="", fmt="%.6f")
    a = np.abs(out[:, 2]); lam = out[:, 1]
    summ = dict(n=len(rows), zero=(float(a[lam == 0].min()), float(a[lam == 0].max())),
                pos=(float(a[lam > 0].min()), float(a[lam > 0].max())), all=(float(a.min()), float(a.max())),
                all_negative=bool((out[:, 2] < 0).all()),
                relerr=float(max(r["gapM1_relerr"] for r in rows)))
    run = [r for r in rows if r["lam"] == LAM and r["theta"] == THETA and r["eta"] == ETA and abs(r["nu"] - NU) < 1e-12][0]
    summ["run"] = dict(LB=run["LB"], LP=run["LP"], gapL_pct=run["gapL_pct"], SB=run["SB"], SP=run["SP"])
    REC["speed"] = summ
    return summ


def write_tipping_values():
    st, bf, tp = REC["stability"], REC["bifurcation"], REC["tipping"]
    sp = REC.get("speed")
    if sp is None:          # provisional build while code/checks/speed_grid.py is still running
        sp = dict(run=dict(LB=float("nan"), LP=float("nan"), gapL_pct=float("nan")), zero=(float("nan"),) * 2,
                  pos=(float("nan"),) * 2, all=(float("nan"),) * 2, relerr=float("nan"))
    run = REC["running"]
    sr, s0 = tp["shape_run"], tp["shape_lam0"]
    M3B, M3P = sr["M3B"], sr["M3P"]
    amp = 100 * (np.sqrt((run["SP"] / run["SB"]) ** 3 * M3B / M3P) - 1)
    lam_nu = LAM + NU
    calvo_lag = max(abs(st["eta0"]["Lbar"] * lam_nu - 1), st["eta0"]["lin_err"])
    kneg = max(0.0, -min(bf["up"]["kmin"], bf["down"]["kmin"]))
    v = {
        "TipLbarZero": f"{st['eta0']['Lbar']:.3f}", "TipLbarOne": f"{st['eta1']['Lbar']:.3f}",
        "TipLbarFive": f"{st['eta5']['Lbar']:.3f}", "TipLbarTwenty": f"{st['eta20']['Lbar']:.3f}",
        "TipLinErr": f"{100 * max(st[k]['lin_err'] for k in ('eta0', 'eta1', 'eta5', 'eta20')):.1f}",
        "TipC": f"{GHF_PAR['c']:g}", "TipNu": f"{GHF_PAR['nu']:g}", "TipZeta": f"{GHF_PAR['zeta']:g}",
        "TipEll": f"{GHF_PAR['ell']:g}", "TipDu": f"{GHF_PAR['h']:g}",
        "TipDownFold": f"{bf['down']['fold_gS']:.2f}", "TipUpMthree": f"{bf['up']['M3']:.3f}",
        "TipDownFoldM": f"{bf['down']['fold_m']:.3f}", "TipDownJumpM": f"{bf['down']['jump_m']:.3f}",
        "TipDownMthree": f"{bf['down']['M3']:.3f}",
        "TipKminTol": tex_sci(max(kneg, 1e-16)),
        "TipKintRel": tex_sci(max(bf["up"]["int_rel"], bf["down"]["int_rel"], 1e-16)),
        "TipCalvoErr": tex_sci(max(bf["calvo_rel_err"], 1e-16)), "TipCalvoLagErr": tex_sci(max(calvo_lag, 1e-16)),
        "RunLbarB": f"{sp['run']['LB']:.4f}", "RunLbarP": f"{sp['run']['LP']:.4f}",
        "RunLbarGapPct": f"{abs(sp['run']['gapL_pct']):.1f}",
        "RunMthreeB": f"{M3B:.3f}", "RunMthreeP": f"{M3P:.3f}", "RunMthreeGapPct": f"{100 * abs(M3B - M3P) / abs(M3B):.0f}",
        "RunAmpGapPct": f"{amp:.0f}",
        "RunMoneB": f"{tp['M1B_base']:.4f}", "RunMoneFV": f"{st['eta5']['M1']:.4f}",
        "ValShapeB": f"{s0['M3B']:.4f}", "ValShapeP": f"{s0['M3P']:.4f}", "ValShapeCross": f"{s0['cross']:.4f}",
        "ValShapeRec": f"{s0['record']:.4f}",
        "MagLagZeroMin": f"{sp['zero'][0]:.1f}", "MagLagZeroMax": f"{sp['zero'][1]:.1f}",
        "MagLagPosMin": f"{sp['pos'][0]:.2f}", "MagLagPosMax": f"{sp['pos'][1]:.1f}",
        "MagLagAllMin": f"{sp['all'][0]:.2f}", "MagLagAllMax": f"{sp['all'][1]:.1f}",
        "GridLagRelErr": f"{100 * sp['relerr']:.1f}",
    }
    lines = ["% Generated by code/make_data.py; do not edit."] + [r"\def\%s{%s}" % (k, s) for k, s in v.items()]
    (TIKZ / "tipping_values.tex").write_text("\n".join(lines) + "\n")
    json.dump(REC, open(DATA / "records.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    t0 = time.time()
    episode(); print("episode", time.time() - t0, flush=True)
    magnitudes(); print("magnitudes", time.time() - t0, flush=True)
    scan_summary(); print("scan", time.time() - t0, flush=True)
    record(); print("record", REC["record"], time.time() - t0, flush=True)
    stability(None); print("stability", REC["stability"], time.time() - t0, flush=True)
    tipping(); print("tipping", time.time() - t0, flush=True)
    if os.environ.get("SKIP_SPEED") != "1":
        speed_magnitudes(); print("speed", REC["speed"], time.time() - t0, flush=True)
    write_values(); write_tipping_values(); print("done", time.time() - t0)

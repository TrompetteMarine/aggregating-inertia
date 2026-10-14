"""c9 driver: one parameter point -> S_B, S_P (two ways each), renewal statistics, claim-1 checks.

S_B : resolvent (time-integrated) episode solve of B on the (u,b) grid; exact in time.
      (i) central finite difference of M(f) = E[I_tau]/E[tau] at f = +-delta,
      (ii) formula (E[I U_-] + lam E[I^2]) / (sig^2 E tau) with I-moment densities.
S_P : B is time-stepped (backward Euler) to tabulate hbar = c/p on the time grid; P is then
      time-stepped with hbar held fixed,
      (i) finite difference of M_P(f) at f = +-delta, (ii) the formula with 1D I-moment densities.
      Repeated with every time step halved (refine = 1, 2, 4); Richardson extrapolation in dt.
Error model:  X(du, Nb, dt) = X + a du^2 + b / Nb + c dt + d dt^2 + ...
"""
import time
import numpy as np
from c9core import (Model, resolvent_B, SB_fd, ts_B, ts_P, SP_fd, make_hbar, stats_from, time_grid,
                    default_L, odd)

NU6 = 0.7136347625525
DELTA = 0.002          # finite-difference half-step in f (truncation error O(delta^2) ~ 1e-7)


def rich_t(vals, refs):
    """Richardson in dt for first-order BE with refinement factors refs (e.g. 1,2,4)."""
    v = dict(zip(refs, vals))
    out = {}
    if len(refs) >= 2:
        a, b = refs[-2], refs[-1]
        out["R2"] = 2 * v[b] - v[a]
    if len(refs) >= 3 and tuple(refs[-3:]) == (refs[-3], 2 * refs[-3], 4 * refs[-3]):
        a, b, c = refs[-3:]
        out["R3"] = (8 * v[c] - 6 * v[b] + v[a]) / 3
    return out


def rich_b(x_coarse, x_fine):          # first order in 1/Nb, fine = 2 x coarse
    return 2 * x_fine - x_coarse


def rich_u(x1, h1, x2, h2):            # second order in du
    return (h1 ** 2 * x2 - h2 ** 2 * x1) / (h1 ** 2 - h2 ** 2)


def resolvent_point(lam, theta, eta, nu, du, Nb, L=None, delta=DELTA):
    L = default_L(lam) if L is None else L
    m = Model(lam=lam, theta=theta, eta=eta, nu=nu, L=L, Nu=odd(2 * L / du), Nb=Nb)
    r = resolvent_B(m)
    s = stats_from(m, r)
    sfd = SB_fd(m, delta)[0]
    return dict(du=m.du, Nb=Nb, SB_formula=s["S_formula"], SB_fd=sfd, Etau=s["Etau"], F=s["F"], VarU=s["VarU"],
                VarU_formula=s["VarU_formula"], Kurt=s["Kurt"], ALO=s["ALO"], EI2_B=r["EI2"], EIU_B=r["EIU"],
                EI2alt_B=r["EI2_alt"], EU2int=r["EU2int"], mass_out=r["mass_out"])


def run_point(lam, theta, eta, nu, Nu=None, Nb=200, L=None, du=0.02, dtmax=0.04, dtmin=1e-3, growth=1.1,
              refs=(1, 2, 4), T=None, delta=DELTA, mass_tol=1e-17, verbose=False, keep=False, nfB=1):
    L = default_L(lam) if L is None else L
    Nu = odd(2 * L / du) if Nu is None else Nu
    T = (40.0 / nu if lam > 0 else 55.0 / nu) if T is None else T
    m = Model(lam=lam, theta=theta, eta=eta, nu=nu, L=L, Nu=Nu, Nb=Nb)
    out = dict(lam=lam, theta=theta, eta=eta, nu=nu, L=L, Nu=Nu, Nb=Nb, du=m.du, dtmax=dtmax, dtmin=dtmin,
               growth=growth, delta=delta, refs=list(refs))
    t0 = time.time()
    rB = resolvent_B(m)
    sB = stats_from(m, rB)
    SBfd, Mp, Mm = SB_fd(m, delta)
    out.update(SB_formula=sB["S_formula"], SB_fd=SBfd, Etau=sB["Etau"], F=sB["F"], VarU=sB["VarU"],
               VarU_formula=sB["VarU_formula"], Kurt=sB["Kurt"], ALO=sB["ALO"], Ed2=sB["Ed2"],
               EIU_B=rB["EIU"], EI2_B=rB["EI2"], EI2alt_B=rB["EI2_alt"], EU2int=rB["EU2int"],
               mass_out_B=rB["mass_out"])
    runs = []
    kept = {}
    for ref in refs:
        t1 = time.time()
        ts = time_grid(T=T, dt_max=dtmax, dt_min=dtmin, growth=growth, refine=ref)
        rb = ts_B(m, ts, nf=nfB, store=("p", "c"), mass_tol=mass_tol)
        hb, nbad = make_hbar(m, rb["store"]["p"], rb["store"]["c"])
        rp = ts_P(m, rb["ts"], hb, nf=3, compare=rb["store"])
        sp = stats_from(m, rp)
        SPfd, MPp, MPm = SP_fd(m, rb["ts"], hb, delta)
        pmax = rb["store"]["p"].max()
        d = dict(ref=ref, nsteps=rb["nsteps"], Tend=float(rb["ts"][-1]), mass_left=float(rb["mass_left"]),
                 nbad=nbad, SP_formula=sp["S_formula"], SP_fd=SPfd, EIU_P=rp["EIU"], EI2_P=rp["EI2"],
                 EI2alt_P=rp["EI2_alt"], Etau_P=sp["Etau"], VarU_P=sp["VarU"], Kurt_P=sp["Kurt"],
                 Etau_Bts=float(rb["Etau"]),
                 # discrete claim-1 identity: P with hbar = c/p reproduces B's marginal quantities
                 maxdiff_p=float(rp["maxdiff"]["p"]), maxdiff_c=float(rp["maxdiff"]["c"]), pmax=float(pmax),
                 occ_maxdiff=float(np.abs(rp["occ"] - rb["occ"]).max()),
                 rev_maxdiff=float(np.abs(rp["rev"] - rb["rev"]).max()),
                 surv_maxdiff=float(np.abs(rp["surv"] - rb["surv"]).max()),
                 dens_maxdiff=float(np.abs(rp["dens"] - rb["dens"]).max()),
                 occ_vs_resolvent=float(np.abs(rb["occ"] - rB["occ"]).max()),
                 secs=time.time() - t1)
        if nfB == 3:
            d["SB_formula_ts"] = stats_from(m, rb)["S_formula"]
        runs.append(d)
        if keep:
            kept[ref] = dict(rb=rb, hb=hb, rp=rp)
        if verbose:
            print(f"    Nb={Nb} ref={ref}: steps={d['nsteps']} S_P form={d['SP_formula']:.7f} fd={d['SP_fd']:.7f} "
                  f"({d['secs']:.1f}s)", flush=True)
    out["runs"] = runs
    for key in ("SP_formula", "SP_fd", "EI2_P", "EIU_P"):
        vals = [r[key] for r in runs]
        R = rich_t(vals, list(refs))
        out[key + "_raw"] = vals[-1]
        out[key] = R.get("R3", R.get("R2", vals[-1]))
        out[key + "_R2"] = R.get("R2", np.nan)
        out[key + "_dterr"] = abs(R["R3"] - R["R2"]) if "R3" in R else (abs(R["R2"] - vals[-1]) if "R2" in R else np.nan)
    out["gap_formula"] = out["SB_formula"] - out["SP_formula"]
    out["gap_fd"] = out["SB_fd"] - out["SP_fd"]
    out["gap_EI2"] = lam * (out["EI2_B"] - out["EI2_P"]) / (2 * m.sig ** 2 * out["Etau"])
    out["secs"] = time.time() - t0
    if keep:
        out["_kept"] = kept
        out["_model"] = m
    return out


def full_point(lam, theta, eta, nu, du=0.02, Nbs=(100, 200), refs=(1, 2, 4), fine_du=(0.02, 0.01),
               fine_Nb=(800, 1600), verbose=False, **kw):
    """Gap from time-stepped runs at (du, Nb in Nbs) extrapolated in dt and Nb; S_B from fine resolvent grids
    extrapolated in du and Nb; S_P = S_B - gap (separately for the formula and the FD versions)."""
    t0 = time.time()
    pts = [run_point(lam, theta, eta, nu, du=du, Nb=Nb, refs=refs, verbose=verbose, **kw) for Nb in Nbs]
    res = {}
    for (h, Nb) in [(h, Nb) for h in fine_du for Nb in fine_Nb]:
        res[(h, Nb)] = resolvent_point(lam, theta, eta, nu, h, Nb)
    out = dict(lam=lam, theta=theta, eta=eta, nu=nu)
    h1, h2 = fine_du
    n1, n2 = fine_Nb
    for key in ("SB_formula", "SB_fd", "Etau", "F", "VarU", "VarU_formula", "Kurt", "ALO", "EI2_B", "EIU_B"):
        x1 = rich_b(res[(h1, n1)][key], res[(h1, n2)][key])
        x2 = rich_b(res[(h2, n1)][key], res[(h2, n2)][key])
        hh1, hh2 = res[(h1, n2)]["du"], res[(h2, n2)]["du"]
        out[key] = rich_u(x1, hh1, x2, hh2) if not np.isnan(x1) else np.nan
        # error indicators: size of the last du and Nb corrections
        out[key + "_err_u"] = abs(out[key] - x2) if not np.isnan(x1) else np.nan
        out[key + "_err_b"] = abs(x2 - res[(h2, n2)][key]) if not np.isnan(x1) else np.nan
    for which in ("formula", "fd"):
        g = [p["gap_" + which] for p in pts]
        out["gap_" + which + "_byNb"] = g
        out["gap_" + which] = rich_b(g[-2], g[-1]) if len(g) >= 2 else g[-1]
        out["gap_" + which + "_err_b"] = abs(out["gap_" + which] - g[-1])
        out["gap_" + which + "_err_t"] = pts[-1]["SP_" + which + "_dterr"]
        out["SP_" + which] = out["SB_" + which] - out["gap_" + which]
    eg = [p["gap_EI2"] for p in pts]
    out["gap_EI2"] = rich_b(eg[-2], eg[-1]) if len(eg) >= 2 else eg[-1]
    out["EI2_P"] = out["EI2_B"] - (rich_b(pts[-2]["EI2_B"] - pts[-2]["EI2_P"], pts[-1]["EI2_B"] - pts[-1]["EI2_P"])
                                   if len(pts) >= 2 else pts[-1]["EI2_B"] - pts[-1]["EI2_P"])
    out["claim1_maxdiff_p"] = max(r["maxdiff_p"] for p in pts for r in p["runs"])
    out["claim1_maxdiff_c"] = max(r["maxdiff_c"] for p in pts for r in p["runs"])
    out["claim1_occ_maxdiff"] = max(r["occ_maxdiff"] for p in pts for r in p["runs"])
    out["claim1_rev_maxdiff"] = max(r["rev_maxdiff"] for p in pts for r in p["runs"])
    out["claim1_surv_maxdiff"] = max(r["surv_maxdiff"] for p in pts for r in p["runs"])
    out["claim1_dens_maxdiff"] = max(r["dens_maxdiff"] for p in pts for r in p["runs"])
    out["nbad_max"] = max(r["nbad"] for p in pts for r in p["runs"])
    out["raw_points"] = pts
    out["resolvent_grid"] = {f"du={k[0]},Nb={k[1]}": v for k, v in res.items()}
    out["secs"] = time.time() - t0
    return out

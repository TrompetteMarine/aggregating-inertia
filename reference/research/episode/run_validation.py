"""Validation in the theta = 0 limit (act = 1, hazard depends on age only, hbar = H, S_B = S_P).
Exact quadrature copied from /home/claude/research_notes/albl_check.py (not imported, not edited), extended with
E[I_tau U_tau] and E[I_tau^2] to check the I-moment equations.  Writes results/validation.json."""
import sys, os, json, time
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c9run import full_point, run_point, resolvent_point

# ---- copied from albl_check.py ---------------------------------------------------------------
GAM = 1.0
def surv(a, eta, nu):  return np.exp(-nu*a - eta*(a + np.expm1(-GAM*a)/GAM))
def haz(a, eta, nu):   return nu + eta*(-np.expm1(-GAM*a))
def g(a, r):           return a if r == 0 else -np.expm1(-r*a)/r
def I(f):              return quad(f, 0, np.inf, epsabs=1e-13, epsrel=1e-12, limit=400)[0]

def stats(eta, nu, lam, sig=1.0):
    mu   = I(lambda a: surv(a, eta, nu))                       # E[L]
    S    = I(lambda a: surv(a, eta, nu) * g(a, lam)) / mu       # dM/df
    V    = sig**2 * I(lambda a: surv(a, eta, nu) * g(a, 2*lam)) / mu
    phi  = lambda a: haz(a, eta, nu) * surv(a, eta, nu)          # duration density
    m2   = sig**2 * I(lambda a: phi(a) * g(a, 2*lam))            # E[Delta^2]
    m4   = 3 * sig**4 * I(lambda a: phi(a) * g(a, 2*lam)**2)     # E[Delta^4]
    kurt = m4 / m2**2
    if lam == 0:
        vint = lambda a: sig**2 * a**3 / 3
    else:
        vint = lambda a: sig**2 / lam**2 * (a - 2*g(a, lam) + g(a, 2*lam))
    eint2 = I(lambda a: phi(a) * vint(a))
    girs  = V / sig**2 + (lam / (2 * sig**2)) * eint2 / mu
    # added for c9: E[I_tau U_tau-] (Cov of int U and U for OU from 0)
    if lam == 0:
        ciu = lambda a: sig**2 * a**2 / 2
    else:
        ciu = lambda a: sig**2 / lam * (g(a, lam) - g(a, 2*lam))
    eiu = I(lambda a: phi(a) * ciu(a))
    return dict(S=S, V=V, F=1/mu, kurt=kurt, albl=kurt * mu / 6, girsanov=girs, EI2=eint2, EIU=eiu, Etau=mu,
                S_formula=(eiu + lam * eint2) / (sig**2 * mu))

def matched_nu(eta, nu0=1.0):
    return nu0 if eta == 0 else brentq(lambda n: I(lambda a: surv(a, eta, n)) - 1/nu0, 1e-4, nu0, xtol=1e-14)
# ----------------------------------------------------------------------------------------------

if __name__ == "__main__":
    out = []
    t0 = time.time()
    for lam in (0.0, 0.25, 1.0):
        for eta in (0.0, 0.6, 2.0):
            nu = matched_nu(eta)
            ex = stats(eta, nu, lam)
            fp = full_point(lam, 0.0, eta, nu)
            row = dict(lam=lam, eta=eta, nu=nu, exact=ex,
                       num={k: float(fp[k]) for k in ("SB_formula", "SB_fd", "SP_formula", "SP_fd", "gap_formula",
                                                      "gap_fd", "VarU", "F", "Kurt", "ALO", "EI2_B", "EIU_B", "Etau")},
                       err={k: float(fp[k + "_err_u"] + fp[k + "_err_b"]) for k in ("SB_formula", "SB_fd", "VarU",
                                                                                    "F", "ALO", "Kurt")})
            out.append(row)
            n = row["num"]
            print(f"lam={lam:4.2f} eta={eta:3.1f} nu={nu:.4f} | S exact={ex['S']:.6f} S_B fd={n['SB_fd']:.6f} "
                  f"form={n['SB_formula']:.6f} S_P fd={n['SP_fd']:.6f} form={n['SP_formula']:.6f} | "
                  f"Var {ex['V']:.6f}/{n['VarU']:.6f} Kurt/6F {ex['albl']:.6f}/{n['ALO']:.6f} "
                  f"EI2 {ex['EI2']:.6f}/{n['EI2_B']:.6f} EIU {ex['EIU']:.6f}/{n['EIU_B']:.6f} F {ex['F']:.6f}/{n['F']:.6f}",
                  flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "validation.json"), "w"),
              indent=1, default=float)
    print(f"done {time.time()-t0:.0f}s")

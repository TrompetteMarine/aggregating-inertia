"""Verbatim copy of matched_nu (and its helpers) from /home/claude/research_notes/albl_check.py
(copied, not imported, because that module runs its table loop at import time; GAM = 1)."""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

GAM = 1.0
def surv(a, eta, nu):  return np.exp(-nu*a - eta*(a + np.expm1(-GAM*a)/GAM))
def I(f):              return quad(f, 0, np.inf, epsabs=1e-13, epsrel=1e-12, limit=400)[0]

def matched_nu(eta, nu0=1.0):
    return nu0 if eta == 0 else brentq(lambda n: I(lambda a: surv(a, eta, n)) - 1/nu0, 1e-4, nu0, xtol=1e-14)

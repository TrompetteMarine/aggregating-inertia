"""Spot checks of two identities used in the paper.

(1) Proposition 5(c): under an age-only hazard, S = E_pi[(1 - exp(-lambda a))/lambda].
    Compared with the exact renewal value stored in data/validation/episode_solver_controls.json
    (theta = 0, so the pausing-clock hazard is nu + eta (1 - exp(-gamma a))).
(2) Proposition 6: E_pi[U_T W_T]/sigma^2 = int_0^T k for the Calvo rule, where k(t) = exp(-(lambda+nu) t).
    Monte Carlo of the stationary population with exact OU steps.
Writes data/validation/identity_checks.json.
"""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[2]
out = {}

# (1) age-only formula
ctl = json.load(open(ROOT / "data" / "validation" / "episode_solver_controls.json"))
rows = []
for c in ctl:
    lam, eta, nu, gam = c["lam"], c["eta"], c["nu"], 1.0
    Lam = lambda a: nu * a + eta * (a - (1 - np.exp(-gam * a)) / gam)      # integrated hazard
    surv = lambda a: np.exp(-Lam(a))
    Etau = quad(surv, 0, np.inf, limit=400)[0]
    g = (lambda b: (1 - np.exp(-lam * b)) / lam) if lam > 0 else (lambda b: b)
    S_age = quad(lambda b: surv(b) * g(b), 0, np.inf, limit=400)[0] / Etau
    rows.append(dict(lam=lam, eta=eta, nu=nu, S_age_formula=S_age, S_exact=c["exact"]["S"],
                     abs_diff=abs(S_age - c["exact"]["S"])))
out["age_only"] = dict(rows=rows, max_abs_diff=max(r["abs_diff"] for r in rows))

# (2) fluctuation-response identity, Calvo rule
rng = np.random.default_rng(11)
lam, nu, sig, T, dt, N = 1.0, 1.0, 1.0, 3.0, 0.002, 400_000
# stationary start: age ~ Exp(nu), U = OU from 0 over that age (exact Gaussian)
a0 = rng.exponential(1 / nu, N)
U = rng.standard_normal(N) * sig * np.sqrt((1 - np.exp(-2 * lam * a0)) / (2 * lam))
W = np.zeros(N)
e, sd = np.exp(-lam * dt), sig * np.sqrt((1 - np.exp(-2 * lam * dt)) / (2 * lam))
for _ in range(int(round(T / dt))):
    # exact OU step driven by the same Gaussian increment: U' = e U + int e^{-lam(t-s)} sig dW_s;
    # use the joint Gaussian law of (OU increment, dW) over the step
    z1, z2 = rng.standard_normal(N), rng.standard_normal(N)
    cov = sig * (1 - e) / lam                       # Cov(int_0^dt e^{-lam(dt-s)} sig dW_s, W_dt)
    dW = np.sqrt(dt) * z1
    rho = cov / (sd * np.sqrt(dt))
    noise = sd * (rho * z1 + np.sqrt(max(1 - rho ** 2, 0.0)) * z2)
    U = e * U + noise
    W += dW
    reset = rng.random(N) < -np.expm1(-nu * dt)     # Calvo completions, full reset
    U[reset] = 0.0
est = U * W / sig
mc, se = est.mean(), est.std() / np.sqrt(N)
exact = (1 - np.exp(-(lam + nu) * T)) / (lam + nu)
out["fluct_response_calvo"] = dict(lam=lam, nu=nu, T=T, dt=dt, N=N, mc=float(mc), se=float(se), exact=float(exact))
json.dump(out, open(ROOT / "data" / "validation" / "identity_checks.json", "w"), indent=1)
print(json.dumps(out, indent=1))

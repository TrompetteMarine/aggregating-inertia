"""Check the first-order formula of Theorem 3(b) (Calvo base, full reset). Writes data/validation/perturbation.json.
Base: hazard h0 = nu. Perturbation: H_eps = nu + eps*psi, psi_s = -phi(X_s) + E0[phi(X_s) | U_s], X_s = I_s U_s/(lam+nu),
phi = tanh. Prediction: d/deps E_eps[I_tau^2] at 0 = 2 int_0^inf e^{-nu s} E[Cov(phi(X_s), X_s | U_s)] ds  (> 0),
using that under the base rule survival is independent of the path, so (U_s, I_s) is the free OU pair from 0.
Monte Carlo: common random numbers at +eps and -eps, exact OU stepping, thinning for completions."""
import numpy as np, time, json
from pathlib import Path
lam, nu, sig = 1.0, 1.0, 1.0
gh_x, gh_w = np.polynomial.hermite_e.hermegauss(40); gh_w = gh_w / gh_w.sum()
def g(a, r): return a if r == 0 else -np.expm1(-r*a)/r
def moments(s):
    vu = sig**2 * g(s, 2*lam); ciu = sig**2/lam*(g(s, lam) - g(s, 2*lam)); vi = sig**2/lam**2*(s - 2*g(s, lam) + g(s, 2*lam))
    return vu, ciu, vi
def cond_I(s, u):
    vu, ciu, vi = moments(s)
    c = ciu/vu if vu > 0 else 0.0; v = max(vi - ciu**2/vu, 0.0) if vu > 0 else vi
    return c*u, np.sqrt(v)
def Ephi_given_u(s, u):
    m, sd = cond_I(s, u)                      # arrays over paths
    I = m[:, None] + sd*gh_x[None, :]
    X = I*u[:, None]/(lam+nu)
    return (np.tanh(X)*gh_w).sum(1)
# prediction by quadrature
def pred():
    ss = np.linspace(1e-4, 25, 2500); vals = []
    ux, uw = gh_x, gh_w
    for s in ss:
        vu, ciu, vi = moments(s); u = np.sqrt(vu)*ux
        m, sd = cond_I(s, u); I = m[:, None] + sd*gh_x[None, :]; X = I*u[:, None]/(lam+nu)
        EphiX = (np.tanh(X)*X*gh_w).sum(1); Ephi = (np.tanh(X)*gh_w).sum(1); EX = (X*gh_w).sum(1)
        vals.append(np.exp(-nu*s)*np.sum(uw*(EphiX - Ephi*EX)))
    return 2*np.trapezoid(vals, ss)
def mc(eps, N, dt, seed):
    rng = np.random.default_rng(seed)
    a = np.exp(-lam*dt); sd = sig*np.sqrt(-np.expm1(-2*lam*dt)/(2*lam))
    U = np.zeros(N); I = np.zeros(N); alive = np.ones(N, bool); I2 = np.zeros(N); t = 0.0
    # common random numbers: same stream for every eps
    while alive.any():
        idx = np.flatnonzero(alive)
        Z = rng.standard_normal(N)[idx]; V = rng.random(N)[idx]
        u = U[idx]
        if t > 0:
            psi = -np.tanh(I[idx]*u/(lam+nu)) + Ephi_given_u(t, u)
        else:
            psi = np.zeros(idx.size)
        H = nu + eps*psi
        done = V < H*dt
        I2[idx[done]] = I[idx[done]]**2; alive[idx[done]] = False
        keep = idx[~done]
        un = U[keep]*a + sd*Z[~done]
        I[keep] += 0.5*(U[keep] + un)*dt; U[keep] = un
        t += dt
    return I2
t0 = time.time(); P = pred(); print(f"prediction d/deps E[I^2] = {P:.6f}  ({time.time()-t0:.1f}s)", flush=True)
N, dt, eps = 400000, 0.005, 0.1
r = {}
for e in (+eps, -eps, 0.0):
    t0 = time.time(); r[e] = mc(e, N, dt, 7); print(f"eps={e:+.2f}: E[I^2]={r[e].mean():.6f} ({time.time()-t0:.0f}s)", flush=True)
d = (r[eps] - r[-eps])/(2*eps)
print(f"MC derivative (common random numbers) = {d.mean():.6f} +- {d.std()/np.sqrt(N):.6f};  prediction {P:.6f}")

out = dict(lam=lam, nu=nu, sig=sig, eps=eps, N=N, dt=dt, seed=7, prediction=float(P),
           mc_derivative=float(d.mean()), mc_se=float(d.std()/np.sqrt(N)),
           E_I2_base=float(r[0.0].mean()), E_I2_base_se=float(r[0.0].std()/np.sqrt(N)),
           E_I2_base_exact=2*sig**2/(nu*(nu+lam)*(nu+2*lam)))
Path(__file__).resolve().parents[2].joinpath("data", "validation").mkdir(parents=True, exist_ok=True)
json.dump(out, open(Path(__file__).resolve().parents[2] / "data" / "validation" / "perturbation.json", "w"), indent=1)

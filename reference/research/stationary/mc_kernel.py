"""Independent check of k(t) by a coupled Monte Carlo (Euler, same conventions as
response_identity_mc_se.py): a stationary population is displaced by +eps and -eps at t = 0, the
three copies share every Gaussian increment and every uniform used for the completion decisions,
and k(t) ~ (mean_+ - mean_-) / (2 eps)."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, json, sys, time
from fvsolver import activation, Reset


def run(theta, eta, nu, lam=1.0, sig=1.0, gam=1.0, reset='pl', c=None, N=400_000, dt=0.01,
        burn=20.0, T=5.0, eps=0.05, seed=1, rec=0.05):
    c = theta / 2 if c is None else c
    rng = np.random.default_rng(seed)
    act = lambda u: activation(u, theta)
    R = Reset(reset, c)
    U = rng.normal(0, 0.5, N); A = np.zeros(N)
    for _ in range(int(burn / dt)):
        H = nu + eta * act(U) * (-np.expm1(-gam * A))
        jump = rng.random(N) < H * dt
        U = np.where(jump, R(U), U - lam * U * dt + sig * np.sqrt(dt) * rng.standard_normal(N))
        A = np.where(jump, 0.0, A + act(U) * dt)
    st = {s: [U + s * eps, A.copy()] for s in (-1, 1)}
    ts, ks, ses = [0.0], [1.0], [0.0]
    every = int(round(rec / dt))
    for n in range(int(round(T / dt))):
        Z = rng.standard_normal(N); V = rng.random(N)
        for s, (u, a) in st.items():
            H = nu + eta * act(u) * (-np.expm1(-gam * a))
            jump = V < H * dt
            un = np.where(jump, R(u), u - lam * u * dt + sig * np.sqrt(dt) * Z)
            an = np.where(jump, 0.0, a + act(un) * dt)
            st[s] = [un, an]
        if (n + 1) % every == 0:
            d = (st[1][0] - st[-1][0]) / (2 * eps)
            ts.append((n + 1) * dt); ks.append(d.mean()); ses.append(d.std() / np.sqrt(N))
    return np.array(ts), np.array(ks), np.array(ses)


if __name__ == '__main__':
    NU06 = 0.7136347625525
    cases = {'V6_Rpl': dict(theta=0.7, eta=0.6, nu=NU06, reset='pl'),
             'BENCH_R0': dict(theta=0.7, eta=0.6, nu=NU06, reset='0'),
             'X_R0_th035_eta20': dict(theta=0.35, eta=20.0, nu=0.2, reset='0', lam=0.25)}
    out = {}
    for name in sys.argv[1:] or cases:
        t0 = time.time()
        ts, ks, se = run(**cases[name], seed=7)
        out[name] = dict(params=cases[name], t=ts.tolist(), k=ks.tolist(), se=se.tolist())
        print(name, f"({time.time() - t0:.0f}s)")
        for i in range(0, len(ts), 4):
            print(f"  t={ts[i]:5.2f} k={ks[i]:+.4f} +- {se[i]:.4f}")
        json.dump(out, open('out/mc_kernel.json', 'w'), indent=1)

"""A Monte Carlo check of the running economy's distribution of processing at consensus (pausing clock, full
resets, the parameters of the paper's Example 1), against the engine's exact share of agents with no processing.

    python3 tests_ref/check_processing.py [output_dir]

50,000 agents from the reference, exact Ornstein-Uhlenbeck steps of 0.001 for 10 time units, processing integrated
by the trapezoid rule. An exit from the tolerance band between two steps is drawn with the Brownian-bridge
crossing probability, so that the share with no processing is not overstated by discrete monitoring. About two
minutes. Exit status 1 if the simulated share differs from the engine's by more than four standard errors.
"""
import json, os, sys
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "output"
meta = json.load(open(os.path.join(OUT, "data", "extras", "stationary_meta_full.json")))
lam, sig, nu, eta, gam, theta = 1.0, 1.0, 1.0, 5.0, 1.0, meta["theta"]


def s(z):
    z = np.clip(z, 1e-12, 1 - 1e-12)
    a, b = np.exp(-1 / z), np.exp(-1 / (1 - z))
    return a / (a + b)


def act(u):                      # alpha = chi = s((|u| - theta) / theta)
    z = (np.abs(u) - theta) / theta
    return np.where(z <= 0, 0.0, np.where(z >= 1, 1.0, s(z)))


rng = np.random.default_rng(7)
N, dt, T = 50_000, 0.001, 10.0
ea, sd = np.exp(-lam * dt), sig * np.sqrt(-np.expm1(-2 * lam * dt) / (2 * lam))
u, A, none = np.zeros(N), np.zeros(N), np.ones(N, dtype=bool)
for _ in range(int(T / dt)):
    H = nu + eta * act(u) * (1 - np.exp(-gam * A))
    done = rng.random(N) < -np.expm1(-H * dt)
    un = u * ea + sd * rng.standard_normal(N)
    inside = (np.abs(u) <= theta) & (np.abs(un) <= theta)
    pu = np.exp(-2 * (theta - u) * (theta - un) / (sig ** 2 * dt))
    pl = np.exp(-2 * (theta + u) * (theta + un) / (sig ** 2 * dt))
    crossed = (~inside) | (rng.random(N) < np.where(inside, pu + pl - pu * pl, 0.0))
    A = A + 0.5 * (act(u) + act(un)) * dt
    none = none & ~crossed
    u, A, none = np.where(done, 0.0, un), np.where(done, 0.0, A), np.where(done, True, none)
p0 = none.mean()
se = np.sqrt(p0 * (1 - p0) / N)
exact = meta["no_processing_share"]
print(f"no processing: simulated {p0:.4f} +- {se:.4f}, exact {exact:.4f} (z = {(p0 - exact) / se:.1f})")
print(f"processing below 0.1: simulated {(A < 0.1).mean():.4f}, solver {meta['share_below_0.1']:.4f}")
sys.exit(0 if abs(p0 - exact) < 4 * se else 1)
